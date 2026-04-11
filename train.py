"""
train.py
========
PAN 2026 - Voight-Kampff AI Detection
Fine-tuning completo de StyleDistance/mStyleDistance

Uso:
    python train.py [--train_path PATH] [--val_path PATH] [--output_dir PATH]
                    [--epochs N] [--batch_size N] [--lr FLOAT] [--seed N]

    Ejemplo local:
        python train.py --train_path TandVDatasets/train.jsonl \
                        --val_path TandVDatasets/val.jsonl

    Ejemplo en Colab (Drive):
        python train.py \
            --train_path /content/drive/MyDrive/PAN2026/dataset/train.jsonl \
            --val_path   /content/drive/MyDrive/PAN2026/dataset/val.jsonl \
            --output_dir /content/drive/MyDrive/PAN2026/full_finetuning/results_v2/
"""

import os
import gc
import json
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import get_linear_schedule_with_warmup
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm
import zipfile

warnings.filterwarnings('ignore')

# Importar módulo propio
from data_loader import (
    preprocess_text,
    AIDetectionDataset,
    augment_genre_human,
    compute_genre_stratified_weights,
)


# =============================================================================
# CONFIGURACIÓN POR DEFECTO (sobreescribible por argumentos CLI)
# =============================================================================

DEFAULT_CONFIG = {
    # === PATHS ===
    'TRAIN_PATH': 'TandVDatasets/train.jsonl',
    'VAL_PATH':   'TandVDatasets/val.jsonl',
    'OUTPUT_DIR': 'output/',

    # === MODELO ===
    'MODEL_NAME': 'StyleDistance/mStyleDistance',

    # === HIPERPARÁMETROS ===
    'MAX_LENGTH':       192,
    'BATCH_SIZE':       16,
    'LEARNING_RATE':    1e-5,
    'EPOCHS':           6,
    'WARMUP_RATIO':     0.1,
    'DROPOUT':          0.3,
    'WEIGHT_DECAY':     0.01,
    'GRADIENT_CLIP':    1.0,
    'GRAD_ACCUM_STEPS': 4,
    'LABEL_SMOOTHING':  0.1,

    # === EARLY STOPPING ===
    'PATIENCE':   2,
    'MIN_DELTA':  0.001,

    # === BALANCEO ===
    'BALANCING_METHOD': 'genre_stratified_weights',

    # === DATA AUGMENTATION ===
    'USE_AUGMENTATION':  True,
    'AUG_RATIO':         0.15,
    'AUG_RATIO_NEWS':    0.25,
    'AUG_GENRES':        ['essays', 'news'],
    'AUG_TECHNIQUES':    ['delete', 'sentence_shuffle', 'truncate'],

    # === THRESHOLD ===
    'OPTIMIZE_THRESHOLD': True,
    'THRESHOLD_METRIC':   'pan_mean',

    # === REPRODUCIBILIDAD ===
    'SEED': 42,

    # === MIXED PRECISION ===
    # mStyleDistance usa BFloat16 internamente; GradScaler no lo soporta en T4.
    # En A100/H100 con bfloat16 nativo, puede ponerse True.
    'USE_AMP': False,
}


# =============================================================================
# REPRODUCIBILIDAD
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# =============================================================================
# MODELO: StyleAIClassifier
# =============================================================================

class StyleAIClassifier(nn.Module):
    """
    Modelo completo para detección de AI:
    - Encoder: StyleDistance/mStyleDistance (fine-tuned end-to-end)
    - Cabeza: Clasificador lineal con dropout

    Args:
        encoder_model: SentenceTransformer pre-entrenado
        num_classes (int): Número de clases (2 para binario)
        dropout (float): Tasa de dropout
        hidden_dim (int | None): Dim. de capa oculta (None = sin capa oculta)
    """

    def __init__(self, encoder_model, num_classes: int = 2,
                 dropout: float = 0.3, hidden_dim=None):
        super(StyleAIClassifier, self).__init__()
        self.encoder = encoder_model
        self.embedding_dim = encoder_model.get_sentence_embedding_dimension()

        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.embedding_dim, num_classes)
            )

    def forward(self, input_ids, attention_mask):
        features = {'input_ids': input_ids, 'attention_mask': attention_mask}
        embeddings = self.encoder(features)['sentence_embedding']
        embeddings = embeddings.to(torch.float32)
        return self.classifier(embeddings)


# =============================================================================
# TRAINING & EVALUACIÓN
# =============================================================================

def train_epoch(model, loader, optimizer, scheduler, criterion,
                scaler, device, epoch, config):
    """Entrena el modelo por una época con gradient accumulation."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    accum_steps = config['GRAD_ACCUM_STEPS']

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch+1}/{config['EPOCHS']} [TRAIN]",
        leave=False,
        dynamic_ncols=True
    )

    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids      = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels         = batch['labels'].to(device, non_blocking=True)

        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels) / accum_steps
        loss.backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['GRADIENT_CLIP'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if (step + 1) % (accum_steps * 50) == 0:
                torch.cuda.empty_cache()

        total_loss += loss.item() * accum_steps
        preds = torch.argmax(logits.detach(), dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({
            'loss': f"{loss.item() * accum_steps:.4f}",
            'lr':   f"{scheduler.get_last_lr()[0]:.2e}"
        })

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def compute_pan_metrics(y_true, y_prob, threshold=0.5, margin=0.0):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    N = len(y_true)
    if N == 0:
        return {'brier':0,'roc_auc':0,'c_at_1':0,'f1':0,'f05_u':0,'pan_mean':0}

    # 1. Brier Score Complement (MSE)
    brier = np.mean((y_prob - y_true)**2)
    brier_comp = 1.0 - brier

    # 2. ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    # Clasificación dura respetando abstenciones
    is_abstain = (y_prob >= threshold - margin) & (y_prob <= threshold + margin)
    is_pos = (y_prob > threshold + margin)
    is_neg = (y_prob < threshold - margin)

    y_pred_hard = np.zeros(N)
    y_pred_hard[is_pos] = 1
    y_pred_hard[is_neg] = 0
    y_pred_hard[is_abstain] = 0.5  # abstención/no-respuesta

    # 3. C@1
    n_u = np.sum(is_abstain)
    n_corr = np.sum((y_pred_hard == y_true) & (~is_abstain))
    c_at_1 = (n_corr + (n_corr / float(N)) * n_u) / float(N)

    # 4. F1 and F0.5u
    TP = np.sum((y_pred_hard == 1) & (y_true == 1))
    FP = np.sum((y_pred_hard == 1) & (y_true == 0))
    FN = np.sum((y_pred_hard != 1) & (y_true == 1))
    
    # Non-answers (0.5) count as False Negatives for F0.5u
    FN_u = FN + np.sum(is_abstain & (y_true == 0))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    recall_u = TP / (TP + FN_u) if (TP + FN_u) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    f05_u = (1.25 * precision * recall_u) / (0.25 * precision + recall_u) if (0.25 * precision + recall_u) > 0 else 0.0

    pan_mean = np.mean([brier_comp, roc_auc, c_at_1, f1, f05_u])

    return {
        'brier': brier_comp,
        'roc_auc': roc_auc,
        'c_at_1': c_at_1,
        'f1': f1,
        'f05_u': f05_u,
        'pan_mean': pan_mean
    }


def evaluate_model(model, loader, criterion, device, return_predictions: bool = False):
    """
    Evalúa el modelo en un DataLoader y calcula las métricas matemáticas de PAN 2026.
    """
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating", leave=False, dynamic_ncols=True):
            input_ids      = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels         = batch['labels'].to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            probs  = torch.softmax(logits, dim=1)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    
    # Calcular métricas PAN con umbral neutral (0.5) sin abstención para rastreo por época
    pan_metrics = compute_pan_metrics(all_labels, all_probs, threshold=0.5, margin=0.0)
    
    pan_metrics['loss'] = avg_loss
    pan_metrics['accuracy'] = accuracy_score(all_labels, all_preds)
    pan_metrics['f1_macro'] = f1_score(all_labels, all_preds, average='macro')
    pan_metrics['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted')

    if return_predictions:
        return pan_metrics, {
            'labels': all_labels,
            'predictions': all_preds,
            'probabilities': all_probs
        }
    return pan_metrics


# =============================================================================
# GUARDAR MODELO EN FORMATO HUGGING FACE
# =============================================================================

def save_hf_format(model, tokenizer, output_dir: str):
    """
    Guarda el encoder y tokenizer en formato HuggingFace estándar.

    Archivos generados:
        config.json, model.safetensors (o pytorch_model.bin),
        tokenizer.json, tokenizer_config.json, training_args.bin (placeholder)
    """
    hf_dir = os.path.join(output_dir, 'hf_model')
    os.makedirs(hf_dir, exist_ok=True)

    print(f"\n💾 Guardando modelo en formato HuggingFace → {hf_dir}")

    # Guardar el encoder (SentenceTransformer)
    model.encoder.save(hf_dir)

    # Guardar el tokenizer por separado (accesible desde AutoTokenizer)
    tokenizer.save_pretrained(hf_dir)

    # Crear un training_args.bin placeholder (requerido por algunas plataformas)
    training_args_path = os.path.join(hf_dir, 'training_args.bin')
    if not os.path.exists(training_args_path):
        torch.save({'placeholder': True}, training_args_path)

    print("   ✅ Archivos generados:")
    for f in sorted(os.listdir(hf_dir)):
        fpath = os.path.join(hf_dir, f)
        size_mb = os.path.getsize(fpath) / 1e6
        print(f"      {f}  ({size_mb:.1f} MB)")

    return hf_dir


# =============================================================================
# DESCARGA DE ARCHIVOS EN COLAB
# =============================================================================

def download_model_files_colab(hf_dir: str):
    """
    Crea un ZIP con los archivos del modelo HuggingFace y lo descarga
    automáticamente si se está ejecutando en Google Colab.

    Archivos incluidos: config.json, model.safetensors (o pytorch_model.bin),
                        tokenizer.json, tokenizer_config.json, training_args.bin

    En entornos no-Colab simplemente guarda el ZIP en el directorio padre.
    """
    target_files = [
        'config.json',
        'model.safetensors',
        'pytorch_model.bin',   # fallback si safetensors no se generó
        'tokenizer.json',
        'tokenizer_config.json',
        'tokenizer.model',
        'special_tokens_map.json',
        'training_args.bin',
    ]

    zip_path = os.path.join(os.path.dirname(hf_dir), 'model_files.zip')

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in target_files:
            fpath = os.path.join(hf_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, fname)
                print(f"   📦 Añadido al ZIP: {fname}")

    print(f"\n✅ ZIP creado: {zip_path}")

    # Intentar descarga automática en Colab
    try:
        from google.colab import files
        print("🚀 Iniciando descarga en Colab...")
        files.download(zip_path)
    except ImportError:
        print(f"ℹ️  No estás en Colab. El ZIP está guardado en:\n   {zip_path}")
    except Exception as e:
        print(f"⚠️  No se pudo descargar automáticamente: {e}")
        print(f"   Descarga manualmente desde: {zip_path}")


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def main(config: dict):
    set_seed(config['SEED'])
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 DISPOSITIVO: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"{'='*60}\n")

    # --- Directorios de salida ---
    for subdir in ['models', 'plots', 'predictions', 'logs']:
        os.makedirs(os.path.join(config['OUTPUT_DIR'], subdir), exist_ok=True)

    with open(os.path.join(config['OUTPUT_DIR'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # -------------------------------------------------------------------------
    # CARGA Y PREPROCESAMIENTO DE DATOS
    # -------------------------------------------------------------------------
    print(f"📥 Cargando datasets...")
    print(f"   Train: {config['TRAIN_PATH']}")
    print(f"   Val:   {config['VAL_PATH']}\n")

    df_train = pd.read_json(config['TRAIN_PATH'], orient='records', lines=True)
    df_val   = pd.read_json(config['VAL_PATH'],   orient='records', lines=True)

    assert 'label' in df_train.columns, "ERROR: No existe columna 'label' en train"
    assert 'label' in df_val.columns,   "ERROR: No existe columna 'label' en val"
    assert 'text'  in df_train.columns, "ERROR: No existe columna 'text' en train"
    assert 'text'  in df_val.columns,   "ERROR: No existe columna 'text' en val"

    print(f"✅ Train: {len(df_train):,} | Val: {len(df_val):,}")

    # Preprocesamiento
    print("🔧 Preprocesando textos...")
    df_train['text'] = df_train['text'].apply(preprocess_text)
    df_val['text']   = df_val['text'].apply(preprocess_text)
    df_train = df_train[df_train['text'].str.len() > 0].reset_index(drop=True)
    df_val   = df_val[df_val['text'].str.len() > 0].reset_index(drop=True)
    print(f"✅ Train: {len(df_train):,} | Val: {len(df_val):,} (post limpieza)")

    # -------------------------------------------------------------------------
    # DISTRIBUCIÓN Y BALANCEO
    # -------------------------------------------------------------------------
    train_dist = df_train['label'].value_counts().sort_index()
    val_dist   = df_val['label'].value_counts().sort_index()

    class_weights_vals = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df_train['label']),
        y=df_train['label']
    )
    class_weights_tensor = torch.FloatTensor(class_weights_vals).to(device)

    # -------------------------------------------------------------------------
    # DATA AUGMENTATION
    # -------------------------------------------------------------------------
    if config['USE_AUGMENTATION']:
        print('\n🔄 Aplicando data augmentation...')
        df_train = augment_genre_human(
            df_train,
            target_genres=config['AUG_GENRES'],
            ratio=config['AUG_RATIO'],
            ratio_news=config['AUG_RATIO_NEWS'],
            techniques=config['AUG_TECHNIQUES'],
            seed=config['SEED']
        )
        print(f'✅ Train tras augmentation: {len(df_train):,}')

    sample_weights = compute_genre_stratified_weights(df_train)

    # -------------------------------------------------------------------------
    # MODELO Y TOKENIZER
    # -------------------------------------------------------------------------
    print(f"\n📦 Cargando modelo base: {config['MODEL_NAME']}...")
    torch.cuda.empty_cache(); gc.collect()

    encoder = SentenceTransformer(config['MODEL_NAME'])
    encoder.to(device)

    try:
        encoder[0].auto_model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing activado")
    except Exception as e:
        print(f"⚠️  Gradient checkpointing no disponible: {e}")

    tokenizer = encoder.tokenizer

    model = StyleAIClassifier(
        encoder_model=encoder,
        num_classes=2,
        dropout=config['DROPOUT'],
        hidden_dim=None
    )
    model = model.float()
    model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parámetros totales: {total_params:,} | Entrenables: {trainable_params:,}")

    # -------------------------------------------------------------------------
    # DATASETS Y DATALOADERS
    # -------------------------------------------------------------------------
    print("\n🔨 Creando datasets y DataLoaders...")

    train_dataset = AIDetectionDataset(
        texts=df_train['text'].tolist(),
        labels=df_train['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config['MAX_LENGTH']
    )
    val_dataset = AIDetectionDataset(
        texts=df_val['text'].tolist(),
        labels=df_val['label'].tolist(),
        tokenizer=tokenizer,
        max_length=config['MAX_LENGTH']
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config['BATCH_SIZE'],
        sampler=sampler, shuffle=False,
        num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['BATCH_SIZE'],
        shuffle=False, num_workers=2, pin_memory=True,
        persistent_workers=True, prefetch_factor=2
    )

    print(f"✅ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # -------------------------------------------------------------------------
    # OPTIMIZER, SCHEDULER Y LOSS
    # -------------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['LEARNING_RATE'],
        weight_decay=config['WEIGHT_DECAY'],
        eps=1e-8
    )

    num_training_steps = len(train_loader) * config['EPOCHS']
    num_warmup_steps   = int(num_training_steps * config['WARMUP_RATIO'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=config['LABEL_SMOOTHING']
    )

    print(f"✅ Optimizer: AdamW (LR={config['LEARNING_RATE']}, WD={config['WEIGHT_DECAY']})")
    print(f"   Total steps: {num_training_steps:,} | Warmup: {num_warmup_steps:,}")

    # -------------------------------------------------------------------------
    # TRAINING LOOP CON EARLY STOPPING
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("🚀 INICIANDO ENTRENAMIENTO")
    print("="*70)

    best_val_metric   = 0
    best_epoch        = 0
    patience_counter  = 0
    history = {k: [] for k in [
        'train_loss', 'train_acc', 'val_loss', 'val_acc',
        'val_f1_macro', 'val_auc_roc', 'val_pan_mean'
    ]}

    log_path = os.path.join(config['OUTPUT_DIR'], 'logs', 'training_log.txt')
    log_file = open(log_path, 'w')
    log_file.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_F1,Val_AUC,Val_PAN_Mean\n")

    for epoch in range(config['EPOCHS']):
        print(f"\n{'='*70}\nEpoch {epoch+1}/{config['EPOCHS']}\n{'='*70}")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, None, device, epoch, config
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])
        history['val_auc_roc'].append(val_metrics['roc_auc'])
        history['val_pan_mean'].append(val_metrics['pan_mean'])

        log_file.write(
            f"{epoch+1},{train_loss:.4f},{train_acc:.4f},"
            f"{val_metrics['loss']:.4f},{val_metrics['accuracy']:.4f},"
            f"{val_metrics['f1_macro']:.4f},{val_metrics['roc_auc']:.4f},"
            f"{val_metrics['pan_mean']:.4f}\n"
        )
        log_file.flush()

        print(f"\n📊 Resultados Epoch {epoch+1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_metrics['loss']:.4f} | Val Acc:  {val_metrics['accuracy']:.4f}")
        print(f"   Val F1-Macro: {val_metrics['f1_macro']:.4f} | Val PAN Mean: {val_metrics['pan_mean']:.4f}")

        # Guardar mejor modelo observando el PAN Mean (o f1_macro como fallback)
        target_metric = val_metrics[config['THRESHOLD_METRIC']]
        if target_metric > best_val_metric + config['MIN_DELTA']:
            best_val_metric = target_metric
            best_epoch        = epoch
            patience_counter  = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_metric': best_val_metric,
                'config': config
            }, os.path.join(config['OUTPUT_DIR'], 'models', 'best_model.pt'))

            print(f"   ✅ Nuevo mejor modelo guardado (PAN Mean: {best_val_metric:.4f})")
        else:
            patience_counter += 1
            print(f"   ⏳ Patience: {patience_counter}/{config['PATIENCE']}")

        if patience_counter >= config['PATIENCE']:
            print(f"\n🛑 Early stopping en época {epoch+1}")
            print(f"   Mejor PAN Mean: {best_val_metric:.4f} (Época {best_epoch+1})")
            break

        if epoch % 2 == 0:
            torch.cuda.empty_cache(); gc.collect()

    log_file.close()

    # Guardar último checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'final_val_metric': val_metrics[config['THRESHOLD_METRIC']],
        'config': config
    }, os.path.join(config['OUTPUT_DIR'], 'models', 'last_model.pt'))

    with open(os.path.join(config['OUTPUT_DIR'], 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # -------------------------------------------------------------------------
    # EVALUACIÓN FINAL CON MEJOR MODELO
    # -------------------------------------------------------------------------
    print("\n📦 Cargando mejor modelo para evaluación final...")
    checkpoint = torch.load(
        os.path.join(config['OUTPUT_DIR'], 'models', 'best_model.pt'),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    val_metrics, val_predictions = evaluate_model(
        model, val_loader, criterion, device, return_predictions=True
    )

    print("\n" + "="*70)
    print("MÉTRICAS FINALES EN VALIDATION")
    print("="*70)
    for k in ['pan_mean', 'brier', 'c_at_1', 'f05_u', 'roc_auc', 'f1_macro']:
        if k in val_metrics:
            print(f"   {k:20s}: {val_metrics[k]:.4f}")

    print('\n' + classification_report(
        val_predictions['labels'],
        val_predictions['predictions'],
        target_names=['Human (0)', 'AI-Generated (1)'],
        digits=4
    ))

    # -------------------------------------------------------------------------
    # THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    best_threshold     = 0.5
    best_margin        = 0.0
    best_metric_value  = 0
    best_pan_metrics   = None

    if config['OPTIMIZE_THRESHOLD']:
        print("\n🔍 Explorando Threshold y Abstention Margin...")
        for thr in np.arange(0.1, 0.91, 0.02):
            for margin in np.arange(0.0, min(thr, 1.0-thr) + 0.01, 0.02):
                m = compute_pan_metrics(
                    val_predictions['labels'], 
                    val_predictions['probabilities'], 
                    threshold=thr, 
                    margin=margin
                )
                if m['pan_mean'] > best_metric_value:
                    best_metric_value = m['pan_mean']
                    best_threshold    = thr
                    best_margin       = margin
                    best_pan_metrics  = m

        print(f"\n🎯 Threshold óptimo: {best_threshold:.3f} ± {best_margin:.3f} margen")
        print(f"   PAN Mean optimizado: {best_metric_value:.4f}")
        if best_pan_metrics:
            print(f"      - Brier score:   {best_pan_metrics['brier']:.4f}")
            print(f"      - C@1 score:     {best_pan_metrics['c_at_1']:.4f}")
            print(f"      - F0.5u score:   {best_pan_metrics['f05_u']:.4f}")

        with open(os.path.join(config['OUTPUT_DIR'], 'models', 'threshold_config.json'), 'w') as f:
            json.dump({
                'best_threshold':            float(best_threshold),
                'best_margin':               float(best_margin),
                'metric_optimized':          config['THRESHOLD_METRIC'],
                'metric_value':              float(best_metric_value),
                'brier_comp':                float(best_pan_metrics['brier']),
                'c_at_1':                    float(best_pan_metrics['c_at_1']),
                'f05_u':                     float(best_pan_metrics['f05_u']),
            }, f, indent=2)

    # -------------------------------------------------------------------------
    # PREDICCIONES FORMATO TIRA
    # -------------------------------------------------------------------------
    probs = np.array(val_predictions['probabilities'])
    is_abstain = (probs >= best_threshold - best_margin) & (probs <= best_threshold + best_margin)
    val_preds_optimal = (probs >= best_threshold).astype(float)
    val_preds_optimal[is_abstain] = 0.5

    predictions_df = pd.DataFrame({
        'id': df_val['id'].tolist() if 'id' in df_val.columns
              else [f"val_{i}" for i in range(len(df_val))],
        'label': val_predictions['probabilities'],
        'pred_binary': val_preds_optimal,
        'true_label':  val_predictions['labels']
    })

    tira_pred = predictions_df[['id', 'label']].copy()
    tira_pred.to_json(
        os.path.join(config['OUTPUT_DIR'], 'predictions', 'val_predictions.jsonl'),
        orient='records', lines=True
    )
    print(f"✅ Predicciones TIRA guardadas ({len(tira_pred):,} filas)")

    # -------------------------------------------------------------------------
    # GUARDAR EN FORMATO HUGGING FACE + DESCARGA COLAB
    # -------------------------------------------------------------------------
    hf_dir = save_hf_format(model, tokenizer, config['OUTPUT_DIR'])
    download_model_files_colab(hf_dir)

    # -------------------------------------------------------------------------
    # RESUMEN FINAL
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("🎉 RESUMEN FINAL - PAN 2026 VOIGHT-KAMPFF")
    print("="*70)
    print(f"   Modelo base:       {config['MODEL_NAME']}")
    print(f"   Épocas totales:    {epoch+1}")
    print(f"   Mejor época:       {best_epoch+1}")
    print(f"   Val F1-Macro:      {val_metrics['f1_macro']:.4f}")
    print(f"   Val PAN Mean:      {val_metrics.get('pan_mean', 0):.4f}")
    print(f"   Val AUC-ROC:       {val_metrics['roc_auc']:.4f}")
    print(f"   Threshold óptimo:  {best_threshold:.3f} ± {best_margin:.3f}")
    print(f"   Output:            {config['OUTPUT_DIR']}")
    print("="*70)
    print("✅ ¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("="*70)


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='PAN 2026 - Fine-tuning mStyleDistance para detección de AI'
    )
    parser.add_argument(
        '--train_path', type=str,
        default=DEFAULT_CONFIG['TRAIN_PATH'],
        help='Ruta al archivo train.jsonl '
             '(default: TandVDatasets/train.jsonl)'
    )
    parser.add_argument(
        '--val_path', type=str,
        default=DEFAULT_CONFIG['VAL_PATH'],
        help='Ruta al archivo val.jsonl '
             '(default: TandVDatasets/val.jsonl)'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=DEFAULT_CONFIG['OUTPUT_DIR'],
        help='Directorio de salida (default: output/)'
    )
    parser.add_argument(
        '--model_name', type=str,
        default=DEFAULT_CONFIG['MODEL_NAME'],
        help='Nombre del modelo HuggingFace (default: StyleDistance/mStyleDistance)'
    )
    parser.add_argument(
        '--epochs', type=int,
        default=DEFAULT_CONFIG['EPOCHS'],
        help='Número máximo de épocas (default: 6)'
    )
    parser.add_argument(
        '--batch_size', type=int,
        default=DEFAULT_CONFIG['BATCH_SIZE'],
        help='Tamaño de batch (default: 16)'
    )
    parser.add_argument(
        '--lr', type=float,
        default=DEFAULT_CONFIG['LEARNING_RATE'],
        help='Learning rate (default: 1e-5)'
    )
    parser.add_argument(
        '--max_length', type=int,
        default=DEFAULT_CONFIG['MAX_LENGTH'],
        help='Longitud máxima de tokens (default: 192)'
    )
    parser.add_argument(
        '--seed', type=int,
        default=DEFAULT_CONFIG['SEED'],
        help='Semilla de reproducibilidad (default: 42)'
    )
    parser.add_argument(
        '--no_augmentation', action='store_true',
        help='Deshabilitar data augmentation'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    config = dict(DEFAULT_CONFIG)  # Copia del config por defecto
    config['TRAIN_PATH']      = args.train_path
    config['VAL_PATH']        = args.val_path
    config['OUTPUT_DIR']      = args.output_dir
    config['MODEL_NAME']      = args.model_name
    config['EPOCHS']          = args.epochs
    config['BATCH_SIZE']      = args.batch_size
    config['LEARNING_RATE']   = args.lr
    config['MAX_LENGTH']      = args.max_length
    config['SEED']            = args.seed
    config['USE_AUGMENTATION'] = not args.no_augmentation

    print("\n📋 CONFIGURACIÓN:")
    print(f"   Train: {config['TRAIN_PATH']}")
    print(f"   Val:   {config['VAL_PATH']}")
    print(f"   Output: {config['OUTPUT_DIR']}")
    print(f"   Épocas: {config['EPOCHS']} | Batch: {config['BATCH_SIZE']} | LR: {config['LEARNING_RATE']}")
    print(f"   Augmentation: {'✅' if config['USE_AUGMENTATION'] else '❌'}")

    main(config)
