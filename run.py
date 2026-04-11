"""
run.py
======
PAN 2026 - Voight-Kampff AI Detection
Script de inferencia compatible con la plataforma TIRA.

Uso:
    python run.py -i $inputDataset -o $outputDir [--model_path ./models]
"""

import os
import json
import argparse
import warnings
import torch
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')
from data_loader import preprocess_text
from train import StyleAIClassifier

def load_custom_model(model_path: str, device: torch.device):
    """
    Carga el modelo V2 (StyleAIClassifierV2) sincronizado con el entrenamiento,
    forzando la precisión Float32 para evitar errores de mismatch de tipos.
    """
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    import os

    print(f"Iniciando carga de configuración y arquitectura desde {model_path}...")
    
    # 1. Cargamos la configuración base (esto no lee los archivos de pesos pesados)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    
    # 2. Definición de la arquitectura exacta V2
    class StyleAIClassifierV2(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # Cargamos la estructura vacía basada en la configuración
            # Esto evita errores si el archivo .safetensors original no está o es corrupto
            self.encoder = AutoModel.from_config(cfg)
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # Usamos el token [CLS] (índice 0) igual que en tu entrenamiento en la Universidad de Guayaquil
            return self.classifier(self.dropout(outputs.last_hidden_state[:, 0, :]))

    # 3. Instanciar el modelo vacío y el tokenizador
    model = StyleAIClassifierV2(config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 4. Carga manual de tus pesos entrenados (best_model.pt)
    pt_path = os.path.join(model_path, 'best_model.pt')
    if os.path.exists(pt_path):
        print(f"Inyectando pesos desde {pt_path}...")
        # Cargamos el state_dict (weights_only=False por compatibilidad con archivos de Colab/Kaggle)
        checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # --- CAMBIO CRÍTICO PARA EL ERROR DE DTYPE ---
        # Convertimos todo el modelo a Float32. Esto asegura que tanto el encoder 
        # como la capa classifier usen el mismo tipo de dato, eliminando el error de BFloat16.
        model = model.float() 
        # ---------------------------------------------
        
        print("✅ Modelo V2 cargado y convertido a Float32 exitosamente.")
    else:
        raise FileNotFoundError(f"❌ ERROR: No se encontró el archivo de pesos en {pt_path}")
        
    model.to(device).eval()
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia PAN 2026 - Versión Híbrida (Docker + Local)")
    
    # 1. Argumentos con banderas (Prioridad para TIRA Docker)
    parser.add_argument("-i", "--input", type=str, help="Carpeta de entrada (usado por TIRA Docker)")
    parser.add_argument("-o", "--output", type=str, help="Carpeta de salida (usado por TIRA Docker)")
    
    # 2. Argumentos posicionales (Compatibilidad con Upload Run y manual)
    # nargs='?' permite que sean opcionales si ya se pasaron las banderas -i/-o
    parser.add_argument("input_pos", type=str, nargs='?', help="Ruta de entrada manual")
    parser.add_argument("output_pos", type=str, nargs='?', help="Ruta de salida manual")
    
    parser.add_argument("--model_path", type=str, default="/app/models", help="Ruta al modelo")
    
    args = parser.parse_args()
    
    # 3. Lógica de selección de rutas: si no hay banderas, usa los posicionales
    raw_input = args.input if args.input else args.input_pos
    final_output_dir = args.output if args.output else args.output_pos

    if not raw_input or not final_output_dir:
        print("❌ ERROR: Debes proporcionar entrada y salida.")
        parser.print_help()
        exit(1)

    # 4. Normalización de la entrada: ¿Es una carpeta o un archivo directo?
    # TIRA suele pasar una carpeta en Docker, pero un archivo en manual.
    if os.path.isdir(raw_input):
        input_file_path = os.path.join(raw_input, 'dataset.jsonl')
    else:
        input_file_path = raw_input

    # Configuración de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Usando dispositivo: {device}")

    # Cargar modelo y configuración de abstención
    model, tokenizer = load_custom_model(args.model_path, device)
    
    threshold, margin = 0.5, 0.0
    thr_path = os.path.join(args.model_path, 'threshold_config.json')
    if os.path.exists(thr_path):
        try:
            with open(thr_path, 'r') as f:
                cfg = json.load(f)
                threshold = cfg.get('best_threshold', 0.5)
                margin = cfg.get('best_margin', 0.0)
            print(f"🎯 Abstención: Threshold={threshold:.3f}, Margin={margin:.3f}")
        except Exception as e:
            print(f"⚠️ Error cargando umbral: {e}")
    
    # Preparar salida
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir)

    output_file = os.path.join(final_output_dir, 'predictions.jsonl')
    total_processed = 0

    print(f"📄 Procesando: {input_file_path}")
    
    # Bucle de inferencia
    if not os.path.exists(input_file_path):
        print(f"❌ ERROR: No existe el archivo {input_file_path}")
        exit(1)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        with open(input_file_path, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                if not line.strip(): continue
                
                item = json.loads(line)
                text_id = item.get('id', f'it_{total_processed}')
                clean_text = preprocess_text(item.get('text', ''))
                
                if not clean_text:
                    out_f.write(json.dumps({"id": text_id, "label": 0.5}) + '\n')
                    total_processed += 1
                    continue
                
                inputs = tokenizer(
                    clean_text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512, 
                    padding=False
                ).to(device)
                
                with torch.inference_mode():
                    logits = model(inputs['input_ids'], inputs['attention_mask'])
                    probs = torch.softmax(logits, dim=1)
                    score = probs[0][1].item()
                    
                # Aplicar abstención
                if (threshold - margin) <= score <= (threshold + margin):
                    final_score = 0.5
                else:
                    final_score = float(score)
                    
                out_f.write(json.dumps({"id": text_id, "label": final_score}) + '\n')
                total_processed += 1
                
                if total_processed % 500 == 0:
                    print(f"  ... {total_processed} procesados")
                            
    print(f"\n✅ ¡Éxito! Predicciones guardadas en: {output_file}")