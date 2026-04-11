"""
data_loader.py
==============
PAN 2026 - Voight-Kampff AI Detection
Fine-tuning de StyleDistance/mStyleDistance

Contiene:
  - preprocess_text: limpieza mínima preservando características estilísticas
  - Funciones de Data Augmentation: delete, sentence_shuffle, truncate
  - compute_genre_stratified_weights: pesos por género para balanceo
  - AIDetectionDataset: Dataset PyTorch con pre-tokenización offline
"""

import re
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =============================================================================
# PREPROCESAMIENTO
# =============================================================================

def preprocess_text(text: str) -> str:
    """
    Preprocesamiento MÍNIMO que preserva características estilísticas.

    CRÍTICO para detección de AI:
    - NO lowercase (la capitalización es una señal estilística)
    - NO eliminar puntuación (patrones de puntuación difieren entre humano/AI)
    - NO stemming/lemmatization (elección de palabras es importante)

    Solo limpiamos:
    - URLs (ruido sin valor estilístico)
    - Menciones @ y hashtags # (específico de redes sociales)
    - Espacios múltiples
    """
    if not isinstance(text, str) or text.strip() == '':
        return ""

    # Eliminar URLs
    #text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Eliminar menciones y hashtags
    #text = re.sub(r'@\w+', '', text)
    #text = re.sub(r'#\w+', '', text)

    # Normalizar espacios múltiples PERO preservar saltos de línea
    #text = re.sub(r'[^\S\n]+', ' ', text)

    # Eliminar líneas vacías múltiples
    #text = re.sub(r'\n\s*\n', '\n\n', text)

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()


    return text.strip()


# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def augment_text_delete(text: str, prob: float = 0.1) -> str:
    """Elimina palabras aleatoriamente — simula texto editado/incompleto."""
    words = text.split()
    if len(words) == 0:
        return text
    augmented = [w for w in words if random.random() > prob]
    if len(augmented) < len(words) * 0.5:
        return text
    return ' '.join(augmented) if augmented else text


def augment_text_sentence_shuffle(text: str) -> str:
    """
    Mezcla oraciones manteniendo cada oración internamente coherente.
    Más realista que swap de palabras — simula reordenamiento editorial.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 2:
        return text
    random.shuffle(sentences)
    return ' '.join(sentences)


def augment_text_truncate(text: str, min_ratio: float = 0.5, max_ratio: float = 0.85) -> str:
    """
    Toma entre 50-85% del texto desde el inicio.
    Motivación: FP en v1 eran textos largos (4163 chars) donde el modelo
    perdía información diagnóstica del final por MAX_LENGTH=128.
    Entrenar con fragmentos hace al modelo más robusto a textos parciales.
    """
    words = text.split()
    if len(words) < 20:
        return text
    keep = random.uniform(min_ratio, max_ratio)
    return ' '.join(words[:int(len(words) * keep)])


def augment_genre_human(
    df: pd.DataFrame,
    target_genres: list,
    ratio: float = 0.15,
    ratio_news: float = 0.25,
    techniques: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Aumenta SOLO la clase Human en los géneros especificados.
    news recibe ratio_news (más alto) porque tuvo 14.3% FP en v1 vs 7.6% essays.

    Args:
        df: DataFrame de training
        target_genres: Lista de géneros a aumentar (e.g. ['essays', 'news'])
        ratio: Ratio de aumento para géneros no-news
        ratio_news: Ratio de aumento para news
        techniques: Lista de técnicas ['delete', 'sentence_shuffle', 'truncate']
        seed: Semilla de reproducibilidad

    Returns:
        DataFrame aumentado
    """
    if techniques is None:
        techniques = ['delete', 'sentence_shuffle', 'truncate']

    aug_rows = []
    for genre in target_genres:
        genre_human = df[(df['genre'] == genre) & (df['label'] == 0)]
        r = ratio_news if genre == 'news' else ratio
        n_aug = int(len(genre_human) * r * 6)
        replace = n_aug > len(genre_human)
        samples = genre_human.sample(n=n_aug, replace=replace, random_state=seed)

        print(f'   {genre:10s}: {len(genre_human):,} human → +{n_aug:,} augmentados '
              f'(ratio {r*6:.0f}x, técnicas: {", ".join(techniques)})')

        for _, row in samples.iterrows():
            technique = random.choice(techniques)
            if technique == 'delete':
                aug_text = augment_text_delete(row['text'])
            elif technique == 'sentence_shuffle':
                aug_text = augment_text_sentence_shuffle(row['text'])
            elif technique == 'truncate':
                aug_text = augment_text_truncate(row['text'])
            else:
                aug_text = row['text']

            new_row = row.copy()
            new_row['text'] = aug_text
            if 'id' in row:
                new_row['id'] = str(row['id']) + '_aug'
            aug_rows.append(new_row)

    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        return pd.concat([df, aug_df], ignore_index=True)
    return df


# =============================================================================
# BALANCEO ESTRATIFICADO POR GÉNERO
# =============================================================================

def compute_genre_stratified_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Calcula pesos por muestra corrigiendo el desbalance DENTRO de cada género.
    Cada género contribuye proporcionalmente a su tamaño total,
    pero con clases balanceadas internamente.

    Args:
        df: DataFrame con columnas 'genre' y 'label'

    Returns:
        weights: Array numpy de pesos por muestra
    """
    weights = np.zeros(len(df))
    total = len(df)

    for genre in df['genre'].unique():
        genre_mask = df['genre'] == genre
        genre_df = df[genre_mask]

        for label in genre_df['label'].unique():
            label_mask = genre_mask & (df['label'] == label)
            label_count = label_mask.sum()
            n_genres = df['genre'].nunique()
            n_classes = genre_df['label'].nunique()
            w = (total / (n_genres * n_classes)) / label_count
            weights[label_mask.values] = w

    return weights


# =============================================================================
# PYTORCH DATASET
# =============================================================================

class AIDetectionDataset(Dataset):
    """
    Dataset con PRE-TOKENIZACIÓN OFFLINE.

    La tokenización se hace UNA sola vez en __init__ (batch tokenization),
    no en cada __getitem__. Esto elimina el cuello de botella principal
    en la GPU: tokenizar 30k muestras × 15 épocas → tokenizar 1× total.

    Args:
        texts (list): Lista de textos
        labels (list): Lista de labels (0=human, 1=AI)
        tokenizer: Tokenizer del modelo
        max_length (int): Longitud máxima de secuencia
    """

    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.labels = labels

        # PRE-TOKENIZAR todo de una vez con batch tokenization
        print(f"   Pre-tokenizando {len(texts):,} textos (batch)...")
        encodings = tokenizer(
            [str(t) for t in texts],
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # Guardar en RAM como tensores ya listos
        self.input_ids = encodings['input_ids']           # [N, max_length]
        self.attention_mask = encodings['attention_mask'] # [N, max_length]
        print(f"   ✅ Pre-tokenización completa ({len(texts):,} ejemplos)")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        # Solo indexado de tensores ya en RAM → velocidad máxima
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }
