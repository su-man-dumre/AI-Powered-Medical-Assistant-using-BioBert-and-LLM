# =====================================================
# Generate BioBERT Embeddings for Entire Dataset
# =====================================================

import pandas as pd
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =====================================================
# Load Dataset
# =====================================================

df = pd.read_csv("dataset/final/final_dataset.csv")

print("Dataset Loaded Successfully!")
print(f"Total Samples: {len(df)}")

# =====================================================
# Device
# =====================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\nUsing Device: {device}")

# =====================================================
# Load Model
# =====================================================

MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

print("\nLoading BioBERT Sentence Transformer...")

model = SentenceTransformer(MODEL_NAME, device=device)

print("Model Loaded Successfully!")

# =====================================================
# Generate Embeddings
# =====================================================

print("\nGenerating embeddings...")

BATCH_SIZE = 64

symptoms = df["symptoms"].tolist()

embeddings = []

for i in tqdm(range(0, len(symptoms), BATCH_SIZE)):

    batch = symptoms[i:i+BATCH_SIZE]

    batch_embeddings = model.encode(
        batch,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )

    embeddings.append(batch_embeddings)

# =====================================================
# Combine Embeddings
# =====================================================

embeddings = np.vstack(embeddings)

print("\nEmbedding Matrix Shape:")
print(embeddings.shape)

# =====================================================
# Save Embeddings
# =====================================================

output_path = "embeddings/symptom_embeddings.npy"

np.save(output_path, embeddings)

print("\nEmbeddings Saved Successfully!")

print(f"Location: {output_path}")