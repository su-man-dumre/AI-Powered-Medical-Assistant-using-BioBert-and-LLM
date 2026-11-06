
from pathlib import Path
import pandas as pd
from transformers import BertTokenizer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
SPLIT_DIR = BASE_DIR / "dataset" / "splits"

# BioBERT model
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=False
)

# Load datasets
train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

# Check required columns
required_columns = {"symptoms", "label"}

for name, df in [
    ("train", train_df),
    ("validation", val_df),
    ("test", test_df),
]:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{name}.csv is missing columns: {missing}")

# Tokenize a small sample to verify the pipeline
sample_texts = train_df["symptoms"].astype(str).head(5).tolist()

encoded = tokenizer(
    sample_texts,
    padding=True,
    truncation=True,
    max_length=256,
    return_tensors="pt",
)

print("BioBERT tokenizer loaded successfully!")
print("Sample count:", len(sample_texts))
print("Input IDs shape:", encoded["input_ids"].shape)
print("Attention mask shape:", encoded["attention_mask"].shape)
print("First sample tokens:")
print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
