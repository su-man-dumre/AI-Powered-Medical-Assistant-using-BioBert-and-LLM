
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

SPLIT_DIR = BASE_DIR / "dataset" / "splits"
OUTPUT_DIR = BASE_DIR / "dataset" / "tokenized"

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"

MAX_LENGTH = 256
BATCH_SIZE = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load BioBERT tokenizer
# --------------------------------------------------

print("Loading BioBERT tokenizer...")

tokenizer = BertTokenizer.from_pretrained(
    MODEL_NAME,
    do_lower_case=False
)

print("BioBERT tokenizer loaded successfully!")

# --------------------------------------------------
# 3. Tokenization function
# --------------------------------------------------

def tokenize_batch(batch):

    return tokenizer(
        batch["symptoms"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

# --------------------------------------------------
# 4. Process each dataset
# --------------------------------------------------

datasets_to_process = {
    "train": SPLIT_DIR / "train.csv",
    "validation": SPLIT_DIR / "validation.csv",
    "test": SPLIT_DIR / "test.csv"
}

for split_name, file_path in datasets_to_process.items():

    print(f"\nProcessing {split_name} dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load CSV
    df = pd.read_csv(file_path)

    # Validate columns
    required_columns = {"symptoms", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{split_name} dataset must contain "
            f"'symptoms' and 'label' columns."
        )

    # Validate data
    if df["symptoms"].isna().any():
        raise ValueError(
            f"Missing symptoms found in {split_name}."
        )

    if df["label"].isna().any():
        raise ValueError(
            f"Missing labels found in {split_name}."
        )

    # Ensure correct data types
    df["symptoms"] = df["symptoms"].astype(str)
    df["label"] = df["label"].astype(int)

    if (df["label"] < 0).any():
        raise ValueError(
            f"Negative labels found in {split_name}."
        )

    # Convert pandas DataFrame to Hugging Face Dataset
    dataset = Dataset.from_pandas(
        df[["symptoms", "label"]],
        preserve_index=False
    )

    # Tokenize in batches
    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["symptoms"],
        desc=f"Tokenizing {split_name}"
    )

    # Save tokenized dataset
    save_path = OUTPUT_DIR / split_name

    tokenized_dataset.save_to_disk(str(save_path))

    # Print information
    print(f"{split_name} tokenization completed!")
    print("Total samples:", len(tokenized_dataset))
    print("Available columns:", tokenized_dataset.column_names)

    print(
        "First sample token count:",
        len(tokenized_dataset[0]["input_ids"])
    )

    print("Saved to:", save_path)

# --------------------------------------------------
# 5. Final verification
# --------------------------------------------------

print("\n" + "=" * 50)
print("FULL DATASET TOKENIZATION COMPLETED!")
print("=" * 50)

for split_name in datasets_to_process:

    print(f"{split_name}: Successfully saved")

print("\nOutput directory:", OUTPUT_DIR)
print("Ready for BioBERT fine-tuning!")
