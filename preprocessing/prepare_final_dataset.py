
# ============================================================
# Phase 6: Data Preparation for BioBERT Model Training
# ============================================================

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# 1. Define Project Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MERGED_PATH = BASE_DIR / "dataset" / "merged" / "merged_dataset.csv"
FINAL_DIR = BASE_DIR / "dataset" / "final"
SPLIT_DIR = BASE_DIR / "dataset" / "splits"

# Automatically create directories if they do not exist
FINAL_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Load Merged Dataset
# ============================================================

if not MERGED_PATH.exists():
    raise FileNotFoundError(
        f"Merged dataset not found: {MERGED_PATH}"
    )

df = pd.read_csv(MERGED_PATH)

print("Merged Dataset Loaded Successfully!")
print("Original Shape:", df.shape)

# ============================================================
# 3. Validate and Clean Dataset
# ============================================================

print("\n========== DATA VALIDATION ==========")

required_columns = {"disease", "symptoms"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain columns: {required_columns}"
    )

# Keep only the required columns
df = df[["disease", "symptoms"]].copy()

# Normalize whitespace and missing values
df["disease"] = df["disease"].astype("string").str.strip()
df["symptoms"] = df["symptoms"].astype("string").str.strip()

# Remove missing and empty records
df = df.dropna(subset=["disease", "symptoms"])

df = df[
    df["disease"].ne("") &
    df["symptoms"].ne("")
].copy()

# Remove exact duplicate disease-symptom pairs
df = df.drop_duplicates(
    subset=["disease", "symptoms"]
).reset_index(drop=True)

print("Shape after validation:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print(
    "Empty Symptoms:",
    df["symptoms"].str.strip().eq("").sum()
)

print(
    "Empty Diseases:",
    df["disease"].str.strip().eq("").sum()
)

# ============================================================
# 4. Analyze Disease Distribution
# ============================================================

print("\n========== DISEASE DISTRIBUTION ==========")

disease_counts = df["disease"].value_counts()

print("Total Samples:", len(df))
print("Unique Diseases:", df["disease"].nunique())

print("\nDistribution Statistics:")
print(disease_counts.describe())

# Save disease distribution
distribution_df = disease_counts.rename_axis(
    "disease"
).reset_index(name="sample_count")

distribution_df.to_csv(
    FINAL_DIR / "disease_distribution.csv",
    index=False
)

# Identify rare disease classes
for threshold in [3, 7, 10]:
    rare_classes = disease_counts[
        disease_counts < threshold
    ]

    print(
        f"\nDiseases with fewer than {threshold} samples:",
        len(rare_classes)
    )

    if not rare_classes.empty:
        print(rare_classes.to_string())

# ============================================================
# 5. Check Whether Stratified Splitting Is Possible
# ============================================================

# A 70/15/15 stratified split requires at least 3 samples
# per class. Rare classes must be reviewed before splitting.

if disease_counts.min() < 3:
    print("\nWARNING: Some diseases have fewer than 3 samples.")
    print("Training/validation/test splits were NOT created.")
    print(
        "Review rare classes and decide how to handle them "
        "before continuing."
    )

    raise SystemExit(1)

# ============================================================
# 6. Encode Disease Labels
# ============================================================

print("\n========== LABEL ENCODING ==========")

# Sort disease names so that label IDs are reproducible
diseases = sorted(df["disease"].unique())

label_to_id = {
    disease: index
    for index, disease in enumerate(diseases)
}

df["label"] = df["disease"].map(
    label_to_id
).astype(int)

# Save the mapping for future predictions
label_mapping = pd.DataFrame({
    "label": range(len(diseases)),
    "disease": diseases
})

label_mapping.to_csv(
    FINAL_DIR / "label_mapping.csv",
    index=False
)

# Save the final dataset with numeric labels
df.to_csv(
    FINAL_DIR / "final_dataset.csv",
    index=False
)

print("Label encoding completed.")
print("Total Labels:", len(diseases))

# ============================================================
# 7. Split Dataset: 70% Train, 15% Validation, 15% Test
# ============================================================

print("\n========== DATASET SPLITTING ==========")

# First, separate 70% training data and 30% temporary data
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

# Divide the remaining 30% equally
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

# ============================================================
# 8. Save Train, Validation, and Test Sets
# ============================================================

train_path = SPLIT_DIR / "train.csv"
validation_path = SPLIT_DIR / "validation.csv"
test_path = SPLIT_DIR / "test.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)
test_df.to_csv(test_path, index=False)

# ============================================================
# 9. Verify Splits
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Total Samples:", len(df))
print("Total Diseases:", df["disease"].nunique())

print(
    f"Training:   {len(train_df)} "
    f"({len(train_df) / len(df) * 100:.2f}%)"
)

print(
    f"Validation: {len(validation_df)} "
    f"({len(validation_df) / len(df) * 100:.2f}%)"
)

print(
    f"Testing:    {len(test_df)} "
    f"({len(test_df) / len(df) * 100:.2f}%)"
)

# Confirm that all classes appear in all three subsets
for name, subset in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df)
]:
    print(
        f"{name} unique diseases:",
        subset["disease"].nunique()
    )

print("\nFiles Created Successfully:")
print(FINAL_DIR / "final_dataset.csv")
print(FINAL_DIR / "label_mapping.csv")
print(FINAL_DIR / "disease_distribution.csv")
print(train_path)
print(validation_path)
print(test_path)

print("\nPhase 6 completed successfully!")
