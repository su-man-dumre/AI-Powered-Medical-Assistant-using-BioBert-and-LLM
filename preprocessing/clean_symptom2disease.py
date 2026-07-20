
# Cleaning Symptom2Disease Dataset


import pandas as pd
import re
df = pd.read_csv("dataset/raw/Symptom2Disease.csv")

print("Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")

# Display Dataset Information
print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove Unnecessary Column
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print("\nShape After Removing 'Unnamed: 0':")
print(df.shape)

df.rename(columns={
    "label": "disease",
    "text": "symptoms"
}, inplace=True)

print("\nUpdated Columns:")
print(df.columns.tolist())

# =====================================================
# Remove Missing Values
# =====================================================

before_rows = len(df)

df.dropna(inplace=True)

after_rows = len(df)

print("\nMissing Values Removed:")
print(f"Rows Before: {before_rows}")
print(f"Rows After : {after_rows}")

# =====================================================
# Remove Duplicate Rows
# =====================================================

before_rows = len(df)

df.drop_duplicates(inplace=True)

after_rows = len(df)

print("\nDuplicate Rows Removed:")
print(f"Rows Before: {before_rows}")
print(f"Rows After : {after_rows}")


# =====================================================
# Clean Symptom Text
# =====================================================

def clean_text(text):
    text = str(text).lower()                  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()     # Remove extra spaces
    return text

df["symptoms"] = df["symptoms"].apply(clean_text)

print("\nCleaned Symptoms:")
print(df["symptoms"].head())


# =====================================================
# Standardize Disease Names
# =====================================================

df["disease"] = (
    df["disease"]
    .str.strip()
    .str.title()
)

print("\nDisease Names:")
print(df["disease"].head())


# =====================================================
# Save Clean Dataset
# =====================================================

output_path = "dataset/cleaned/symptom2disease_clean.csv"

df.to_csv(output_path, index=False)

print("\nCleaning Completed Successfully!")
print(f"Final Shape: {df.shape}")
print(f"Dataset saved to: {output_path}")