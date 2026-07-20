# =====================================================
# Merge Cleaned Datasets
# =====================================================

import pandas as pd

# =====================================================
# Load Cleaned Datasets
# =====================================================

df1 = pd.read_csv("dataset/cleaned/symptom2disease_clean.csv")
df2 = pd.read_csv("dataset/cleaned/final_symptoms_clean.csv")

print("Dataset 1 Shape:", df1.shape)
print("Dataset 2 Shape:", df2.shape)

# =====================================================
# Merge Datasets
# =====================================================

merged_df = pd.concat([df1, df2], ignore_index=True)

print("\nShape After Merging:", merged_df.shape)

# =====================================================
# Remove Duplicate Rows
# =====================================================

before = len(merged_df)

merged_df.drop_duplicates(inplace=True)

after = len(merged_df)

print("\nDuplicate Rows Removed:", before - after)

print("Final Shape:", merged_df.shape)

# =====================================================
# Shuffle Dataset
# =====================================================

merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nDataset Shuffled Successfully!")

# =====================================================
# Save Merged Dataset
# =====================================================

output_path = "dataset/merged/merged_dataset.csv"

merged_df.to_csv(output_path, index=False)

print("\nMerged Dataset Saved Successfully!")
print("Location:", output_path)

# =====================================================
# Display Information
# =====================================================

print("\nFirst 5 Rows:")
print(merged_df.head())

print("\nColumns:")
print(merged_df.columns.tolist())

print("\nMissing Values:")
print(merged_df.isnull().sum())

print("\nUnique Diseases:", merged_df["disease"].nunique())