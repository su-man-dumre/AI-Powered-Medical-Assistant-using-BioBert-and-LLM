# =====================================================
# Exploratory Data Analysis (EDA)
# =====================================================

import pandas as pd

# =====================================================
# Load Merged Dataset
# =====================================================

df = pd.read_csv("dataset/merged/merged_dataset.csv")

print("Merged Dataset Loaded Successfully!")

print("\nShape:")
print(df.shape)


# =====================================================
# Dataset Overview
# =====================================================

print("\n========== DATASET OVERVIEW ==========")

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
print(df.info())

# =====================================================
# Missing Values
# =====================================================

print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

# =====================================================
# Duplicate Rows
# =====================================================

print("\n========== DUPLICATE ROWS ==========")

duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

# =====================================================
# Dataset Statistics
# =====================================================

print("\n========== BASIC STATISTICS ==========")

print("Total Samples :", len(df))

print("Total Diseases :", df["disease"].nunique())


import matplotlib.pyplot as plt

# =====================================================
# Disease Distribution
# =====================================================

top_diseases = df["disease"].value_counts().head(20)

plt.figure(figsize=(12,8))

top_diseases.plot(kind="bar")

plt.title("Top 20 Most Common Diseases")

plt.xlabel("Disease")

plt.ylabel("Number of Samples")

plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("results/plots/top20_diseases.png")

plt.show()

df["symptom_length"] = df["symptoms"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(10,6))

plt.hist(df["symptom_length"], bins=30)

plt.title("Distribution of Symptom Length")

plt.xlabel("Number of Words")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("results/plots/symptom_length_distribution.png")

plt.show()

print("\n========== TOP 20 DISEASES ==========\n")

print(df["disease"].value_counts().head(20))

print("\n========== SYMPTOM LENGTH ==========\n")

print("Average Length :", round(df["symptom_length"].mean(),2))

print("Minimum Length :", df["symptom_length"].min())

print("Maximum Length :", df["symptom_length"].max())

summary = {
    "Total Samples":[len(df)],
    "Unique Diseases":[df["disease"].nunique()],
    "Missing Values":[df.isnull().sum().sum()],
    "Duplicate Rows":[df.duplicated().sum()]
}

summary_df = pd.DataFrame(summary)

summary_df.to_csv("results/dataset_summary.csv", index=False)

print("\nDataset Summary Saved Successfully!")