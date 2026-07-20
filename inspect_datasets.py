import pandas as pd

datasets = {
    "Symptom2Disease": "dataset/raw/Symptom2Disease.csv",
    "FinalSymptoms": "dataset/raw/final_symptoms_to_disease.csv",
    #"BinaryData": "dataset/raw/data.csv",
    "MedCorpus": "dataset/raw/medcorpus_disease.csv"
}

for name, path in datasets.items():
    print("=" * 70)
    print(f"Dataset: {name}")
    print("=" * 70)

    df = pd.read_csv(path)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nInformation:")
    print(df.info())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\n")