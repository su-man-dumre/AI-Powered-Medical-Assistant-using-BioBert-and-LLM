from datasets import load_dataset
import pandas as pd

dataset = load_dataset("Adarshreddy/Medcorpus_disease")

df = dataset["train"].to_pandas()

df.to_csv("dataset/raw/medcorpus_disease.csv", index=False)

print(df.head())