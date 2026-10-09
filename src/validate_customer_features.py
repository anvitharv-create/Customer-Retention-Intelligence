
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/customer_features.csv")
df = pd.read_csv(file_path)

print("Customer features loaded successfully")
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate customers:", df["CustomerID"].duplicated().sum())
print("\nChurn counts:")
print(df["Churn"].value_counts())
print("\nChurn percentages:")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))
print("\nFeature summary:")
print(df.describe().round(2))
