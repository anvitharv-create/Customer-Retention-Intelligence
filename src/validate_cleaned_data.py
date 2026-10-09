
import pandas as pd
from pathlib import Path

file_path = Path("data/processed/online_retail_cleaned.csv")
df = pd.read_csv(file_path, parse_dates=["InvoiceDate"])

print("Cleaned CSV loaded successfully")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:")
print(df.isna().sum())
print("Unique customers:", df["CustomerID"].nunique())
print("Revenue total:", round(df["Revenue"].sum(), 2))
print("Duplicate rows:", df.duplicated().sum())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
