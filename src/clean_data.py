
import pandas as pd
from pathlib import Path

input_path = Path("data/raw/online+retail/Online Retail.xlsx")
output_path = Path("data/processed/online_retail_cleaned.csv")

df = pd.read_excel(input_path)

initial_rows = len(df)

df["CustomerID"] = df["CustomerID"].astype("Int64").astype("string")
df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
df["StockCode"] = df["StockCode"].astype(str).str.strip()

df = df.dropna(subset=["CustomerID"])
df = df[~df["InvoiceNo"].str.startswith("C")]
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
df = df.drop_duplicates()

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("Data cleaning completed")
print("Original rows:", initial_rows)
print("Rows after cleaning:", len(df))
print("Rows removed:", initial_rows - len(df))
print("Unique customers:", df["CustomerID"].nunique())
print("Unique invoices:", df["InvoiceNo"].nunique())
print("Date range:", df["InvoiceDate"].min(), "to", df["InvoiceDate"].max())
print("Cleaned dataset saved to:", output_path)
print("Missing customer IDs:", df["CustomerID"].isna().sum())
print("Non-positive quantities:", (df["Quantity"] <= 0).sum())
print("Non-positive prices:", (df["UnitPrice"] <= 0).sum())
