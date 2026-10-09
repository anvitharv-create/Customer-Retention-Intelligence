
import pandas as pd
from pathlib import Path

file_path = Path("data/raw/online+retail/Online Retail.xlsx")
df = pd.read_excel(file_path)

print("Invoice date range:")
print("Start:", df["InvoiceDate"].min())
print("End:", df["InvoiceDate"].max())

print("\nCancelled invoice rows:", df["InvoiceNo"].astype(str).str.startswith("C").sum())
print("Rows with zero or negative quantity:", (df["Quantity"] <= 0).sum())
print("Rows with zero or negative unit price:", (df["UnitPrice"] <= 0).sum())
print("Rows with missing customer ID:", df["CustomerID"].isna().sum())

print("\nUnique customers:", df["CustomerID"].nunique())
print("Unique invoices:", df["InvoiceNo"].nunique())
print("Unique products:", df["StockCode"].nunique())

print("\nQuantity summary:")
print(df["Quantity"].describe())

print("\nUnit price summary:")
print(df["UnitPrice"].describe())
