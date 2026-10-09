
import pandas as pd
from pathlib import Path

input_path = Path("data/processed/online_retail_cleaned.csv")
output_path = Path("data/processed/customer_features.csv")

df = pd.read_csv(input_path, parse_dates=["InvoiceDate"])

observation_end = pd.Timestamp("2011-09-09 23:59:59")
prediction_start = pd.Timestamp("2011-09-10 00:00:00")
prediction_end = pd.Timestamp("2011-12-08 23:59:59")

observation_df = df[df["InvoiceDate"] <= observation_end].copy()

prediction_df = df[
    (df["InvoiceDate"] >= prediction_start)
    & (df["InvoiceDate"] <= prediction_end)
].copy()

customer_features = observation_df.groupby("CustomerID").agg(
    Last_Purchase_Date=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Unique_Products=("StockCode", "nunique"),
    Active_Days=("InvoiceDate", lambda dates: dates.dt.date.nunique())
).reset_index()

customer_features["Recency"] = (
    observation_end - customer_features["Last_Purchase_Date"]
).dt.days

customer_features["Average_Order_Value"] = (
    customer_features["Monetary"] / customer_features["Frequency"]
)

prediction_customers = set(prediction_df["CustomerID"].unique())

customer_features["Churn"] = (
    ~customer_features["CustomerID"].isin(prediction_customers)
).astype(int)

customer_features = customer_features.drop(
    columns=["Last_Purchase_Date"]
)

output_path.parent.mkdir(parents=True, exist_ok=True)
customer_features.to_csv(output_path, index=False)

print("Customer feature dataset created successfully")
print("Observation period ends:", observation_end)
print("Prediction period starts:", prediction_start)
print("Prediction period ends:", prediction_end)
print("Observation transactions:", len(observation_df))
print("Prediction transactions:", len(prediction_df))
print("Number of customers:", len(customer_features))
print("Number of features:", len(customer_features.columns) - 2)
print("\nChurn distribution:")
print(
    customer_features["Churn"]
    .value_counts()
    .rename(index={0: "Retained", 1: "Churned"})
)
print("\nChurn percentages:")
print(
    (customer_features["Churn"].value_counts(normalize=True) * 100).round(2)
)
print("\nMissing values:", customer_features.isna().sum().sum())
print(
    "Duplicate customers:",
    customer_features["CustomerID"].duplicated().sum()
)
print("\nFeature preview:")
print(customer_features.head())
print("\nSaved to:", output_path)
