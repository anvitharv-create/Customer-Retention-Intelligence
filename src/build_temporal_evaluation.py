
import pandas as pd
from pathlib import Path

input_path = Path("data/processed/online_retail_cleaned.csv")
output_path = Path("data/processed/temporal_evaluation.csv")

df = pd.read_csv(input_path, parse_dates=["InvoiceDate"])

observation_end = pd.Timestamp("2011-06-30 23:59:59")
prediction_start = pd.Timestamp("2011-07-01 00:00:00")
prediction_end = pd.Timestamp("2011-09-28 23:59:59")

observation_df = df[df["InvoiceDate"] <= observation_end].copy()

prediction_df = df[
    (df["InvoiceDate"] >= prediction_start)
    & (df["InvoiceDate"] <= prediction_end)
].copy()

features = observation_df.groupby("CustomerID").agg(
    Last_Purchase_Date=("InvoiceDate", "max"),
    Frequency=("InvoiceNo", "nunique"),
    Monetary=("Revenue", "sum"),
    Total_Quantity=("Quantity", "sum"),
    Unique_Products=("StockCode", "nunique"),
    Active_Days=("InvoiceDate", lambda dates: dates.dt.date.nunique())
).reset_index()

features["Recency"] = (
    observation_end - features["Last_Purchase_Date"]
).dt.days

features["Average_Order_Value"] = (
    features["Monetary"] / features["Frequency"]
)

prediction_customers = set(prediction_df["CustomerID"].unique())

features["Churn"] = (
    ~features["CustomerID"].isin(prediction_customers)
).astype(int)

features = features.drop(columns=["Last_Purchase_Date"])

output_path.parent.mkdir(parents=True, exist_ok=True)
features.to_csv(output_path, index=False)

print("TEMPORAL EVALUATION DATASET CREATED")
print("Observation ends:", observation_end)
print("Prediction starts:", prediction_start)
print("Prediction ends:", prediction_end)
print("Observation transactions:", len(observation_df))
print("Prediction transactions:", len(prediction_df))
print("Customers:", len(features))
print("Missing values:", features.isna().sum().sum())
print("Duplicate customers:", features["CustomerID"].duplicated().sum())
print("\nChurn distribution:")
print(
    features["Churn"]
    .value_counts()
    .rename(index={0: "Retained", 1: "Churned"})
)
print("\nSaved to:", output_path)
