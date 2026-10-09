
import pandas as pd
from pathlib import Path

input_path = Path("data/processed/customer_features.csv")

df = pd.read_csv(input_path)

columns = [
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Active_Days",
    "Recency",
    "Average_Order_Value"
]

print("CUSTOMER FEATURE OUTLIER ANALYSIS")
print("=" * 45)
print("Dataset shape:", df.shape)
print("Number of customers:", df["CustomerID"].nunique())

for column in columns:
    print("\nFeature:", column)
    print("-" * 45)
    print(
        df[column]
        .describe(percentiles=[0.50, 0.90, 0.95, 0.99])
        .round(2)
    )

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    ]

    print("Lower bound:", round(lower_bound, 2))
    print("Upper bound:", round(upper_bound, 2))
    print("Potential outliers:", len(outliers))

print("\nAnalysis completed.")
