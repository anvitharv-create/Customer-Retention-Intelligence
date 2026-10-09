import pandas as pd
from pathlib import Path

input_path = Path("data/processed/online_retail_cleaned.csv")

df = pd.read_csv(input_path, parse_dates=["InvoiceDate"])

periods = [
    ("Period 1", "2011-03-01", "2011-05-31"),
    ("Period 2", "2011-06-01", "2011-08-31"),
    ("Period 3", "2011-09-01", "2011-11-30"),
]

previously_seen = set()

print("CUSTOMER COHORT CHECK")
print("=" * 75)

for name, start, end in periods:
    period_data = df[
        (df["InvoiceDate"] >= pd.Timestamp(start))
        & (
            df["InvoiceDate"]
            < pd.Timestamp(end) + pd.Timedelta(days=1)
        )
    ]

    current_customers = set(period_data["CustomerID"].unique())
    returning_customers = current_customers & previously_seen
    new_customers = current_customers - previously_seen

    print(f"\n{name}: {start} to {end}")
    print(f"Transactions: {len(period_data):,}")
    print(f"Active customers: {len(current_customers):,}")
    print(f"Previously seen customers: {len(returning_customers):,}")
    print(f"Newly observed customers: {len(new_customers):,}")

    previously_seen.update(current_customers)

print("\nCohort check completed.")