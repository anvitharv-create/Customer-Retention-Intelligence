
import pandas as pd
import joblib
from pathlib import Path

data_path = Path("data/processed/customer_features.csv")
model_path = Path("models/churn_model.joblib")
output_path = Path("reports/model_coefficients.csv")

df = pd.read_csv(data_path)

excluded_columns = ["CustomerID", "Churn"]
features = [column for column in df.columns if column not in excluded_columns]

model = joblib.load(model_path)
classifier = model.named_steps["model"]
scaler = model.named_steps["scaler"]

coefficients = classifier.coef_[0]

results = pd.DataFrame({
    "Feature": features,
    "Coefficient": coefficients
})

results["Absolute_Importance"] = results["Coefficient"].abs()
results["Direction"] = results["Coefficient"].apply(
    lambda value: "Higher predicted churn" if value > 0 else "Lower predicted churn"
)

results = results.sort_values(
    by="Absolute_Importance",
    ascending=False
)

output_path.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(output_path, index=False)

print("LOGISTIC REGRESSION COEFFICIENT ANALYSIS")
print("=" * 45)
print(results.round(4).to_string(index=False))
print("\nSaved to:", output_path)
print("\nNote: Coefficients describe model associations, not causal effects.")
