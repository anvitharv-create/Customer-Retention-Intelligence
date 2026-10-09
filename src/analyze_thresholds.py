
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

data_path = Path("data/processed/customer_features.csv")
model_path = Path("models/churn_model.joblib")
output_path = Path("reports/threshold_analysis.csv")

df = pd.read_csv(data_path)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = joblib.load(model_path)
probabilities = model.predict_proba(X_test)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
results = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1_Score": f1_score(y_test, predictions, zero_division=0),
        "True_Negatives": matrix[0, 0],
        "False_Positives": matrix[0, 1],
        "False_Negatives": matrix[1, 0],
        "True_Positives": matrix[1, 1]
    })

results_df = pd.DataFrame(results)
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print("THRESHOLD ANALYSIS")
print("=" * 75)
print(results_df.round(4).to_string(index=False))
print("\nSaved to:", output_path)
