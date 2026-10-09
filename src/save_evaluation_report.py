
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

data_path = Path("data/processed/customer_features.csv")
model_path = Path("models/churn_model.joblib")
report_dir = Path("reports")

report_dir.mkdir(parents=True, exist_ok=True)

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
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

metrics = pd.DataFrame([{
    "Model": "Logistic Regression",
    "Test_Customers": len(y_test),
    "Accuracy": accuracy_score(y_test, predictions),
    "Precision": precision_score(y_test, predictions, zero_division=0),
    "Recall": recall_score(y_test, predictions, zero_division=0),
    "F1_Score": f1_score(y_test, predictions, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, probabilities)
}])

matrix = confusion_matrix(y_test, predictions)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual_Retained", "Actual_Churned"],
    columns=["Predicted_Retained", "Predicted_Churned"]
)

metrics.to_csv(report_dir / "final_evaluation_metrics.csv", index=False)
matrix_df.to_csv(report_dir / "confusion_matrix.csv")

print("Evaluation reports saved successfully")
print("\nMetrics:")
print(metrics.round(4).to_string(index=False))
print("\nConfusion matrix:")
print(matrix_df)
print("\nFiles saved:")
print(report_dir / "final_evaluation_metrics.csv")
print(report_dir / "confusion_matrix.csv")

