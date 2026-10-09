
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
    confusion_matrix,
    classification_report
)

data_path = Path("data/processed/customer_features.csv")
model_path = Path("models/churn_model.joblib")

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

print("MODEL EVALUATION")
print("=" * 40)
print("Test customers:", len(y_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Retained", "Churned"],
    zero_division=0
))
