
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
THRESHOLDS = [round(value, 2) for value in [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70
]]

data_path = Path("data/processed/customer_features.csv")
model_dir = Path("models")
reports_dir = Path("reports")

model_dir.mkdir(parents=True, exist_ok=True)
reports_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

target = "Churn"
excluded_columns = ["CustomerID", target]
features = [column for column in df.columns if column not in excluded_columns]

if target not in df.columns:
    raise ValueError("The dataset must contain a Churn column.")

if df[target].isna().any():
    raise ValueError("The target column contains missing values.")

if df[target].nunique() != 2:
    raise ValueError("The Churn target must contain exactly two classes.")

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("Churn must be encoded as 0 for retained and 1 for churned.")

if not features:
    raise ValueError("No model features were found.")

X = df[features]
y = df[target].astype(int)

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=TEST_SIZE + VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

validation_fraction_of_temp = VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1 - validation_fraction_of_temp,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1
        ))
    ])
}

validation_results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    validation_results.append({
        "Model": name,
        "Validation_Accuracy": accuracy_score(y_validation, predictions),
        "Validation_Balanced_Accuracy": balanced_accuracy_score(
            y_validation, predictions
        ),
        "Validation_Precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "Validation_Recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "Validation_F1": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "Validation_ROC_AUC": roc_auc_score(
            y_validation, probabilities
        ),
        "Validation_PR_AUC": average_precision_score(
            y_validation, probabilities
        )
    })

validation_df = pd.DataFrame(validation_results).sort_values(
    by="Validation_ROC_AUC",
    ascending=False
)

validation_df.to_csv(
    reports_dir / "validation_model_comparison.csv",
    index=False
)

print("\nVALIDATION MODEL COMPARISON")
print(validation_df.round(4).to_string(index=False))

best_model_name = validation_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

validation_probabilities = best_model.predict_proba(X_validation)[:, 1]

threshold_results = []

for threshold in THRESHOLDS:
    predictions = (validation_probabilities >= threshold).astype(int)
    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=[0, 1]
    )

    threshold_results.append({
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_validation, predictions),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_validation, predictions
        ),
        "Precision": precision_score(
            y_validation, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_validation, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_validation, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_validation, validation_probabilities
        ),
        "PR_AUC": average_precision_score(
            y_validation, validation_probabilities
        ),
        "TN": matrix[0, 0],
        "FP": matrix[0, 1],
        "FN": matrix[1, 0],
        "TP": matrix[1, 1]
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    reports_dir / "threshold_analysis.csv",
    index=False
)

best_threshold_row = threshold_df.loc[
    threshold_df["F1_Score"].idxmax()
]

best_threshold = float(best_threshold_row["Threshold"])

test_probabilities = best_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

test_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

test_metrics = {
    "Model": best_model_name,
    "Decision_Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Balanced_Accuracy": balanced_accuracy_score(
        y_test, test_predictions
    ),
    "Precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "F1_Score": f1_score(
        y_test, test_predictions, zero_division=0
    ),
    "ROC_AUC": roc_auc_score(y_test, test_probabilities),
    "PR_AUC": average_precision_score(y_test, test_probabilities)
}

pd.DataFrame([test_metrics]).to_csv(
    reports_dir / "final_evaluation_metrics.csv",
    index=False
)

pd.DataFrame(
    test_matrix,
    index=["Actual_Retained", "Actual_Churned"],
    columns=["Predicted_Retained", "Predicted_Churned"]
).to_csv(reports_dir / "confusion_matrix.csv")

joblib.dump(best_model, model_dir / "churn_model.joblib")

metadata = {
    "model": best_model_name,
    "decision_threshold": best_threshold,
    "target_definition": "1 = no purchase during the defined prediction window",
    "threshold_selection": "Maximum validation F1-score among tested thresholds",
    "model_selection": "Highest validation ROC-AUC",
    "random_state": RANDOM_STATE,
    "split_sizes": {
        "training": len(X_train),
        "validation": len(X_validation),
        "testing": len(X_test)
    },
    "features": features
}

with open(reports_dir / "model_metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nDATASET SIZES")
print("Training:", len(X_train))
print("Validation:", len(X_validation))
print("Testing:", len(X_test))

print("\nSELECTED MODEL:", best_model_name)
print("Validation threshold selected by F1-score:", best_threshold)
print("Validation F1-score:", round(best_threshold_row["F1_Score"], 4))

print("\nFINAL TEST METRICS")
for metric, value in test_metrics.items():
    if isinstance(value, (int, float)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nTEST CLASSIFICATION REPORT")
print(classification_report(
    y_test,
    test_predictions,
    labels=[0, 1],
    target_names=["Retained", "Churned"],
    zero_division=0
))

print("Test confusion matrix:")
print(test_matrix)

print("\nSaved model:", model_dir / "churn_model.joblib")
print("Saved validation comparison:", reports_dir / "validation_model_comparison.csv")
print("Saved threshold analysis:", reports_dir / "threshold_analysis.csv")
print("Saved final test metrics:", reports_dir / "final_evaluation_metrics.csv")
print("Saved confusion matrix:", reports_dir / "confusion_matrix.csv")
print("Saved model metadata:", reports_dir / "model_metadata.json")
