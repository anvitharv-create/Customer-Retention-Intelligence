
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
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
    confusion_matrix
)

data_path = Path("data/processed/online_retail_cleaned.csv")
output_path = Path("reports")
output_path.mkdir(parents=True, exist_ok=True)

required_columns = [
    "InvoiceDate",
    "CustomerID",
    "InvoiceNo",
    "Revenue",
    "Quantity",
    "StockCode"
]

if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path, parse_dates=["InvoiceDate"])

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError("The cleaned dataset contains missing required values.")

if df["CustomerID"].duplicated().all():
    raise ValueError("CustomerID validation failed.")

if df["InvoiceDate"].isnull().any():
    raise ValueError("InvoiceDate contains invalid or missing dates.")

data_start = df["InvoiceDate"].min()
data_end = df["InvoiceDate"].max()

feature_columns = [
    "Frequency",
    "Monetary",
    "Total_Quantity",
    "Unique_Products",
    "Active_Days",
    "Recency",
    "Average_Order_Value"
]

snapshot_settings = [
    {
        "name": "training",
        "cutoff": "2011-03-10",
        "prediction_start": "2011-03-11",
        "prediction_end": "2011-06-08"
    },
    {
        "name": "validation",
        "cutoff": "2011-06-09",
        "prediction_start": "2011-06-10",
        "prediction_end": "2011-09-07"
    },
    {
        "name": "test",
        "cutoff": "2011-09-09",
        "prediction_start": "2011-09-10",
        "prediction_end": "2011-12-08"
    }
]

for setting in snapshot_settings:
    start = pd.Timestamp(setting["prediction_start"])
    end = pd.Timestamp(setting["prediction_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    cutoff_end = pd.Timestamp(setting["cutoff"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if start != cutoff_end + pd.Timedelta(seconds=1):
        raise ValueError(f"Invalid prediction start for {setting['name']}.")

    if (pd.Timestamp(setting["prediction_end"]) - start.normalize()).days != 89:
        raise ValueError(f"Prediction window is not 90 calendar days for {setting['name']}.")

    if end > data_end:
        raise ValueError(
            f"Dataset ends at {data_end}, before the complete prediction window for {setting['name']}."
        )

    if pd.Timestamp(setting["cutoff"]) < data_start.normalize():
        raise ValueError(f"Observation cutoff precedes available data for {setting['name']}.")

def build_snapshot(cutoff_date, prediction_start_date, prediction_end_date):
    cutoff = pd.Timestamp(cutoff_date)
    observation_end = cutoff + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    prediction_start = pd.Timestamp(prediction_start_date)
    prediction_end = (
        pd.Timestamp(prediction_end_date)
        + pd.Timedelta(days=1)
        - pd.Timedelta(seconds=1)
    )

    observation = df[df["InvoiceDate"] <= observation_end].copy()

    prediction = df[
        (df["InvoiceDate"] >= prediction_start)
        & (df["InvoiceDate"] <= prediction_end)
    ]

    features = observation.groupby("CustomerID").agg(
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

    active_future_customers = set(prediction["CustomerID"].unique())

    features["Churn"] = (
        ~features["CustomerID"].isin(active_future_customers)
    ).astype(int)

    features = features.drop(columns=["Last_Purchase_Date"])

    if features.empty:
        raise ValueError(f"No customers found for cutoff {cutoff_date}.")

    if features["CustomerID"].duplicated().any():
        raise ValueError(f"Duplicate customer rows found for cutoff {cutoff_date}.")

    if features[feature_columns].isnull().any().any():
        raise ValueError(f"Missing feature values found for cutoff {cutoff_date}.")

    if features["Churn"].nunique() != 2:
        raise ValueError(f"Both churn classes are not present for cutoff {cutoff_date}.")

    return features

train_data = build_snapshot(
    snapshot_settings[0]["cutoff"],
    snapshot_settings[0]["prediction_start"],
    snapshot_settings[0]["prediction_end"]
)

validation_data = build_snapshot(
    snapshot_settings[1]["cutoff"],
    snapshot_settings[1]["prediction_start"],
    snapshot_settings[1]["prediction_end"]
)

test_data = build_snapshot(
    snapshot_settings[2]["cutoff"],
    snapshot_settings[2]["prediction_start"],
    snapshot_settings[2]["prediction_end"]
)

X_train = train_data[feature_columns]
y_train = train_data["Churn"]

X_validation = validation_data[feature_columns]
y_validation = validation_data["Churn"]

X_test = test_data[feature_columns]
y_test = test_data["Churn"]

logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": forest_model
}

comparison = []
trained_models = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    validation_probabilities = model.predict_proba(X_validation)[:, 1]

    comparison.append({
        "Model": model_name,
        "Validation_ROC_AUC": roc_auc_score(
            y_validation, validation_probabilities
        ),
        "Validation_PR_AUC": average_precision_score(
            y_validation, validation_probabilities
        )
    })

    trained_models[model_name] = model

comparison_df = pd.DataFrame(comparison).sort_values(
    "Validation_ROC_AUC",
    ascending=False
)

comparison_df.to_csv(
    output_path / "temporal_model_comparison.csv",
    index=False
)

selected_name = comparison_df.iloc[0]["Model"]
selected_model = trained_models[selected_name]

validation_probabilities = selected_model.predict_proba(X_validation)[:, 1]

threshold_results = []

for threshold in np.round(np.arange(0.20, 0.801, 0.05), 2):
    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": float(threshold),
        "Precision": precision_score(
            y_validation,
            validation_predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_validation,
            validation_predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    output_path / "temporal_threshold_analysis.csv",
    index=False
)

best_threshold = float(
    threshold_df.loc[
        threshold_df["F1_Score"].idxmax(),
        "Threshold"
    ]
)

test_probabilities = selected_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

metrics = {
    "Model": selected_name,
    "Threshold": best_threshold,
    "Train_Customers": int(len(train_data)),
    "Validation_Customers": int(len(validation_data)),
    "Test_Customers": int(len(test_data)),
    "Train_Churn_Rate": float(y_train.mean()),
    "Validation_Churn_Rate": float(y_validation.mean()),
    "Test_Churn_Rate": float(y_test.mean()),
    "Accuracy": accuracy_score(y_test, test_predictions),
    "Balanced_Accuracy": balanced_accuracy_score(y_test, test_predictions),
    "Precision": precision_score(y_test, test_predictions, zero_division=0),
    "Recall": recall_score(y_test, test_predictions, zero_division=0),
    "F1_Score": f1_score(y_test, test_predictions, zero_division=0),
    "ROC_AUC": roc_auc_score(y_test, test_probabilities),
    "PR_AUC": average_precision_score(y_test, test_probabilities)
}

pd.DataFrame([metrics]).to_csv(
    output_path / "temporal_evaluation_metrics.csv",
    index=False
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

pd.DataFrame(
    matrix,
    index=["Actual_Retained", "Actual_Churned"],
    columns=["Predicted_Retained", "Predicted_Churned"]
).to_csv(
    output_path / "temporal_confusion_matrix.csv"
)

metadata = {
    "dataset_path": str(data_path),
    "dataset_start": str(data_start),
    "dataset_end": str(data_end),
    "evaluation_type": "Temporal snapshot evaluation",
    "customer_ids_may_appear_in_multiple_snapshots": True,
    "prediction_windows_are_sequential": True,
    "model_selection_metric": "Validation ROC-AUC",
    "threshold_selection_metric": "Validation F1-score",
    "test_used_for_model_selection": False,
    "snapshots": snapshot_settings,
    "train_customers": int(len(train_data)),
    "validation_customers": int(len(validation_data)),
    "test_customers": int(len(test_data)),
    "selected_model": selected_name,
    "selected_threshold": best_threshold,
    "features": feature_columns
}

with open(
    output_path / "temporal_evaluation_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metadata, file, indent=4)

print("\nTEMPORAL MODEL EVALUATION COMPLETED")

print("\nDataset date range:")
print("Start:", data_start)
print("End:", data_end)

print("\nCustomer cohort sizes:")
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Testing:", len(test_data))

print("\nChurn rates:")
print("Training:", round(y_train.mean() * 100, 2), "%")
print("Validation:", round(y_validation.mean() * 100, 2), "%")
print("Testing:", round(y_test.mean() * 100, 2), "%")

print("\nModel comparison:")
print(comparison_df.to_string(index=False))

print("\nSelected model:", selected_name)
print("Selected threshold:", best_threshold)

print("\nFinal test metrics:")
for metric, value in metrics.items():
    if isinstance(value, (float, np.floating)):
        print(f"{metric}: {value:.4f}")
    else:
        print(f"{metric}: {value}")

print("\nTest confusion matrix:")
print(matrix)

print("\nReports saved in:", output_path)
