
import pandas as pd
from pathlib import Path

reports_path = Path("reports")

original_path = reports_path / "final_evaluation_metrics.csv"
temporal_path = reports_path / "temporal_evaluation_metrics.csv"

if not original_path.exists():
    raise FileNotFoundError(f"File not found: {original_path}")

if not temporal_path.exists():
    raise FileNotFoundError(f"File not found: {temporal_path}")

original = pd.read_csv(original_path)
temporal = pd.read_csv(temporal_path)

metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC",
    "PR_AUC"
]

for column in metrics:
    if column not in original.columns or column not in temporal.columns:
        raise ValueError(f"Required metric missing: {column}")

comparison = pd.DataFrame({
    "Metric": metrics,
    "Original_Test": [original.iloc[0][metric] for metric in metrics],
    "Temporal_Test": [temporal.iloc[0][metric] for metric in metrics]
})

comparison["Original_Test"] = comparison["Original_Test"].round(4)
comparison["Temporal_Test"] = comparison["Temporal_Test"].round(4)
comparison["Difference_Temporal_Minus_Original"] = (
    comparison["Temporal_Test"] - comparison["Original_Test"]
).round(4)

output_file = reports_path / "evaluation_comparison.csv"
comparison.to_csv(output_file, index=False)

print("\nEVALUATION COMPARISON COMPLETED")
print("\nOriginal model:", original.iloc[0]["Model"])
print("Temporal model:", temporal.iloc[0]["Model"])
print("\nMetric comparison:")
print(comparison.to_string(index=False))
print("\nComparison report saved to:", output_file)
