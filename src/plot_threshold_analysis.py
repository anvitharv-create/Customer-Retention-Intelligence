
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = Path("reports/threshold_analysis.csv")
output_path = Path("reports/threshold_analysis.png")

df = pd.read_csv(input_path)

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

ax.plot(
    df["Threshold"],
    df["Precision"],
    marker="o",
    linewidth=2.5,
    markersize=7,
    color="#E76F51",
    label="Precision"
)

ax.plot(
    df["Threshold"],
    df["Recall"],
    marker="o",
    linewidth=2.5,
    markersize=7,
    color="#2A9D8F",
    label="Recall"
)

ax.plot(
    df["Threshold"],
    df["F1_Score"],
    marker="o",
    linewidth=2.5,
    markersize=7,
    color="#457B9D",
    label="F1-score"
)

ax.set_title(
    "Customer Churn — Classification Threshold Analysis",
    fontsize=15,
    fontweight="bold",
    color="#0F172A",
    pad=18
)

ax.set_xlabel("Decision Threshold", fontsize=11)
ax.set_ylabel("Metric Score", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_xticks(df["Threshold"])
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(frameon=False, ncol=3, loc="lower center")
plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
    facecolor=fig.get_facecolor()
)

plt.show()

print("Chart saved to:", output_path)
