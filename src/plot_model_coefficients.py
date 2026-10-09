
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

input_path = Path("reports/model_coefficients.csv")
output_path = Path("reports/model_coefficients.png")

df = pd.read_csv(input_path)
df = df.sort_values("Coefficient")

colors = [
    "#E76F51" if value > 0 else "#2A9D8F"
    for value in df["Coefficient"]
]

fig, ax = plt.subplots(figsize=(11, 7))

fig.patch.set_facecolor("#F8FAFC")
ax.set_facecolor("#F8FAFC")

bars = ax.barh(
    df["Feature"],
    df["Coefficient"],
    color=colors,
    height=0.65,
    edgecolor="white",
    linewidth=1.2
)

ax.axvline(0, color="#475569", linewidth=1.2)

ax.grid(axis="x", linestyle="--", alpha=0.25)
ax.set_axisbelow(True)

for bar, value in zip(bars, df["Coefficient"]):
    offset = 0.025 if value >= 0 else -0.025
    alignment = "left" if value >= 0 else "right"

    ax.text(
        value + offset,
        bar.get_y() + bar.get_height() / 2,
        f"{value:+.3f}",
        va="center",
        ha=alignment,
        fontsize=10,
        fontweight="bold",
        color="#1E293B"
    )

ax.set_title(
    "Customer Churn — Model Coefficients",
    fontsize=17,
    fontweight="bold",
    color="#0F172A",
    pad=20
)

ax.set_xlabel(
    "Logistic Regression Coefficient",
    fontsize=11,
    color="#334155",
    labelpad=10
)

ax.set_ylabel("Customer Feature", fontsize=11, color="#334155", labelpad=10)

ax.tick_params(axis="both", labelsize=10, colors="#334155")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#CBD5E1")

ax.set_xlim(
    min(0, df["Coefficient"].min()) - 0.15,
    max(0, df["Coefficient"].max()) + 0.15
)

fig.text(
    0.5,
    0.015,
    "Teal: associated with lower predicted churn  |  Coral: associated with higher predicted churn",
    ha="center",
    fontsize=9,
    color="#64748B"
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

print("Aesthetic coefficient chart saved to:", output_path)
