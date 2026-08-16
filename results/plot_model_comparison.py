from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_PATH = PROJECT_ROOT / "results" / "model_comparison.csv"
GRAPH_PATH = PROJECT_ROOT / "graphs" / "model_comparison_failure_recall.png"

# load model comparison results
df = pd.read_csv(CSV_PATH)

# plot failure recall
plt.figure(figsize=(10, 6))

plt.bar(
    df["Model"],
    df["Failure_Recall"]
)

plt.title("Model Comparison - Failure Recall")
plt.xlabel("Machine Learning Model")
plt.ylabel("Failure Recall")

plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1)

plt.tight_layout()

# save graph
plt.savefig(GRAPH_PATH)

plt.show()