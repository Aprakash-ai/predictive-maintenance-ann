import os
import pandas as pd
import matplotlib.pyplot as plt
os.makedirs("../graphs", exist_ok=True)


df = pd.read_csv(
    "sequence_model_comparison.csv"
)

print("\n" + "=" * 70)
print("SEQUENCE MODEL COMPARISON")
print("=" * 70)

print(df.to_string(index=False))

# Accuracy Comparison
plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("RNN vs LSTM vs GRU - Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "../graphs/sequence_model_accuracy.png"
)

plt.show()

# failure recall comparison
plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Failure_Recall"]
)

plt.title("RNN vs LSTM vs GRU - Failure Recall")
plt.xlabel("Model")
plt.ylabel("Failure Recall")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "../graphs/sequence_model_failure_recall.png"
)

plt.show()

# Failure F1 Comparison
plt.figure(figsize=(9, 6))

plt.bar(
    df["Model"],
    df["Failure_F1"]
)

plt.title("RNN vs LSTM vs GRU - Failure F1")
plt.xlabel("Model")
plt.ylabel("Failure F1")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    "../graphs/sequence_model_failure_f1.png"
)

plt.show()


print("\n" + "=" * 70)
print("SEQUENCE MODEL GRAPHS SAVED SUCCESSFULLY")
print("=" * 70)