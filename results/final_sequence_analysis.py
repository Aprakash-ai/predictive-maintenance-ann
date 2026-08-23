from pathlib import Path

import pandas as pd


def main():

    print("\n" + "=" * 70)
    print("FINAL SEQUENCE MODEL ANALYSIS")
    print("=" * 70)

    # ----------------------------------------------------------
    # Project paths
    # ----------------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    results_dir = project_root / "results"

    csv_path = (
        results_dir
        / "sequence_model_comparison.csv"
    )

    # ----------------------------------------------------------
    # Load sequence model results
    # ----------------------------------------------------------

    df = pd.read_csv(csv_path)

    print("\n" + "=" * 70)
    print("SEQUENCE MODELS")
    print("=" * 70)

    print(
        df.to_string(index=False)
    )

    # ----------------------------------------------------------
    # Best models
    # ----------------------------------------------------------

    best_accuracy = df.loc[
        df["Accuracy"].idxmax()
    ]

    best_recall = df.loc[
        df["Failure_Recall"].idxmax()
    ]

    best_f1 = df.loc[
        df["Failure_F1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print(
        f"\nBest Accuracy : "
        f"{best_accuracy['Model']} "
        f"({best_accuracy['Accuracy'] * 100:.2f}%)"
    )

    print(
        f"Best Failure Recall : "
        f"{best_recall['Model']} "
        f"({best_recall['Failure_Recall'] * 100:.2f}%)"
    )

    print(
        f"Best Failure F1 : "
        f"{best_f1['Model']} "
        f"({best_f1['Failure_F1']:.2f})"
    )

    # ----------------------------------------------------------
    # Optimized XGBoost reference
    # ----------------------------------------------------------

    xgb_accuracy = 0.9890
    xgb_recall = 0.74
    xgb_f1 = 0.82

    # ----------------------------------------------------------
    # Final comparison
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print("\nOptimized XGBoost:")
    print(
        f"Accuracy       : "
        f"{xgb_accuracy * 100:.2f}%"
    )
    print(
        f"Failure Recall : "
        f"{xgb_recall * 100:.2f}%"
    )
    print(
        f"Failure F1     : "
        f"{xgb_f1:.2f}"
    )

    print("\nBest Sequence Model:")

    print(
        f"Model          : "
        f"{best_recall['Model']}"
    )

    print(
        f"Accuracy       : "
        f"{best_recall['Accuracy'] * 100:.2f}%"
    )

    print(
        f"Failure Recall : "
        f"{best_recall['Failure_Recall'] * 100:.2f}%"
    )

    print(
        f"Failure F1     : "
        f"{best_recall['Failure_F1']:.2f}"
    )

    # ----------------------------------------------------------
    # Performance differences
    # ----------------------------------------------------------

    accuracy_difference = (
        xgb_accuracy
        - best_recall["Accuracy"]
    )

    f1_difference = (
        xgb_f1
        - best_recall["Failure_F1"]
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE DIFFERENCE")
    print("=" * 70)

    print(
        f"\nXGBoost Accuracy Advantage : "
        f"{accuracy_difference * 100:.2f} percentage points"
    )

    print(
        f"XGBoost Failure Recall Difference : "
        f"{(xgb_recall - best_recall['Failure_Recall']) * 100:.2f} percentage points"
    )

    print(
        f"XGBoost Failure F1 Advantage : "
        f"{f1_difference:.2f}"
    )

    # ----------------------------------------------------------
    # Final conclusion
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    print(
        """
Sequence models were evaluated to determine whether temporal
patterns could improve predictive maintenance performance.

Among the evaluated sequence and advanced deep learning models,
the Transformer achieved the highest failure recall of 74%.
This indicates that the Transformer was effective at identifying
machine failure cases.

However, the optimized XGBoost model achieved the same failure
recall while providing substantially higher overall accuracy and
failure-class F1-score.

Optimized XGBoost achieved 98.90% accuracy, 74% failure recall,
and 0.82 failure F1-score, whereas the Transformer achieved
89.64% accuracy, 74% failure recall, and 0.22 failure F1-score.

Therefore, the optimized XGBoost model remains the final
predictive maintenance model for this project, while the
Transformer represents the strongest advanced deep learning
sequence model evaluated during Phase 10.
"""
    )


if __name__ == "__main__":
    main()