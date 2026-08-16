import pandas as pd


def main():

    print("\n" + "=" * 70)
    print("FINAL SEQUENCE MODEL ANALYSIS")
    print("=" * 70)

    # Sequence model results
    sequence_models = pd.DataFrame({
        "Model": ["RNN", "LSTM", "GRU"],
        "Accuracy": [0.8969, 0.8504, 0.8599],
        "Failure_Recall": [0.28, 0.49, 0.44],
        "Failure_F1": [0.10, 0.11, 0.11]
    })

    print("\nSEQUENCE MODELS")
    print(sequence_models.to_string(index=False))

    # Best models
    best_accuracy = sequence_models.loc[
        sequence_models["Accuracy"].idxmax()
    ]

    best_recall = sequence_models.loc[
        sequence_models["Failure_Recall"].idxmax()
    ]

    best_f1 = sequence_models.loc[
        sequence_models["Failure_F1"].idxmax()
    ]

    print("\n" + "=" * 70)
    print("BEST RESULTS")
    print("=" * 70)

    print(
        f"\nBest Accuracy : "
        f"{best_accuracy['Model']} "
        f"({best_accuracy['Accuracy']:.2%})"
    )

    print(
        f"Best Failure Recall : "
        f"{best_recall['Model']} "
        f"({best_recall['Failure_Recall']:.2%})"
    )

    print(
        f"Best Failure F1 : "
        f"{best_f1['Model']} "
        f"({best_f1['Failure_F1']:.2f})"
    )

    # Final ML model
    xgboost_accuracy = 0.9890
    xgboost_recall = 0.74
    xgboost_f1 = 0.82

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print("\nOptimized XGBoost:")
    print(f"Accuracy       : {xgboost_accuracy:.2%}")
    print(f"Failure Recall : {xgboost_recall:.2%}")
    print(f"Failure F1     : {xgboost_f1:.2f}")

    print("\nBest Sequence Model:")
    print(f"Model          : {best_recall['Model']}")
    print(f"Accuracy       : {best_recall['Accuracy']:.2%}")
    print(f"Failure Recall : {best_recall['Failure_Recall']:.2%}")
    print(f"Failure F1     : {best_recall['Failure_F1']:.2f}")

    print("\n" + "=" * 70)
    print("FINAL CONCLUSION")
    print("=" * 70)

    print("""
Sequence models were evaluated to determine whether temporal
patterns improve predictive maintenance performance.

Among the sequence models, LSTM achieved the highest failure
recall (49%), while RNN achieved the highest overall accuracy
(89.69%).

However, the optimized XGBoost model significantly outperformed
all sequence models, achieving 98.90% accuracy, 74% failure recall,
and 0.82 failure F1-score.

Therefore, optimized XGBoost remains the final predictive
maintenance model for this project.
""")


if __name__ == "__main__":
    main()