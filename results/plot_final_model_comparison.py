from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    # ----------------------------------------------------------
    # Paths
    # ----------------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent

    results_dir = project_root / "results"
    graphs_dir = project_root / "graphs"

    graphs_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------------------------------------
    # Load data
    # ----------------------------------------------------------

    csv_path = (
        results_dir
        / "final_model_comparison.csv"
    )

    df = pd.read_csv(csv_path)

    # ----------------------------------------------------------
    # Select important models
    # ----------------------------------------------------------

    selected_models = [
        "Optimized XGBoost",
        "RNN",
        "LSTM",
        "GRU",
        "Attention-LSTM",
        "Transformer"
    ]

    comparison_df = df[
        df["Model"].isin(selected_models)
    ].copy()

    # ----------------------------------------------------------
    # Failure Recall
    # ----------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["Failure_Recall"]
    )

    plt.xlabel("Model")
    plt.ylabel("Failure Recall")
    plt.title(
        "Final Failure Detection Recall Comparison"
    )

    plt.ylim(0, 1)

    plt.xticks(rotation=25)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        graphs_dir
        / "final_failure_recall_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ----------------------------------------------------------
    # Failure F1
    # ----------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["Failure_F1"]
    )

    plt.xlabel("Model")
    plt.ylabel("Failure F1-Score")
    plt.title(
        "Final Failure Detection F1 Comparison"
    )

    plt.ylim(0, 1)

    plt.xticks(rotation=25)

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        graphs_dir
        / "final_failure_f1_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 70)
    print("FINAL GRAPHS GENERATED")
    print("=" * 70)

    print(
        graphs_dir
        / "final_failure_recall_comparison.png"
    )

    print(
        graphs_dir
        / "final_failure_f1_comparison.png"
    )


if __name__ == "__main__":
    main()