import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():

    # project paths
    project_root = Path(__file__).resolve().parent.parent

    input_path = (
        project_root
        / "results"
        / "model_comparison.csv"
    )

    output_path = (
        project_root
        / "graphs"
        / "final_model_comparison.png"
    )

    # load comparison results
    df = pd.read_csv(input_path)

    print("\n" + "=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    print(df.to_string(index=False))

    # plot failure F1
    plt.figure(figsize=(10, 6))

    plt.bar(
        df["Model"],
        df["Failure_F1"]
    )

    plt.title("Model Comparison - Failure F1 Score")
    plt.xlabel("Model")
    plt.ylabel("Failure F1 Score")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.show()

    print("\nFinal comparison graph saved")
    print(output_path)


if __name__ == "__main__":
    main()