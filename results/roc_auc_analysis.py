from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_curve, roc_auc_score

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features

from src.ml_models import (
    build_logistic_regression,
    build_decision_tree,
    build_random_forest,
    build_svm,
    build_xgboost
)


def main():

    print("\n" + "=" * 70)
    print("ROC-AUC ANALYSIS")
    print("=" * 70)

    # ==========================================================
    # Project directories
    # ==========================================================

    project_root = Path(__file__).resolve().parent.parent

    graphs_dir = project_root / "graphs"
    results_dir = project_root / "results"

    graphs_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # ==========================================================
    # Load dataset
    # ==========================================================

    df = load_dataset()

    # ==========================================================
    # Preprocess dataset
    # ==========================================================

    X, y = preprocess_data(df)

    # ==========================================================
    # Train-test split
    # ==========================================================

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ==========================================================
    # Feature scaling
    # ==========================================================

    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()

    # ==========================================================
    # Models
    # ==========================================================

    models = {
        "Logistic Regression": build_logistic_regression(),
        "Decision Tree": build_decision_tree(),
        "Random Forest": build_random_forest(),
        "SVM": build_svm(),
        "XGBoost": build_xgboost()
    }

    results = []

    # ==========================================================
    # Train models and calculate ROC-AUC
    # ==========================================================

    for name, model in models.items():

        print("\n" + "-" * 70)
        print(f"Training: {name}")
        print("-" * 70)

        model.fit(X_train, y_train)

        # ------------------------------------------------------
        # Get prediction scores
        # ------------------------------------------------------

        if hasattr(model, "predict_proba"):

            y_score = model.predict_proba(
                X_test
            )[:, 1]

        else:

            y_score = model.decision_function(
                X_test
            )

        # ------------------------------------------------------
        # ROC-AUC
        # ------------------------------------------------------

        auc_score = roc_auc_score(
            y_test,
            y_score
        )

        fpr, tpr, thresholds = roc_curve(
            y_test,
            y_score
        )

        results.append({
            "Model": name,
            "ROC_AUC": auc_score
        })

        print(f"ROC-AUC : {auc_score:.4f}")

        # ------------------------------------------------------
        # Plot ROC curve
        # ------------------------------------------------------

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.3f})"
        )

    # ==========================================================
    # Random classifier reference line
    # ==========================================================

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    # ==========================================================
    # Graph formatting
    # ==========================================================

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Machine Failure Prediction")
    plt.legend()
    plt.grid(True)

    # ==========================================================
    # Save ROC curve
    # ==========================================================

    roc_curve_path = graphs_dir / "roc_curve_all_models.png"

    plt.savefig(
        roc_curve_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ==========================================================
    # Save ROC-AUC results
    # ==========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="ROC_AUC",
        ascending=False
    )

    roc_auc_csv_path = results_dir / "roc_auc_results.csv"

    results_df.to_csv(
        roc_auc_csv_path,
        index=False
    )

    # ==========================================================
    # Display results
    # ==========================================================

    print("\n" + "=" * 70)
    print("ROC-AUC RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print("\nROC curve saved:")
    print(roc_curve_path)

    print("\nROC-AUC results saved:")
    print(roc_auc_csv_path)


if __name__ == "__main__":
    main()