import matplotlib.pyplot as plt
import shap

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.ml_models import build_xgboost


def main():
    # Load dataset
    df = load_dataset()

    # Preprocess data
    X, y = preprocess_data(df)

    # Train optimized XGBoost
    model = build_xgboost()
    model.fit(X.to_numpy(), y.to_numpy())

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X.to_numpy())

    # -------------------------------
    # 1. SHAP Feature Importance
    # -------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        show=False
    )
    plt.title("SHAP Feature Importance - Optimized XGBoost")
    plt.tight_layout()
    plt.savefig("../graphs/shap_feature_importance.png", dpi=300)
    plt.close()

    # -------------------------------
    # 2. SHAP Summary Plot
    # -------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )
    plt.title("SHAP Summary Plot - Optimized XGBoost")
    plt.tight_layout()
    plt.savefig("../graphs/shap_summary_plot.png", dpi=300)
    plt.close()

    print("\nXAI analysis completed successfully.")
    print("Generated:")
    print("1. shap_feature_importance.png")
    print("2. shap_summary_plot.png")

    # -------------------------------
    # 3. Local SHAP Explanation
    # -------------------------------

    # Find one actual failure sample
    failure_indices = y[y == 1].index
    sample_index = failure_indices[0]

    sample_position = X.index.get_loc(sample_index)
    sample = X.iloc[[sample_position]]

    # Calculate SHAP values for this sample
    sample_shap = explainer.shap_values(sample.to_numpy())

    # Waterfall plot
    shap_explanation = shap.Explanation(
        values=sample_shap[0],
        base_values=explainer.expected_value,
        data=sample.iloc[0].to_numpy(),
        feature_names=X.columns
    )

    shap.plots.waterfall(
        shap_explanation,
        show=False
    )

    plt.tight_layout()
    plt.savefig("../graphs/shap_local_explanation.png", dpi=300)
    plt.close()

    print("3. shap_local_explanation.png")

    # -------------------------------
    # 4. SHAP Dependence Plot
    # -------------------------------

    shap.dependence_plot(
        "Torque [Nm]",
        shap_values,
        X,
        show=False
    )

    plt.title("SHAP Dependence Plot - Torque [Nm]")
    plt.tight_layout()
    plt.savefig("../graphs/shap_dependence_torque.png", dpi=300)
    plt.close()

    print("4. shap_dependence_torque.png")

    # -------------------------------
    # 5. SHAP Interaction Analysis
    # -------------------------------

    interaction_values = explainer.shap_interaction_values(
        X.to_numpy()
    )

    # Mean absolute interaction strength
    mean_interactions = abs(interaction_values).mean(axis=0)

    # Plot interaction matrix
    plt.figure(figsize=(10, 8))

    plt.imshow(mean_interactions, aspect="auto")

    plt.xticks(
        range(len(X.columns)),
        X.columns,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(len(X.columns)),
        X.columns
    )

    plt.colorbar(label="Mean |SHAP interaction value|")

    plt.title("SHAP Feature Interaction Matrix - Optimized XGBoost")
    plt.tight_layout()

    plt.savefig(
        "../graphs/shap_interaction_matrix.png",
        dpi=300
    )

    plt.close()

    print("5. shap_interaction_matrix.png")


if __name__ == "__main__":
    main()