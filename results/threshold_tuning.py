from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "threshold_tuning_results.csv"
)
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features
from src.ml_models import build_xgboost


def main():

    # load and preprocess data
    df = load_dataset()

    X, y = preprocess_data(df)

    #split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # scale features
    X_train, X_test, scaler = scale_features(
        X_train,
        X_test
    )

    # build baseline XGBoost
    model = build_xgboost()

    # train model
    print("\n" + "=" * 70)
    print("XGBOOST TRAINING FOR THRESHOLD TUNING")
    print("=" * 70)

    model.fit(
        X_train.to_numpy(),
        y_train
    )

    # get failure probabilities
    probabilities = model.predict_proba(
        X_test.to_numpy()
    )[:, 1]

    # test different thresholds
    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    results = []

    for threshold in thresholds:

        y_pred = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        results.append({
            "Threshold": round(threshold, 2),
            "Failure_Precision": round(precision, 4),
            "Failure_Recall": round(recall, 4),
            "Failure_F1": round(f1, 4)
        })

    # create results table
    results_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 70)

    print(results_df.to_string(index=False))

    # save results
    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\nThreshold results saved successfully.")


if __name__ == "__main__":
    main()