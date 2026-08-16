import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data


def main():

    # load and preprocess data
    df = load_dataset()

    X, y = preprocess_data(df)

    #split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # xgboost parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # best xgboost model
    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    )

    # grid search
    print("\n" + "=" * 70)
    print("XGBOOST HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train.to_numpy(),
        y_train
    )

    # best parameters
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)

    print(grid_search.best_params_)

    print("\nBest Cross-Validation F1 Score:")
    print(f"{grid_search.best_score_:.4f}")

    #best model
    best_model = grid_search.best_estimator_

    # test prediction
    y_pred = best_model.predict(
        X_test.to_numpy()
    )

    # evaluation
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("OPTIMIZED XGBOOST EVALUATION")
    print("=" * 70)

    print(f"\nTest Accuracy : {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(confusion_matrix(
        y_test,
        y_pred
    ))

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Failure", "Failure"]
    ))


if __name__ == "__main__":
    main()