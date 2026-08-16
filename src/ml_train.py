from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features
#from src.ml_models import build_logistic_regression
#from src.ml_models import build_decision_tree
#from src.ml_models import build_random_forest
#from src.ml_models import build_svm
from src.ml_models import build_xgboost
#from src.ml_models import build_xgboost_balanced
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


def main():

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

    # ==========================================================
    # Build model
    # ==========================================================

    #model = build_logistic_regression()
    #model = build_decision_tree()
    #model = build_random_forest()
    #model = build_svm()
    model = build_xgboost()

    '''
    # calculate class imbalance ratio
    class_counts = y_train.value_counts()

    majority_class = class_counts.max()
    minority_class = class_counts.min()

    scale_pos_weight = majority_class / minority_class

    print("\n" + "=" * 70)
    print("XGBOOST CLASS IMBALANCE HANDLING")
    print("=" * 70)

    print(f"Majority class samples : {majority_class}")
    print(f"Minority class samples : {minority_class}")
    print(f"Scale Pos Weight       : {scale_pos_weight:.2f}")

    # build balanced xgboost
    model = build_xgboost_balanced(
        scale_pos_weight=scale_pos_weight
    )
    '''

    # train model
    print("\n" + "=" * 70)
    print("XGBOOST TRAINING")
    print("=" * 70)

    model.fit(X_train.to_numpy(), y_train)

    # ==========================================================
    # Prediction
    # ==========================================================

    y_pred = model.predict(X_test.to_numpy())

    # ==========================================================
    # Evaluation
    # ==========================================================

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 70)
    print("XGBOOST EVALUATION")
    print("=" * 70)

    print(f"\nTest Accuracy : {accuracy:.4f}")

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

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