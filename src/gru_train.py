import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.sequence_preprocessing import prepare_sequence_data
from src.gru_model import build_gru_model


def main():

    print("\n" + "=" * 70)
    print("GRU TRAINING")
    print("=" * 70)

    # Prepare sequence data
    X_train, X_test, y_train, y_test, scaler = (
        prepare_sequence_data()
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data :", X_test.shape)

    # Handle class imbalance
    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weight_dict = dict(
        zip(classes, class_weights)
    )

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    print(class_weight_dict)

    # Build GRU
    model = build_gru_model(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    print("\n" + "=" * 70)
    print("GRU MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # Train GRU
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        class_weight=class_weight_dict,
        verbose=1
    )

    # Prediction
    print("\n" + "=" * 70)
    print("GRU EVALUATION")
    print("=" * 70)

    y_probability = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (
        y_probability >= 0.5
    ).astype(int)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        f"\nTest Accuracy : {accuracy:.4f}"
    )

    # Confusion Matrix
    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # Classification Report
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Failure",
                "Failure"
            ],
            digits=2
        )
    )


if __name__ == "__main__":
    main()