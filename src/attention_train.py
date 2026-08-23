import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from src.sequence_preprocessing import prepare_sequence_data
from src.attention_model import build_attention_model


def main():

    print("\n" + "=" * 70)
    print("ATTENTION MODEL TRAINING")
    print("=" * 70)

    # Sequence data preprocessing
    X_train, X_test, y_train, y_test, scaler = (
        prepare_sequence_data()
    )

    print("\nTraining Data:", X_train.shape)
    print("Testing Data :", X_test.shape)

    # class weights
    classes = np.unique(y_train)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(classes, weights)
    )

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    print(class_weights)

    # build attention model
    model = build_attention_model()

    print("\n" + "=" * 70)
    print("ATTENTION MODEL SUMMARY")
    print("=" * 70)

    model.summary()

    # train_model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=30,
        batch_size=32,
        class_weight=class_weights,
        verbose=1
    )

    # Prediction
    y_probability = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (
        y_probability >= 0.5
    ).astype(int)

    # Evaluation
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n" + "=" * 70)
    print("ATTENTION MODEL EVALUATION")
    print("=" * 70)

    print(
        f"\nTest Accuracy : {accuracy:.4f}"
    )

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(cm)

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
            ]
        )
    )


if __name__ == "__main__":
    main()