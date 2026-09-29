import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.sequence_preprocessing import prepare_sequence_data
from src.bilstm_attention_model import build_bilstm_attention_model


def main():

    print("\n" + "=" * 70)
    print("BiLSTM + ATTENTION TRAINING")
    print("=" * 70)

    # Load sequence data
    X_train, X_test, y_train, y_test, scaler = (
        prepare_sequence_data()
    )

    print("\nTraining Data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTesting Data:")
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # Class weights
    classes = np.unique(y_train)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = dict(
        zip(classes, class_weights_array)
    )

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)

    print(class_weights)

    # Build model
    model = build_bilstm_attention_model(
        sequence_length=X_train.shape[1],
        n_features=X_train.shape[2]
    )

    print("\n" + "=" * 70)
    print("MODEL")
    print("=" * 70)

    model.summary()

    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    # Prediction
    y_prob = model.predict(
        X_test,
        verbose=0
    ).ravel()

    y_pred = (
        y_prob >= 0.5
    ).astype(int)

    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

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

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Results
    print("\n" + "=" * 70)
    print("BiLSTM + ATTENTION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()