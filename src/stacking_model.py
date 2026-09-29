import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.ml_models import build_xgboost
from src.sequence_preprocessing import prepare_sequence_data
from src.lstm_transformer_model import build_lstm_transformer_model


def train_stacking_model():

    # ==================================================
    # 1. Prepare normal ML data
    # ==================================================

    df = load_dataset()
    X_ml, y_ml = preprocess_data(df)

    split_index = int(len(X_ml) * 0.80)

    X_train_ml = X_ml.iloc[:split_index]
    X_test_ml = X_ml.iloc[split_index:]

    y_train_ml = y_ml.iloc[:split_index]
    y_test_ml = y_ml.iloc[split_index:]

    print("ML training shape:", X_train_ml.shape)
    print("ML testing shape:", X_test_ml.shape)

    # ==================================================
    # 2. Prepare sequence data
    # ==================================================

    (
        X_train_seq,
        X_test_seq,
        y_train_seq,
        y_test_seq,
        scaler
    ) = prepare_sequence_data(
        sequence_length=10
    )

    print("Sequence training shape:", X_train_seq.shape)
    print("Sequence testing shape:", X_test_seq.shape)

    # ==================================================
    # 3. Create validation split for stacking
    # ==================================================

    meta_split = int(len(X_train_seq) * 0.80)

    X_base_seq = X_train_seq[:meta_split]
    y_base_seq = y_train_seq[:meta_split]

    X_meta_seq = X_train_seq[meta_split:]
    y_meta_seq = y_train_seq[meta_split:]

    print("\nBase sequence training shape:", X_base_seq.shape)
    print("Meta sequence validation shape:", X_meta_seq.shape)

    # ==================================================
    # 4. Align corresponding ML data
    # ==================================================
    #
    # Sequence target at position i corresponds to
    # original dataset row i + 10.
    #
    # Therefore:
    #
    # Base sequence targets:
    # 10 ... corresponding range
    #
    # Meta sequence targets:
    # corresponding validation range
    #
    # Test sequence targets:
    # original rows 8002 ... 9999
    #
    # ML test starts at row 8000, so first two rows
    # must be removed.
    # ==================================================

    sequence_length = 10

    base_start = sequence_length
    base_end = base_start + meta_split

    meta_start = base_end
    meta_end = meta_start + len(X_meta_seq)

    # ML data corresponding to base sequence targets
    X_base_ml = X_ml.iloc[
        base_start:base_end
    ]

    y_base_ml = y_ml.iloc[
        base_start:base_end
    ]

    # ML data corresponding to meta validation targets
    X_meta_ml = X_ml.iloc[
        meta_start:meta_end
    ]

    y_meta_ml = y_ml.iloc[
        meta_start:meta_end
    ]

    # ML data corresponding to sequence test targets
    test_target_start = split_index + 2

    X_stack_test_ml = X_ml.iloc[
        test_target_start:
    ]

    y_stack_test_ml = y_ml.iloc[
        test_target_start:
    ]

    print("\nAligned base ML shape:", X_base_ml.shape)
    print("Aligned meta ML shape:", X_meta_ml.shape)
    print("Aligned test ML shape:", X_stack_test_ml.shape)

    # ==================================================
    # 5. Train XGBoost base model
    # ==================================================

    print("\nTraining XGBoost base model...")

    xgb_model = build_xgboost()

    xgb_model.fit(
        X_base_ml.to_numpy(),
        y_base_ml.to_numpy()
    )

    # Prediction for meta validation set
    xgb_meta_prob = xgb_model.predict_proba(
        X_meta_ml.to_numpy()
    )[:, 1]

    # ==================================================
    # 6. Train LSTM + Transformer base model
    # ==================================================

    print("\nTraining LSTM + Transformer base model...")

    dl_model = build_lstm_transformer_model(
        input_shape=(
            X_base_seq.shape[1],
            X_base_seq.shape[2]
        )
    )

    dl_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # Class weights
    classes = np.unique(y_base_seq)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_base_seq
    )

    class_weights = dict(
        zip(classes, weights)
    )

    print("DL class weights:", class_weights)

    dl_model.fit(
        X_base_seq,
        y_base_seq,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weights,
        verbose=1
    )

    # Prediction for meta validation set
    dl_meta_prob = dl_model.predict(
        X_meta_seq,
        verbose=0
    ).ravel()

    # ==================================================
    # 7. Create meta-training features
    # ==================================================

    meta_train = np.column_stack([
        xgb_meta_prob,
        dl_meta_prob
    ])

    y_meta = y_meta_seq

    print("\nMeta-training shape:", meta_train.shape)
    print("Meta-training target shape:", y_meta.shape)

    # ==================================================
    # 8. Train balanced meta-classifier
    # ==================================================

    print("\nTraining meta-classifier...")

    meta_model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )

    meta_model.fit(
        meta_train,
        y_meta
    )

    # ==================================================
    # 9. Re-train XGBoost on complete ML training data
    # ==================================================

    print("\nRetraining XGBoost on complete training data...")

    final_xgb_model = build_xgboost()

    final_xgb_model.fit(
        X_train_ml.to_numpy(),
        y_train_ml.to_numpy()
    )

    # ==================================================
    # 10. Re-train LSTM + Transformer on complete
    #     sequence training data
    # ==================================================

    print("\nRetraining LSTM + Transformer on complete training data...")

    final_dl_model = build_lstm_transformer_model(
        input_shape=(
            X_train_seq.shape[1],
            X_train_seq.shape[2]
        )
    )

    final_dl_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    full_classes = np.unique(y_train_seq)

    full_weights = compute_class_weight(
        class_weight="balanced",
        classes=full_classes,
        y=y_train_seq
    )

    full_class_weights = dict(
        zip(full_classes, full_weights)
    )

    print(
        "Full DL class weights:",
        full_class_weights
    )

    final_dl_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        class_weight=full_class_weights,
        verbose=1
    )

    # ==================================================
    # 11. Generate final test probabilities
    # ==================================================

    xgb_test_prob = final_xgb_model.predict_proba(
        X_stack_test_ml.to_numpy()
    )[:, 1]

    dl_test_prob = final_dl_model.predict(
        X_test_seq,
        verbose=0
    ).ravel()

    # ==================================================
    # 12. Final alignment check
    # ==================================================

    min_test_length = min(
        len(xgb_test_prob),
        len(dl_test_prob),
        len(y_test_seq)
    )

    xgb_test_prob = xgb_test_prob[:min_test_length]
    dl_test_prob = dl_test_prob[:min_test_length]
    y_test_final = y_test_seq[:min_test_length]

    # ==================================================
    # 13. Create final meta-test features
    # ==================================================

    meta_test = np.column_stack([
        xgb_test_prob,
        dl_test_prob
    ])

    print("\nFinal meta-test shape:", meta_test.shape)
    print("Final test target shape:", y_test_final.shape)

    # ==================================================
    # 14. Final stacking prediction
    # ==================================================

    y_pred = meta_model.predict(
        meta_test
    )

    # ==================================================
    # 15. Evaluation
    # ==================================================

    accuracy = accuracy_score(
        y_test_final,
        y_pred
    )

    print("\n" + "=" * 50)
    print("FINAL STACKING RESULTS")
    print("=" * 50)

    print("\nAccuracy:", accuracy)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_final,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test_final,
            y_pred
        )
    )

    return meta_model


if __name__ == "__main__":
    train_stacking_model()