import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf

from src.sequence_preprocessing import prepare_sequence_data
from src.lstm_transformer_model import build_lstm_transformer_model


# Prepare sequence data
X_train, X_test, y_train, y_test, scaler = prepare_sequence_data(
    sequence_length=10
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# Compute class weights
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("Class weights:", class_weights)


# Build model
model = build_lstm_transformer_model(
    input_shape=(X_train.shape[1], X_train.shape[2])
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)


# Predictions
y_prob = model.predict(X_test)
y_pred = (y_prob >= 0.5).astype(int).ravel()


# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))