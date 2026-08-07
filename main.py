import os
import joblib
import matplotlib.pyplot as plt

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features, train_model
from src.model import build_model
from src.evaluate import evaluate_model
from src.predict import predict_failure


def train_pipeline():
    """
    Complete ANN Training Pipeline
    """

    # ==========================================================
    # Load Dataset
    # ==========================================================

    df = load_dataset()

    # ==========================================================
    # Data Preprocessing
    # ==========================================================

    X, y = preprocess_data(df)

    # ==========================================================
    # Train-Test Split
    # ==========================================================

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ==========================================================
    # Feature Scaling
    # ==========================================================

    X_train, X_test, scaler = scale_features(X_train, X_test)

    # ==========================================================
    # Build ANN Model
    # ==========================================================

    model = build_model(input_dim=X_train.shape[1])

    # ==========================================================
    # Train ANN Model
    # ==========================================================

    history = train_model(
        model,
        X_train,
        y_train
    )

    # ==========================================================
    # Plot Training Accuracy
    # ==========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()

    plt.savefig("graphs/training_accuracy.png")

    plt.show()

    # ==========================================================
    # Plot Training Loss
    # ==========================================================

    plt.figure(figsize=(8, 5))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.savefig("graphs/training_loss.png")

    plt.show()

    # ==========================================================
    # Save Model and Scaler
    # ==========================================================

    os.makedirs("saved_models", exist_ok=True)

    model.save("saved_models/predictive_maintenance_ann.keras")

    joblib.dump(
        scaler,
        "saved_models/scaler.pkl"
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)
    print("ANN Model : saved_models/predictive_maintenance_ann.keras")
    print("Scaler    : saved_models/scaler.pkl")

    # ==========================================================
    # Evaluate Model
    # ==========================================================

    evaluate_model(
        model,
        X_test,
        y_test
    )

def main():

    print("\n" + "=" * 70)
    print("      PREDICTIVE MAINTENANCE USING ANN")
    print("=" * 70)

    print("1. Train New ANN Model")
    print("2. Predict Machine Failure")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ")

    if choice == "1":
        train_pipeline()

    elif choice == "2":
        predict_failure()

    elif choice == "3":
        print("\nThank you for using Predictive Maintenance ANN.")

    else:
        print("\nInvalid Choice! Please run the program again.")


if __name__ == "__main__":
    main()