import os
import joblib
import matplotlib.pyplot as plt
from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features, train_model
from src.model import build_model
from src.evaluate import evaluate_model


def main():

    # load dataset
    df = load_dataset()

    # preprocess dataset
    X, y = preprocess_data(df)

    # data  split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # feature scaling
    X_train, X_test, scaler = scale_features(X_train, X_test)


    model = build_model(input_dim=X_train.shape[1])

    # train ANN
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

    # saved model and scaler
    os.makedirs("saved_models", exist_ok=True)

    # Save trained ANN model
    model.save("saved_models/predictive_maintenance_ann.keras")

    # Save fitted StandardScaler
    joblib.dump(
        scaler,
        "saved_models/scaler.pkl"
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED SUCCESSFULLY")
    print("=" * 70)
    print("ANN Model : saved_models/predictive_maintenance_ann.keras")
    print("Scaler    : saved_models/scaler.pkl")

    y_pred = evaluate_model(
        model,
        X_test,
        y_test
    ) # evaluate model


if __name__ == "__main__":
    main()