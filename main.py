from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features


def main():

    # Load Dataset
    df = load_dataset()

    # Preprocess Dataset
    X, y = preprocess_data(df)

    # Train-Test Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Feature Scaling
    X_train, X_test, scaler = scale_features(X_train, X_test)

    print("=" * 70)
    print("TRAINING DATA")
    print("=" * 70)
    print(X_train.head())

    print("\nTraining Shape :", X_train.shape)

    print("\n" + "=" * 70)
    print("TEST DATA")
    print("=" * 70)
    print(X_test.head())

    print("\nTesting Shape :", X_test.shape)


if __name__ == "__main__":
    main()