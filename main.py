from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data, scale_features, train_model
from src.model import build_model


def main():

    # ---------------------------------
    # Load Dataset
    # ---------------------------------

    df = load_dataset()

    # ---------------------------------
    # Preprocess Dataset
    # ---------------------------------

    X, y = preprocess_data(df)

    # ---------------------------------
    # Train-Test Split
    # ---------------------------------

    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---------------------------------
    # Feature Scaling
    # ---------------------------------

    X_train, X_test, scaler = scale_features(X_train, X_test)

    # ---------------------------------
    # Build ANN
    # ---------------------------------

    model = build_model(input_dim=X_train.shape[1])

    # ---------------------------------
    # Train ANN
    # ---------------------------------

    history = train_model(
        model,
        X_train,
        y_train
    )


if __name__ == "__main__":
    main()