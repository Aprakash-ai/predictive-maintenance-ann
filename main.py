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

    y_pred = evaluate_model(
        model,
        X_test,
        y_test
    ) # evaluate model


if __name__ == "__main__":
    main()