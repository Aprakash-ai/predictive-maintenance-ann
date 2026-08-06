from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data


def main():

    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=" * 70)
    print("TRAINING SET")
    print("=" * 70)

    print("X_train :", X_train.shape)
    print("y_train :", y_train.shape)

    print()

    print("=" * 70)
    print("TESTING SET")
    print("=" * 70)

    print("X_test  :", X_test.shape)
    print("y_test  :", y_test.shape)


if __name__ == "__main__":
    main()