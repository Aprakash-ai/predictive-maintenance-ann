from src.data_loader import load_dataset
from src.preprocessing import preprocess_data


def main():

    # Load Dataset
    df = load_dataset()

    # Preprocess Dataset
    X, y = preprocess_data(df)

    print("=" * 60)
    print("INPUT FEATURES (X)")
    print("=" * 60)
    print(X.head())

    print("\nShape of X:", X.shape)

    print("\n" + "=" * 60)
    print("TARGET VARIABLE (y)")
    print("=" * 60)
    print(y.head())

    print("\nShape of y:", y.shape)


if __name__ == "__main__":
    main()