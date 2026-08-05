from src.data_loader import load_dataset
from src.preprocessing import preprocess_data


def main():

    df = load_dataset()

    print("=" * 60)
    print("ORIGINAL DATASET")
    print("=" * 60)
    print(df.head())

    df = preprocess_data(df)

    print("\n" + "=" * 60)
    print("PREPROCESSED DATASET")
    print("=" * 60)
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nShape:")
    print(df.shape)


if __name__ == "__main__":
    main()