from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.train import split_data


def main():

    # load dataset
    df = load_dataset()

    # preprocess dataset
    X, y = preprocess_data(df)

    # split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # class distribution
    print("\n" + "=" * 70)
    print("CLASS IMBALANCE ANALYSIS")
    print("=" * 70)

    print("\nComplete Dataset:")
    print(y.value_counts().sort_index())

    print("\nTraining Dataset:")
    print(y_train.value_counts().sort_index())

    print("\nTest Dataset:")
    print(y_test.value_counts().sort_index())

    # class percentages
    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION (%)")
    print("=" * 70)

    train_percentages = (
        y_train.value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print(train_percentages)

    # imbalance ratio
    class_counts = y_train.value_counts()

    majority_class = class_counts.max()
    minority_class = class_counts.min()

    imbalance_ratio = majority_class / minority_class

    print("\n" + "=" * 70)
    print("IMBALANCE RATIO")
    print("=" * 70)

    print(f"Majority class samples : {majority_class}")
    print(f"Minority class samples : {minority_class}")
    print(f"Imbalance ratio        : {imbalance_ratio:.2f}:1")


if __name__ == "__main__":
    main()