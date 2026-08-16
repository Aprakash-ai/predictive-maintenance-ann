import numpy as np
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_dataset
from src.sequence_data import create_sequences


def prepare_sequence_data(sequence_length=10):

    # load dataset
    df = load_dataset()

    # create sequences
    X, y = create_sequences(
        df,
        sequence_length=sequence_length
    )

    # chronological train/test split
    split_index = int(len(X) * 0.80)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    # feature scaling
    scaler = StandardScaler()

    # Reshape 3D → 2D for scaler
    n_train_samples, seq_len, n_features = X_train.shape

    X_train_2d = X_train.reshape(
        -1,
        n_features
    )

    X_test_2d = X_test.reshape(
        -1,
        n_features
    )

    # Fit ONLY on training data
    X_train_2d = scaler.fit_transform(
        X_train_2d
    )

    X_test_2d = scaler.transform(
        X_test_2d
    )

    # Reshape back to 3D
    X_train = X_train_2d.reshape(
        n_train_samples,
        seq_len,
        n_features
    )

    X_test = X_test_2d.reshape(
        X_test.shape[0],
        seq_len,
        n_features
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )


def main():

    print("\n" + "=" * 70)
    print("SEQUENCE DATA PREPROCESSING")
    print("=" * 70)

    X_train, X_test, y_train, y_test, scaler = (
        prepare_sequence_data()
    )

    # dataset shapes
    print("\nTraining Data:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nTesting Data:")
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # class distribution
    print("\n" + "=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        "No Failure:",
        np.sum(y_train == 0)
    )

    print(
        "Failure:",
        np.sum(y_train == 1)
    )

    print("\n" + "=" * 70)
    print("TESTING CLASS DISTRIBUTION")
    print("=" * 70)

    print(
        "No Failure:",
        np.sum(y_test == 0)
    )

    print(
        "Failure:",
        np.sum(y_test == 1)
    )

    # scaled sample
    print("\n" + "=" * 70)
    print("FIRST SCALED SEQUENCE")
    print("=" * 70)

    print(X_train[0])


if __name__ == "__main__":
    main()