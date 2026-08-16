import numpy as np
import pandas as pd
from src.data_loader import load_dataset


FEATURE_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
    "Type_H"
]


def create_sequences(df, sequence_length=10):

    # encode machine type
    df = pd.get_dummies(
        df,
        columns=["Type"],
        dtype=int
    )

    # Ensure all Type columns exist
    for column in ["Type_L", "Type_M", "Type_H"]:
        if column not in df.columns:
            df[column] = 0

    # select features
    X = df[FEATURE_COLUMNS].values
    y = df["Machine failure"].values

    sequences = []
    targets = []

    # create sliding-window sequences
    for i in range(len(df) - sequence_length):

        sequences.append(
            X[i:i + sequence_length]
        )

        targets.append(
            y[i + sequence_length]
        )

    return np.array(sequences), np.array(targets)


def main():

    print("\n" + "=" * 70)
    print("SEQUENCE DATASET CREATION")
    print("=" * 70)

    # Load dataset
    df = load_dataset()

    # Create sequences
    X_seq, y_seq = create_sequences(
        df,
        sequence_length=10
    )

    # display results
    print("\nOriginal Dataset Shape:")
    print(df.shape)

    print("\nSequence Dataset Shape:")
    print("X:", X_seq.shape)
    print("y:", y_seq.shape)

    print("\nExpected X shape:")
    print("(samples, sequence_length, features)")

    print("\nSequence Length:")
    print(X_seq.shape[1])

    print("\nNumber of Features:")
    print(X_seq.shape[2])

    # class distribution
    failure_count = np.sum(y_seq == 1)
    no_failure_count = np.sum(y_seq == 0)

    print("\n" + "=" * 70)
    print("SEQUENCE CLASS DISTRIBUTION")
    print("=" * 70)

    print("No Failure samples :", no_failure_count)
    print("Failure samples    :", failure_count)

    print(
        "Failure percentage  :",
        round((failure_count / len(y_seq)) * 100, 2),
        "%"
    )

    # first sequence
    print("\n" + "=" * 70)
    print("FIRST SEQUENCE")
    print("=" * 70)

    print(X_seq[0])

    print("\nFirst Sequence Target:")
    print(y_seq[0])


if __name__ == "__main__":
    main()