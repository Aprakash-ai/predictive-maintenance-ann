import pandas as pd
from pathlib import Path


def load_dataset():
    dataset_path = (
        Path(__file__).resolve().parent.parent
        / "dataset"
        / "ai4i2020.csv"
    )

    df = pd.read_csv(dataset_path)
    return df

def display_dataset_overview(df):

    print("=" * 70)
    print("AI4I 2020 PREDICTIVE MAINTENANCE DATASET OVERVIEW")
    print("=" * 70)

    print(f"\nDataset Shape: {df.shape}")

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")

    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nStatistical Summary:")
    print(df.describe())


def main():
    df = load_dataset()
    display_dataset_overview(df)


if __name__ == "__main__":
    main()