from src.data_loader import load_dataset


def main():

    # load dataset
    df = load_dataset()

    print("\n" + "=" * 70)
    print("SEQUENCE / TEMPORAL DATA ANALYSIS")
    print("=" * 70)

    # dataset information
    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # check first row
    print("\nFirst 10 Rows:")
    print(df.head(10).to_string())

    # check data types
    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes)

    # check possible ordering columns
    print("\n" + "=" * 70)
    print("POTENTIAL ORDERING / TIME FEATURES")
    print("=" * 70)

    possible_columns = [
        "UDI",
        "Product ID",
        "Air temperature [K]",
        "Process temperature [K]",
        "Tool wear [min]"
    ]

    for column in possible_columns:

        if column in df.columns:

            print(
                f"\n{column}:"
            )

            print(
                f"Unique values : {df[column].nunique()}"
            )

            print(
                f"First values  : "
                f"{df[column].head(10).tolist()}"
            )

    # check UDI ordering
    if "UDI" in df.columns:

        print("\n" + "=" * 70)
        print("UDI ORDERING CHECK")
        print("=" * 70)

        udi_diff = df["UDI"].diff().dropna()

        print(
            "UDI strictly increasing:",
            udi_diff.gt(0).all()
        )

        print(
            "Minimum UDI difference:",
            udi_diff.min()
        )

        print(
            "Maximum UDI difference:",
            udi_diff.max()
        )

    # check tool wear
    if "Tool wear [min]" in df.columns:

        print("\n" + "=" * 70)
        print("TOOL WEAR ANALYSIS")
        print("=" * 70)

        print(
            "Minimum:",
            df["Tool wear [min]"].min()
        )

        print(
            "Maximum:",
            df["Tool wear [min]"].max()
        )

        print(
            "Unique values:",
            df["Tool wear [min]"].nunique()
        )


if __name__ == "__main__":
    main()