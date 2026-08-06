import pandas as pd


def preprocess_data(df):
    columns_to_drop = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ] # dropping the unwanted columns
    df = df.drop(columns=columns_to_drop)
    df = pd.get_dummies(
        df,
        columns=["Type"],
        dtype=int
    ) #encoding the categorical feature using One-Hot Encoding
    # seperate features and target
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    return X, y
