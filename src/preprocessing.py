import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):

    columns_to_drop = [
        "UDI",
        "Product ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF"
    ]
    df = df.drop(columns=columns_to_drop) # unwanted columns are removed

    df = pd.get_dummies(
        df,
        columns=["Type"],
        dtype=int
    ) # categorical features are encoded using One-Hot Encoding

    # now separating the features and target
    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"]

    return X, y