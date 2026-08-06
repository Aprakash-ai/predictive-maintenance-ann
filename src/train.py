from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    ) #split in train and test data

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):

    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]"
    ] # scaling the continuous numerical features

    scaler = StandardScaler()

    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    ) # learning from training data

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    ) # applying  scaling to test data

    return X_train, X_test, scaler