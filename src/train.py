from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.utils.class_weight import compute_class_weight

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

def train_model(model, X_train, y_train):
    """
    import numpy as np
    # calculate class weights

    #class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))

    print("\n" + "=" * 70)
    print("CLASS WEIGHTS")
    print("=" * 70)
    print(class_weights)
    """

    history = model.fit(
        X_train,
        y_train,
        epochs = 30,
        batch_size = 32,
        validation_split = 0.2,
        # class_weight = class_weights,  only when class weights are added for better recall
        verbose = 1,
    )
    return history