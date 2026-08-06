from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input

def build_model(input_dim):

    model = Sequential(name="Predictive_Maintenance_ANN")

    # input + hidden layer
    model.add(Input(shape=(input_dim,)))

    model.add(
        Dense(
            units=32,
            activation="relu"
        )
    )

    model.add(Dropout(0.30))

    # hidden layer 2
    model.add(
        Dense(
            units=16,
            activation="relu"
        )
    )

    model.add(Dropout(0.20))

    # output layer
    model.add(
        Dense(
            units=1,
            activation="sigmoid"
        )
    )

    # compile model
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model