import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout


def build_gru_model(input_shape):

    model = Sequential([
        tf.keras.Input(shape=input_shape),

        GRU(
            32,
            activation="tanh"
        ),

        Dropout(0.2),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model