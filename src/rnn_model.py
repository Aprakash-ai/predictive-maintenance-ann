import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


def build_rnn_model(input_shape):

    model = Sequential([
        SimpleRNN(
            32,
            activation="tanh",
            input_shape=input_shape
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