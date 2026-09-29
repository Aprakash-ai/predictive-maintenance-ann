from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout
)


def build_cnn_lstm_model(
        sequence_length=10,
        n_features=8
):
    model = Sequential([

        Conv1D(
            filters=32,
            kernel_size=3,
            activation="relu",
            input_shape=(sequence_length, n_features)
        ),

        MaxPooling1D(
            pool_size=2
        ),

        LSTM(
            32
        ),

        Dropout(
            0.3
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dropout(
            0.3
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


def main():
    print("\n" + "=" * 70)
    print("CNN + LSTM MODEL")
    print("=" * 70)

    model = build_cnn_lstm_model()

    model.summary()


if __name__ == "__main__":
    main()