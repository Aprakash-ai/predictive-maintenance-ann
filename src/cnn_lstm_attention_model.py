from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)


def build_cnn_lstm_attention_model(
    sequence_length=10,
    n_features=8
):

    inputs = Input(
        shape=(sequence_length, n_features)
    )

    # CNN feature extraction
    x = Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu"
    )(inputs)

    x = MaxPooling1D(
        pool_size=2
    )(x)

    # LSTM sequence representation
    x = LSTM(
        32,
        return_sequences=True
    )(x)

    # Self-attention
    attention_output = Attention()(
        [x, x]
    )

    # Combine LSTM and attention representation
    x = attention_output

    # Global representation
    x = GlobalAveragePooling1D()(x)

    x = Dense(
        16,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():

    print("\n" + "=" * 70)
    print("CNN + LSTM + ATTENTION MODEL")
    print("=" * 70)

    model = build_cnn_lstm_attention_model()

    model.summary()


if __name__ == "__main__":
    main()