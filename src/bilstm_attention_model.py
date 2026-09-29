from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)


def build_bilstm_attention_model(
    sequence_length=10,
    n_features=8
):

    inputs = Input(
        shape=(sequence_length, n_features)
    )

    # Bidirectional LSTM
    x = Bidirectional(
        LSTM(
            32,
            return_sequences=True
        )
    )(inputs)

    # Self-attention
    attention_output = Attention()(
        [x, x]
    )

    # Global sequence representation
    x = GlobalAveragePooling1D()(
        attention_output
    )

    # Fully connected layers
    x = Dense(
        16,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    # Binary classification output
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
    print("BiLSTM + ATTENTION MODEL")
    print("=" * 70)

    model = build_bilstm_attention_model()

    model.summary()


if __name__ == "__main__":
    main()