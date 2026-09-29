from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D
)


def build_cnn_multihead_attention_model(
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

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=8
    )(
        x,
        x
    )

    # Residual connection
    x = Add()(
        [x, attention_output]
    )

    # Normalization
    x = LayerNormalization()(
        x
    )

    # Feed-forward projection
    x = Dense(
        32,
        activation="relu"
    )(x)

    x = Dropout(
        0.3
    )(x)

    # Classification representation
    x = GlobalAveragePooling1D()(
        x
    )

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
    print("CNN + MULTI-HEAD ATTENTION MODEL")
    print("=" * 70)

    model = build_cnn_multihead_attention_model()

    model.summary()


if __name__ == "__main__":
    main()