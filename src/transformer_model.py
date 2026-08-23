import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)


def build_transformer_model():

    inputs = Input(shape=(10, 8))

    # ==========================================================
    # Transformer Encoder
    # ==========================================================

    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=2,
        key_dim=16
    )(
        inputs,
        inputs
    )

    # Residual connection + normalization
    attention_output = LayerNormalization()(
        inputs + attention_output
    )

    # Feed-forward network
    feed_forward = Dense(
        32,
        activation="relu"
    )(attention_output)

    feed_forward = Dropout(0.2)(
        feed_forward
    )

    feed_forward = Dense(
        8
    )(feed_forward)

    # Residual connection + normalization
    encoder_output = LayerNormalization()(
        attention_output + feed_forward
    )

    # ==========================================================
    # Classification
    # ==========================================================

    pooled_output = GlobalAveragePooling1D()(
        encoder_output
    )

    dense_output = Dense(
        16,
        activation="relu"
    )(pooled_output)

    dropout_output = Dropout(0.3)(
        dense_output
    )

    outputs = Dense(
        1,
        activation="sigmoid"
    )(dropout_output)

    # ==========================================================
    # Build model
    # ==========================================================

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