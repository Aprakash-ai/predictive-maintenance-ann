import tensorflow as tf
from tensorflow.keras import layers, Model


def build_lstm_transformer_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # LSTM feature extraction
    x = layers.LSTM(
        32,
        return_sequences=True
    )(inputs)

    # Multi-Head Self Attention
    attention_output = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=8
    )(x, x)

    # Residual connection + Layer Normalization
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization()(x)

    # Transformer feed-forward network
    ff = layers.Dense(64, activation="relu")(x)
    ff = layers.Dropout(0.3)(ff)
    ff = layers.Dense(32)(ff)

    # Second residual connection + Layer Normalization
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    # Classification head
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)

    return model