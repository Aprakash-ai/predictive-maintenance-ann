from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D
)


def build_attention_model():

    inputs = Input(shape=(10, 8))

    # Temporal representation
    lstm_output = LSTM(
        32,
        return_sequences=True
    )(inputs)

    # Self-attention
    attention_output = Attention()(
        [lstm_output, lstm_output]
    )

    # Convert sequence representation to fixed-size vector
    pooled_output = GlobalAveragePooling1D()(
        attention_output
    )

    # Classification layers
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