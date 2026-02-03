"""Model architecture for image captioning."""
from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers

IMAGE_SIZE: Tuple[int, int] = (224, 224)


def build_encoder(trainable: bool = False) -> tf.keras.Model:
    """Build a CNN encoder using ResNet50."""
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = trainable

    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    return tf.keras.Model(inputs, x, name="encoder")


def build_decoder(vocab_size: int, max_length: int) -> tf.keras.Model:
    """Build an LSTM decoder."""
    caption_inputs = layers.Input(shape=(max_length - 1,), name="caption_input")
    image_features = layers.Input(shape=(256,), name="image_features")

    embeddings = layers.Embedding(vocab_size, 256, mask_zero=True)(caption_inputs)
    embeddings = layers.Dropout(0.3)(embeddings)

    state_h = layers.Dense(256, activation="tanh")(image_features)
    state_c = layers.Dense(256, activation="tanh")(image_features)

    lstm_out = layers.LSTM(256, return_sequences=True)(
        embeddings, initial_state=[state_h, state_c]
    )
    outputs = layers.Dense(vocab_size)(lstm_out)

    return tf.keras.Model([caption_inputs, image_features], outputs, name="decoder")


def build_caption_model(vocab_size: int, max_length: int) -> tf.keras.Model:
    """Build the full image captioning model."""
    image_inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    caption_inputs = layers.Input(shape=(max_length - 1,), name="caption_input")

    encoder = build_encoder()
    decoder = build_decoder(vocab_size, max_length)

    image_features = encoder(image_inputs)
    outputs = decoder([caption_inputs, image_features])

    return tf.keras.Model([image_inputs, caption_inputs], outputs, name="caption_model")