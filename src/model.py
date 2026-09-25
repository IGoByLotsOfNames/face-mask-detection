from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def build_model(input_shape: tuple[int, int, int] = (128, 128, 3)) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="mask_wearing_binary_classifier",
    )

