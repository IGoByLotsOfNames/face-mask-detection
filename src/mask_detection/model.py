"""Small from-scratch CNN with explicit binary/categorical output contracts."""

from __future__ import annotations


def validate_model_config(input_shape=(128, 128, 3), num_classes=2):
    """Reject impossible valid-convolution dimensions before importing TensorFlow."""
    if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 3:
        raise ValueError("input_shape must contain height, width and RGB channels")
    if (
        any(type(value) is not int for value in input_shape)
        or input_shape[2] != 3
        or min(input_shape[:2]) < 46
    ):
        raise ValueError("mask_cnn requires RGB images with height and width at least 46")
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two")


def build_model(input_shape=(128, 128, 3), num_classes=2):
    validate_model_config(input_shape, num_classes)
    import tensorflow as tf
    from tensorflow.keras import layers

    binary = num_classes == 2
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
            layers.Dense(
                1 if binary else num_classes, activation="sigmoid" if binary else "softmax"
            ),
        ],
        name="mask_wearing_binary_classifier" if binary else "mask_wearing_categorical_classifier",
    )
