from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a correct-versus-incorrect mask-wearing classifier")
    parser.add_argument("data", type=Path, help="Directory with one child directory per class")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/mask-classifier.keras"))
    args = parser.parse_args()

    train = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(128, 128),
        batch_size=32,
        label_mode="binary",
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        args.data,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(128, 128),
        batch_size=32,
        label_mode="binary",
    )
    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.fit(
        train,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
    )
    model.save(args.output)


if __name__ == "__main__":
    main()

