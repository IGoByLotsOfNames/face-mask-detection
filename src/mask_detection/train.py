from __future__ import annotations
import argparse
import json
from pathlib import Path
from .artifacts import metadata_path, save_metadata
from .data import dataset, validate_manifest


def train_model(root, output, *, architecture="mask_cnn", epochs=30, batch_size=32,
                seed=42, pretrained=False, size=(128, 128)):
    import tensorflow as tf
    from .model import build_model
    root, output = Path(root), Path(output)
    if output.suffix != ".keras" or epochs <= 0 or batch_size <= 0:
        raise ValueError("Use a .keras output and positive epochs/batch size")
    history_path = output.with_suffix(".history.json")
    if any(p.exists() for p in (output, metadata_path(output), history_path)):
        raise FileExistsError("Choose a fresh model path")
    manifest = validate_manifest(root)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    train = dataset(root, manifest, "train", size, batch_size, shuffle=True, seed=seed)
    validation = dataset(root, manifest, "validation", size, batch_size)
    model = build_model((*size, 3))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    history = model.fit(train, validation_data=validation, epochs=epochs,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)], shuffle=False, verbose=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    history_path.write_text(json.dumps(history.history, indent=2), encoding="utf-8")
    save_metadata(output, manifest, size=size, architecture=architecture, seed=seed,
                  extra=dict(epochs_requested=epochs, epochs_completed=len(history.history["loss"]),
                             batch_size=batch_size, pretrained=pretrained, tensorflow=tf.__version__,
                             keras=tf.keras.__version__, deterministic_ops=True))
    return model


def main():
    parser = argparse.ArgumentParser(description="Train without using test labels for selection")
    parser.add_argument("data", type=Path)
    parser.add_argument("--architecture", choices=("mask_cnn",), default="mask_cnn")
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.keras"))
    args = parser.parse_args()
    train_model(args.data, args.output, architecture=args.architecture, epochs=args.epochs,
                batch_size=args.batch_size, seed=args.seed, pretrained=False)


if __name__ == "__main__":
    main()
