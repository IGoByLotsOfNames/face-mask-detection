"""Train on declared training/validation splits and publish a complete model bundle."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path

from .artifacts import (
    load_metadata,
    metadata_path,
    resolve_model_path,
    save_metadata,
    validate_classes,
    validate_model,
    write_bundle_manifest,
)
from .data import dataset, validate_manifest
from .model import validate_model_config


def _validate_options(*, architecture, epochs, batch_size, seed, pretrained, size):
    # Validation intentionally precedes TensorFlow import and all dataset work.
    if architecture != "mask_cnn" or pretrained is not False:
        raise ValueError("Only architecture='mask_cnn' with pretrained=False is supported")
    if type(epochs) is not int or epochs <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive integers")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError("size must contain height and width")
    validate_model_config((*size, 3))


def _exists(path):
    return path.exists() or path.is_symlink()


@contextmanager
def _reservation(destination, members):
    """Exclude concurrent cooperating trainers without overwriting any prior run.

    A killed process can leave its reservation/staging directory for inspection.
    This is not intended as a hostile-writer filesystem security boundary.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = destination.parent / f".{destination.name}.training.lock"
    # Only enter cleanup after this invocation has exclusively created the lock.
    # Writing or closing it can fail (for example, a full disk); both belong
    # inside the cleanup boundary just like training and publication failures.
    stream = lock.open("x", encoding="utf-8")
    try:
        with stream:
            stream.write(f"Training reservation; process {os.getpid()}\n")
        if any(_exists(member) for member in members):
            raise FileExistsError("Choose a fresh model destination")
        yield
    finally:
        lock.unlink()


def _fit_and_save(root, staged_model, *, architecture, epochs, batch_size, seed, pretrained, size):
    manifest = validate_manifest(root)
    num_classes = len(validate_classes(manifest["class_names"]))
    import tensorflow as tf

    from .model import build_model

    categorical = num_classes >= 3
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    train = dataset(
        root, manifest, "train", size, batch_size, shuffle=True, seed=seed, categorical=categorical
    )
    validation = dataset(root, manifest, "validation", size, batch_size, categorical=categorical)
    model = build_model((*size, 3), num_classes=num_classes)
    if categorical:
        loss, metrics = "sparse_categorical_crossentropy", ["accuracy"]
    else:
        loss, metrics = "binary_crossentropy", ["accuracy", tf.keras.metrics.AUC(name="auc")]
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss, metrics=metrics)
    history = model.fit(
        train,
        validation_data=validation,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        shuffle=False,
        verbose=2,
    )
    model.save(staged_model)
    staged_model.with_suffix(".history.json").write_text(
        json.dumps(history.history, indent=2, allow_nan=False), encoding="utf-8"
    )
    save_metadata(
        staged_model,
        manifest,
        size=size,
        architecture=architecture,
        seed=seed,
        extra=dict(
            epochs_requested=epochs,
            epochs_completed=len(history.history["loss"]),
            batch_size=batch_size,
            pretrained=pretrained,
            tensorflow=tf.__version__,
            keras=tf.keras.__version__,
            deterministic_ops=True,
        ),
    )
    metadata = load_metadata(staged_model)
    # Ensure the serialized file, not only the in-memory object, is a valid Keras model.
    restored = tf.keras.models.load_model(staged_model, compile=False)
    validate_model(restored, metadata)
    return model


def train_bundle(
    root,
    destination,
    *,
    architecture="mask_cnn",
    epochs=30,
    batch_size=32,
    seed=42,
    pretrained=False,
    size=(128, 128),
):
    """Publish a new directory only after its three artifacts are complete/verified.

    Staging and destination are siblings on the same filesystem. Directory rename
    provides whole-bundle visibility; this does not promise power-loss durability.
    """
    options = dict(
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        pretrained=pretrained,
        size=size,
    )
    _validate_options(**options)
    root, destination = Path(root), Path(destination)
    if not destination.name or destination.suffix == ".keras":
        raise ValueError("Bundle output must be a fresh directory, not a .keras file")
    with _reservation(destination, [destination]):
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.staging-", dir=destination.parent
        ) as temp:
            stage = Path(temp)
            model = _fit_and_save(root, stage / "model.keras", **options)
            write_bundle_manifest(stage)
            resolve_model_path(stage)
            load_metadata(stage)
            if _exists(destination):
                raise FileExistsError("Choose a fresh model destination")
            os.rename(stage, destination)
    return model


def train_model(
    root,
    output,
    *,
    architecture="mask_cnn",
    epochs=30,
    batch_size=32,
    seed=42,
    pretrained=False,
    size=(128, 128),
):
    """Deprecated three-file output; prefer train_bundle for atomic publication.

    Stages/validates all files first, publishes metadata last, and rolls back files
    created here on ordinary errors. Sudden termination can leave a partial trio;
    three independent filenames cannot provide whole-bundle atomic visibility.
    """
    options = dict(
        architecture=architecture,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        pretrained=pretrained,
        size=size,
    )
    _validate_options(**options)
    root, output = Path(root), Path(output)
    if output.suffix != ".keras":
        raise ValueError("Legacy output must end in .keras; use train_bundle for directory output")
    warnings.warn(
        "train_model's separate files are deprecated; use train_bundle for atomic publication",
        DeprecationWarning,
        stacklevel=2,
    )
    destinations = [output, output.with_suffix(".history.json"), metadata_path(output)]
    with _reservation(output, destinations):
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.staging-", dir=output.parent
        ) as temp:
            staged = Path(temp) / output.name
            model = _fit_and_save(root, staged, **options)
            created = []
            try:
                for source, destination in zip(
                    [staged, staged.with_suffix(".history.json"), metadata_path(staged)],
                    destinations,
                ):
                    # Same-filesystem hardlinks publish each complete file exclusively.
                    # A destination created by another writer is never overwritten.
                    os.link(source, destination)
                    created.append(destination)
            except BaseException:
                for path in reversed(created):
                    path.unlink()
                raise
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--architecture", choices=("mask_cnn",), default="mask_cnn")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/run"),
        help="Fresh bundle directory (default), or deprecated separate .keras files",
    )
    args = parser.parse_args()
    trainer = train_model if args.output.suffix == ".keras" else train_bundle
    if trainer is train_model:
        warnings.warn(
            ".keras output publishes separate legacy files; prefer a bundle directory",
            FutureWarning,
        )
    trainer(
        args.data,
        args.output,
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained=False,
    )


if __name__ == "__main__":
    main()
