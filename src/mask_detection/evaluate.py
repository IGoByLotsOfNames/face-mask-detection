from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import load_metadata, resolve_model_path, validate_model
from .data import dataset, digest, manifest_fingerprint, validate_manifest
from .metrics import evaluation_report


def evaluate_models(model_paths, root, *, batch_size=32, threshold=None):
    import numpy as np
    import tensorflow as tf

    root = Path(root)
    manifest = validate_manifest(root)
    num_classes = len(manifest["class_names"])
    categorical = num_classes >= 3
    if categorical and threshold is not None:
        raise ValueError("Categorical predictions use argmax; a threshold is not applicable")
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    if not rows:
        raise ValueError("Held-out test manifest is empty")
    predictions, identities = {}, {}
    for index, path in enumerate(model_paths):
        path = Path(path)
        model_path = resolve_model_path(path)
        metadata = load_metadata(path)
        if metadata["class_names"] != manifest["class_names"]:
            raise ValueError("Model class order does not match the dataset")
        if metadata["source_fingerprint"] != manifest["source_fingerprint"]:
            raise ValueError("Model source fingerprint does not match the dataset")
        if set(metadata["trained_split_hashes"]) & {r["sha256"] for r in rows}:
            raise ValueError("Test sample overlaps model training/validation history")
        if metadata["split_manifest_fingerprint"] != manifest_fingerprint(manifest):
            raise ValueError(
                "Use the model's recorded dataset manifest; external validation needs a separate protocol"
            )
        model = tf.keras.models.load_model(model_path, compile=False)
        validate_model(model, metadata)
        ds = dataset(
            root,
            manifest,
            "test",
            tuple(metadata["image_size"]),
            batch_size,
            categorical=categorical,
        )
        # Direct batches avoid implicit dataset cardinality/retrace behaviour.
        batches = []
        for x, _ in ds:
            output = np.asarray(model(x, training=False))
            width = num_classes if categorical else 1
            if output.ndim != 2 or output.shape != (len(x), width):
                raise ValueError("Model output batch shape does not match its class contract")
            batches.append(output if categorical else output.reshape(-1))
        if not batches:
            raise ValueError("Held-out dataset produced no batches")
        probability = np.concatenate(batches, axis=0)
        name = f"model_{index + 1}"
        predictions[name] = probability
        identities[name] = dict(
            file=model_path.name, sha256=digest(model_path), architecture=metadata["architecture"]
        )
    report = evaluation_report(rows, predictions, manifest["class_names"], threshold=threshold)
    report.update(
        models=identities,
        source_fingerprint=manifest["source_fingerprint"],
        manifest_sha256=digest(root / "manifest.json"),
        split="test",
    )
    return report


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite an earlier experiment report.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model on its validated held-out manifest"
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary only (default 0.5); preselect on validation data, never tune on test",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    report = evaluate_models(
        [args.model], args.data, batch_size=args.batch_size, threshold=args.threshold
    )
    write_report(report, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
