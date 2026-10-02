from __future__ import annotations
import argparse
import json
from pathlib import Path

from .artifacts import load_metadata, validate_model
from .data import dataset, digest, validate_manifest, manifest_fingerprint
from .metrics import evaluation_report


def evaluate_models(model_paths, root, *, batch_size=32, threshold=.5):
    import numpy as np
    import tensorflow as tf
    root = Path(root)
    manifest = validate_manifest(root)
    rows = [r for r in manifest["rows"] if r["split"] == "test"]
    predictions, identities = {}, {}
    for index, path in enumerate(model_paths):
        path = Path(path)
        metadata = load_metadata(path)
        if metadata["class_names"] != manifest["class_names"]:
            raise ValueError("Model class order does not match the dataset")
        if set(metadata["trained_split_hashes"]) & {r["sha256"] for r in rows}:
            raise ValueError("Test sample overlaps model training/validation history")
        if metadata["split_manifest_fingerprint"] != manifest_fingerprint(manifest):
            raise ValueError("Use the model's recorded dataset manifest; external validation needs a separate protocol")
        model = tf.keras.models.load_model(path, compile=False)
        validate_model(model, metadata)
        ds = dataset(root, manifest, "test", tuple(metadata["image_size"]), batch_size)
        # Direct batches avoid implicit dataset cardinality/retrace behaviour.
        probability = np.concatenate([np.asarray(model(x, training=False)).reshape(-1) for x, _ in ds])
        name = f"model_{index + 1}"
        predictions[name] = probability
        identities[name] = dict(file=path.name, sha256=digest(path), architecture=metadata["architecture"])
    report = evaluation_report(rows, predictions, manifest["class_names"], threshold=threshold)
    report.update(models=identities, source_fingerprint=manifest["source_fingerprint"],
                  manifest_sha256=digest(root / "manifest.json"), split="test")
    return report


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite an earlier experiment report.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on its validated held-out manifest")
    parser.add_argument("model", type=Path); parser.add_argument("data", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=.5, help="Preselect on validation data, never tune on test")
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation.json"))
    args = parser.parse_args()
    report = evaluate_models([args.model], args.data, batch_size=args.batch_size, threshold=args.threshold)
    write_report(report, args.output)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
