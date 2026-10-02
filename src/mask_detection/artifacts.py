"""Versioned model contract and hash checks shared by training and evaluation."""
from __future__ import annotations
import json
from pathlib import Path
from .data import digest, manifest_fingerprint

PREPROCESSING = "Pillow RGB, bilinear resize, float32 [0,255]; model owns normalization"


def metadata_path(model_path):
    return Path(model_path).with_suffix(".metadata.json")


def save_metadata(model_path, manifest, *, size, architecture, seed, extra=None):
    metadata = dict(schema_version=1, model_sha256=digest(model_path), class_names=manifest["class_names"],
                    image_size=list(size), preprocessing=PREPROCESSING, architecture=architecture, seed=seed,
                    source_fingerprint=manifest["source_fingerprint"],
                    split_manifest_fingerprint=manifest_fingerprint(manifest),
                    trained_split_hashes=sorted(r["sha256"] for r in manifest["rows"] if r["split"] in ("train", "validation")))
    if extra:
        metadata["training"] = extra
    metadata_path(model_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_metadata(model_path):
    metadata = json.loads(metadata_path(model_path).read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("preprocessing") != PREPROCESSING:
        raise ValueError("Unsupported model preprocessing contract")
    if len(metadata["class_names"]) != 2 or len(set(metadata["class_names"])) != 2:
        raise ValueError("Model must record exactly two ordered classes")
    if len(metadata["image_size"]) != 2 or any(type(v) is not int or v <= 0 for v in metadata["image_size"]):
        raise ValueError("Invalid model image dimensions")
    if digest(model_path) != metadata["model_sha256"]:
        raise ValueError("Model bytes do not match the metadata")
    return metadata


def validate_model(model, metadata):
    if tuple(model.input_shape) != (None, *metadata["image_size"], 3) or tuple(model.output_shape) != (None, 1):
        raise ValueError("Model input/output shape disagrees with its binary RGB contract")
