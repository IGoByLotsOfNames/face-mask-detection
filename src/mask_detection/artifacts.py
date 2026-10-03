"""Versioned model contracts and complete, hash-checked directory bundles.

Hashes detect accidental changes and bind the ordered class meanings to a model;
they are not signatures or protection against an adversary rewriting all files.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .data import digest, manifest_fingerprint
from .metrics import DECISION_POLICY

PREPROCESSING = "Pillow RGB, bilinear resize, float32 [0,255]; model owns normalization"
OUTPUT_MODE = "categorical_softmax"
BUNDLE_FILES = ("model.keras", "model.metadata.json", "model.history.json")


def validate_classes(classes):
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or any(not isinstance(name, str) or not name.strip() for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError(
            "Model must record a list of at least two distinct, nonempty ordered classes"
        )
    return classes


def metadata_path(model_path):
    return Path(model_path).with_suffix(".metadata.json")


def _semantics_hash(metadata):
    keys = (
        "model_sha256",
        "class_names",
        "image_size",
        "preprocessing",
        "architecture",
        "output_mode",
        "decision_policy",
    )
    payload = json.dumps(
        {key: metadata[key] for key in keys},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_metadata(model_path, manifest, *, size, architecture, seed, extra=None):
    classes = validate_classes(manifest["class_names"])
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("Invalid model image dimensions")
    metadata = dict(
        schema_version=1 if len(classes) == 2 else 2,
        model_sha256=digest(model_path),
        class_names=classes,
        image_size=list(size),
        preprocessing=PREPROCESSING,
        architecture=architecture,
        seed=seed,
        source_fingerprint=manifest["source_fingerprint"],
        split_manifest_fingerprint=manifest_fingerprint(manifest),
        trained_split_hashes=sorted(
            r["sha256"] for r in manifest["rows"] if r["split"] in ("train", "validation")
        ),
    )
    if len(classes) >= 3:
        metadata.update(output_mode=OUTPUT_MODE, decision_policy=dict(DECISION_POLICY))
        metadata["semantics_sha256"] = _semantics_hash(metadata)
    if extra:
        metadata["training"] = extra
    metadata_path(model_path).write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    return metadata


def _bundle_member(directory, name):
    member = directory / name
    if (
        member.is_symlink()
        or not member.is_file()
        or member.resolve().parent != directory.resolve()
    ):
        raise ValueError(f"Missing or unsafe bundle member: {name}")
    return member


def write_bundle_manifest(directory):
    """Seal a completed staging bundle; publication is the trainer's responsibility."""
    directory = Path(directory)
    manifest = {
        "schema_version": 1,
        "files": {name: digest(_bundle_member(directory, name)) for name in BUNDLE_FILES},
    }
    with (directory / "bundle.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
    return manifest


def resolve_model_path(path):
    """Accept a legacy file or validate every member of an immutable model bundle."""
    path = Path(path)
    if not path.is_dir():
        return path
    try:
        manifest = json.loads(_bundle_member(path, "bundle.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != set(BUNDLE_FILES)
        ):
            raise ValueError("Unsupported or incomplete model bundle manifest")
        if {member.name for member in path.iterdir()} != {*BUNDLE_FILES, "bundle.json"}:
            raise ValueError("Unexpected members in model bundle")
        for name, expected in manifest["files"].items():
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("Invalid bundle member hash")
            if digest(_bundle_member(path, name)) != expected:
                raise ValueError(f"Bundle member does not match its recorded hash: {name}")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Cannot read complete model bundle") from error
    return path / "model.keras"


def load_metadata(model_path):
    model_path = resolve_model_path(model_path)
    try:
        metadata = json.loads(metadata_path(model_path).read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema_version") not in (1, 2)
            or metadata.get("preprocessing") != PREPROCESSING
        ):
            raise ValueError("Unsupported model preprocessing contract")
        classes = validate_classes(metadata.get("class_names"))
        dimensions = metadata.get("image_size")
        if (
            not isinstance(dimensions, list)
            or len(dimensions) != 2
            or any(type(value) is not int or value <= 0 for value in dimensions)
        ):
            raise ValueError("Invalid model image dimensions")
        if metadata["schema_version"] == 1:
            if len(classes) != 2:
                raise ValueError("Legacy schema 1 requires exactly two ordered classes")
        elif (
            len(classes) < 3
            or metadata.get("output_mode") != OUTPUT_MODE
            or metadata.get("decision_policy") != DECISION_POLICY
            or metadata.get("semantics_sha256") != _semantics_hash(metadata)
        ):
            raise ValueError("Unsupported or altered categorical model contract")
        if digest(model_path) != metadata["model_sha256"]:
            raise ValueError("Model bytes do not match the metadata")
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete model metadata contract") from error
    return metadata


def validate_model(model, metadata):
    classes = validate_classes(metadata["class_names"])
    outputs = 1 if len(classes) == 2 else len(classes)
    if tuple(model.input_shape) != (None, *metadata["image_size"], 3) or tuple(
        model.output_shape
    ) != (None, outputs):
        raise ValueError("Model input/output shape disagrees with its ordered-class RGB contract")
