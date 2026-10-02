"""Manifest-based binary image splitting. No TensorFlow import is needed here."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if not relative or "\\" in relative or p.is_absolute() or any(x in ("..", ".", "") for x in relative.split("/")):
        raise ValueError(f"Unsafe relative path: {relative!r}")
    result = root.joinpath(*p.parts)
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes the dataset")
    if any(parent.is_symlink() for parent in [result, *result.parents]):
        raise ValueError("Symlinks are not accepted in datasets")
    return result


def read_groups(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "group"]:
            raise ValueError("Group CSV must have exactly path,group columns")
        result = {}
        for row in reader:
            key, group = row["path"], row["group"].strip()
            if not group or key in result:
                raise ValueError("Group IDs must be nonempty and paths unique")
            result[key] = group
        return result


def manifest_fingerprint(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fingerprint(rows: list[dict]) -> str:
    content = [{key: r[key] for key in ("source", "sha256", "group", "label")} for r in rows]
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def create_split(source: Path, destination: Path, *, groups: dict[str, str] | None = None,
                 independent_images: bool = False, train: float = .70,
                 validation: float = .15, seed: int = 42) -> dict:
    """Copy into new output, keeping connected groups/byte duplicates together.

    Ratios are approximate because groups are indivisible. Each split must contain
    both classes. Group IDs identify patients/cases/original source images, not
    augmented filenames. Perceptual duplicates cannot be inferred automatically.
    """
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError("Destination already exists; choose a fresh path (never merge splits)")
    if destination.resolve().is_relative_to(source) or source.is_relative_to(destination.resolve()):
        raise ValueError("Source and destination must not contain each other")
    ratios = (train, validation, 1 - train - validation)
    if any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError("Train, validation and test proportions must all be positive")
    if (groups is None) == (not independent_images):
        raise ValueError("Provide group IDs OR explicitly assert independent images")
    classes = sorted(p.name for p in source.iterdir() if p.is_dir())
    if len(classes) != 2:
        raise ValueError("Exactly two nonempty class directories are required")
    rows = []
    for label, name in enumerate(classes):
        for path in sorted((source / name).rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not accepted")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                relative = path.relative_to(source).as_posix()
                safe_path(source, relative)
                group = relative if independent_images else groups.get(relative)
                if not group:
                    raise ValueError(f"Missing group for {relative}")
                rows.append(dict(source=relative, group=group, label=label,
                                 sha256=digest(path), size=path.stat().st_size))
    if not rows or set(r["label"] for r in rows) != {0, 1}:
        raise ValueError("Both classes need supported images")
    if groups is not None and set(groups) != {r["source"] for r in rows}:
        raise ValueError("Group CSV must cover exactly the supported image paths")
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        parent[find(i)] = find(j)
    seen_group, seen_hash = {}, {}
    for i, row in enumerate(rows):
        if row["group"] in seen_group:
            union(i, seen_group[row["group"]])
        seen_group[row["group"]] = i
        if row["sha256"] in seen_hash:
            j = seen_hash[row["sha256"]]
            if rows[j]["label"] != row["label"]:
                raise ValueError("Identical image bytes have conflicting class labels")
            union(i, j)
        seen_hash[row["sha256"]] = i
    components = {}
    for i, row in enumerate(rows):
        components.setdefault(find(i), []).append(row)
    units = list(components.values())
    if any(sum(any(r["label"] == label for r in unit) for unit in units) < 3 for label in (0, 1)):
        raise ValueError("Each class needs at least three independent group/duplicate components")
    random.Random(seed).shuffle(units)
    units.sort(key=len, reverse=True)
    totals = Counter(r["label"] for r in rows)
    counts = [[0, 0] for _ in SPLITS]
    targets = [[totals[label] * ratio for label in (0, 1)] for ratio in ratios]
    for unit in units:
        added = Counter(r["label"] for r in unit)
        def cost(index):
            coverage = sum(counts[index][label] == 0 for label in added)
            change = sum(((counts[index][label] + added[label] - targets[index][label]) ** 2
                          - (counts[index][label] - targets[index][label]) ** 2)
                         / targets[index][label] for label in (0, 1))
            return (-coverage, change, index)
        chosen = min(range(3), key=cost)
        for row in unit:
            row["split"] = SPLITS[chosen]
            row["path"] = f"{row['split']}/{row['source']}"
            counts[chosen][row["label"]] += 1
    if any(0 in count for count in counts):
        raise ValueError("Cannot produce class-complete partitions; review grouping or add independent groups")
    manifest = dict(schema_version=1, seed=seed, class_names=classes,
                    requested_ratios=dict(zip(SPLITS, ratios)),
                    grouping="explicit" if groups is not None else "asserted_independent_images",
                    duplicate_policy="exact_bytes_connected_with_groups", rows=rows,
                    source_fingerprint=_fingerprint(rows))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".split-", dir=destination.parent))
    try:
        for row in rows:
            target = staging / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / row["source"], target)
            if digest(target) != row["sha256"]:
                raise ValueError("Source changed during the copy")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        validate_manifest(staging)
        if destination.exists():
            raise FileExistsError("Destination appeared during splitting")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def validate_manifest(root: Path) -> dict:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    classes, rows = manifest["class_names"], manifest["rows"]
    if manifest.get("schema_version") != 1 or len(classes) != 2 or len(set(classes)) != 2 or not rows:
        raise ValueError("Invalid binary dataset manifest")
    paths, sources, groups, hashes, hash_labels, coverage = set(), set(), {}, {}, {}, set()
    for row in rows:
        label, split = row["label"], row["split"]
        if type(label) is not int or label not in (0, 1) or split not in SPLITS:
            raise ValueError("Invalid class index or split")
        if not isinstance(row["group"], str) or not row["group"]:
            raise ValueError("Invalid group ID")
        source_parts = PurePosixPath(row["source"]).parts
        if not source_parts or source_parts[0] != classes[label] or row["path"] != f"{split}/{row['source']}":
            raise ValueError("Class/path mapping does not match the manifest")
        if row["path"] in paths or row["source"] in sources:
            raise ValueError("Repeated sample identity")
        paths.add(row["path"]); sources.add(row["source"]); coverage.add((split, label))
        for key, registry in ((row["group"], groups), (row["sha256"], hashes)):
            if key in registry and registry[key] != split:
                raise ValueError("Group or exact duplicate crosses a split boundary")
            registry[key] = split
        if row["sha256"] in hash_labels and hash_labels[row["sha256"]] != label:
            raise ValueError("Identical images have conflicting labels")
        hash_labels[row["sha256"]] = label
        path = safe_path(root, row["path"])
        if not path.is_file() or path.stat().st_size != row["size"] or digest(path) != row["sha256"]:
            raise ValueError(f"Missing or changed sample: {row['path']}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
    if actual != paths or coverage != {(s, label) for s in SPLITS for label in (0, 1)}:
        raise ValueError("Untracked files or incomplete class coverage")
    if manifest["source_fingerprint"] != _fingerprint(rows):
        raise ValueError("Source fingerprint mismatch")
    return manifest


def load_rgb(path: Path, size: tuple[int, int]):
    """RGB float32 [0,255], bilinear resize; normalization lives in the model."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def dataset(root: Path, manifest: dict, split: str, size: tuple[int, int], batch_size: int,
            *, shuffle: bool = False, seed: int = 42):
    import numpy as np
    import tensorflow as tf
    if batch_size <= 0 or split not in SPLITS:
        raise ValueError("Invalid batch size or split")
    rows = [r for r in manifest["rows"] if r["split"] == split]
    def generate():
        for row in rows:
            yield load_rgb(root / row["path"], size), np.asarray([row["label"]], dtype=np.float32)
    ds = tf.data.Dataset.from_generator(generate, output_signature=(
        tf.TensorSpec((*size, 3), tf.float32), tf.TensorSpec((1,), tf.float32)))
    if shuffle:
        ds = ds.shuffle(len(rows), seed=seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    ds = ds.batch(batch_size).apply(tf.data.experimental.assert_cardinality(math.ceil(len(rows) / batch_size)))
    return ds.with_options(options).prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path); parser.add_argument("destination", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--groups", type=Path); group.add_argument("--independent-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=.70)
    parser.add_argument("--validation", type=float, default=.15)
    args = parser.parse_args()
    manifest = create_split(args.source, args.destination, groups=read_groups(args.groups) if args.groups else None,
                            independent_images=args.independent_images, seed=args.seed, train=args.train, validation=args.validation)
    print(json.dumps({"source_fingerprint": manifest["source_fingerprint"],
                      "counts": dict(Counter(r["split"] for r in manifest["rows"]))}, indent=2))


if __name__ == "__main__":
    main()
