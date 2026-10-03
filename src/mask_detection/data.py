"""Manifest-based grouped image splitting. No TensorFlow import is needed here."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")
COVERAGE_SEARCH_MAX_STATES = 100_000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or p.is_absolute()
        or any(x in ("..", ".", "") for x in relative.split("/"))
    ):
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
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fingerprint(rows: list[dict]) -> str:
    content = [{key: r[key] for key in ("source", "sha256", "group", "label")} for r in rows]
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_class_names(classes):
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or any(not isinstance(name, str) or not name.strip() for name in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Need at least two distinct, nonempty class names")
    return len(classes)


def inspect_image(path: Path) -> dict:
    """Decode before accepting a sample; hashing alone does not validate an image."""
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format not in ("PNG", "JPEG") or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("Need a single-frame PNG or JPEG")
                image.verify()
            with Image.open(path) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise ValueError("Image dimensions must be positive")
                return {"width": image.width, "height": image.height}
    except (
        OSError,
        ValueError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ) as exc:
        raise ValueError(f"Invalid image {path.name}: {exc}") from exc


class CoverageSearchExhausted(ValueError):
    """The coverage search stopped without proving feasibility or infeasibility."""


def _coverage_seeds(units: list[list[dict]], class_count: int) -> tuple[dict[int, int], int]:
    """Find whole components that cover every class in each of three splits.

    Components with the same class-presence mask are interchangeable for coverage.
    At most one such component is needed per split, so keeping three representatives
    of each mask preserves feasibility. Smaller components are preferred as seeds;
    unused components remain available to the ratio-based assignment afterwards.
    The iterative search works beyond three classes without a recursion-depth cap.
    """
    by_mask = {}
    for index, unit in enumerate(units):
        mask = sum(1 << label for label in {row["label"] for row in unit})
        by_mask.setdefault(mask, []).append(index)
    masks = sorted(by_mask)
    pools = [
        sorted(by_mask[mask], key=lambda i: (len(units[i]), i))[: len(SPLITS)] for mask in masks
    ]
    initial = tuple(len(pool) for pool in pools)
    complete = (1 << class_count) - 1
    # State identity excludes historical assignments: remaining components with
    # equal masks have the same future coverage possibilities.
    stack = [((0,) * len(SPLITS), initial, {})]
    visited = set()
    while stack:
        coverage, remaining, selected = stack.pop()
        state = (coverage, remaining)
        if state in visited:
            continue
        if len(visited) >= COVERAGE_SEARCH_MAX_STATES:
            raise CoverageSearchExhausted(
                f"Coverage search budget exhausted after {COVERAGE_SEARCH_MAX_STATES} states; "
                "feasibility remains undetermined. Review groups or use another split strategy."
            )
        visited.add(state)
        if all(mask == complete for mask in coverage):
            return selected, len(visited)
        requirements = []
        impossible = False
        for label in range(class_count):
            bit = 1 << label
            missing = [split for split, present in enumerate(coverage) if not present & bit]
            if not missing:
                continue
            candidates = [i for i, mask in enumerate(masks) if remaining[i] and mask & bit]
            if sum(remaining[i] for i in candidates) < len(missing):
                impossible = True
                break
            for split in missing:
                requirements.append((len(candidates), split, label, candidates))
        if impossible:
            continue
        # Resolve the most constrained missing class first, with stable tie breaks.
        _, split, _, candidates = min(requirements, key=lambda item: item[:3])
        candidates.sort(
            key=lambda i: (
                -(masks[i] & ~coverage[split]).bit_count(),
                len(units[pools[i][initial[i] - remaining[i]]]),
                i,
            )
        )
        for i in reversed(candidates):
            next_remaining = list(remaining)
            next_remaining[i] -= 1
            next_coverage = list(coverage)
            next_coverage[split] |= masks[i]
            next_selected = dict(selected)
            next_selected[pools[i][initial[i] - remaining[i]]] = split
            stack.append((tuple(next_coverage), tuple(next_remaining), next_selected))
    raise ValueError(
        "No class-complete three-way partition exists for the declared group/duplicate components"
    )


def create_split(
    source: Path,
    destination: Path,
    *,
    groups: dict[str, str] | None = None,
    independent_images: bool = False,
    train: float = 0.70,
    validation: float = 0.15,
    seed: int = 42,
) -> dict:
    """Copy into new output, keeping connected groups/byte duplicates together.

    Ratios are approximate because groups are indivisible. Each split must contain
    every class. Group IDs identify patients/cases/original source images, not
    augmented filenames. Perceptual duplicates cannot be inferred automatically.
    """
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists():
        raise FileExistsError(
            "Destination already exists; choose a fresh path (never merge splits)"
        )
    if destination.resolve().is_relative_to(source) or source.is_relative_to(destination.resolve()):
        raise ValueError("Source and destination must not contain each other")
    ratios = (train, validation, 1 - train - validation)
    if any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise ValueError("Train, validation and test proportions must all be positive")
    if (groups is None) == (not independent_images):
        raise ValueError("Provide group IDs OR explicitly assert independent images")
    classes = sorted(p.name for p in source.iterdir() if p.is_dir())
    class_count = validate_class_names(classes)
    rows = []
    for label, name in enumerate(classes):
        for path in sorted((source / name).rglob("*")):
            if path.is_symlink():
                raise ValueError("Symlinks are not accepted")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                relative = path.relative_to(source).as_posix()
                safe_path(source, relative)
                dimensions = inspect_image(path)
                group = relative if independent_images else groups.get(relative)
                if not group:
                    raise ValueError(f"Missing group for {relative}")
                rows.append(
                    dict(
                        source=relative,
                        group=group,
                        label=label,
                        sha256=digest(path),
                        size=path.stat().st_size,
                        image_dimensions=dimensions,
                    )
                )
    if not rows or set(r["label"] for r in rows) != set(range(class_count)):
        raise ValueError("Every class needs supported images")
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
    if any(
        sum(any(r["label"] == label for r in unit) for unit in units) < 3
        for label in range(class_count)
    ):
        raise ValueError("Each class needs at least three independent group/duplicate components")
    random.Random(seed).shuffle(units)
    units.sort(key=len, reverse=True)
    totals = Counter(r["label"] for r in rows)
    counts = [[0] * class_count for _ in SPLITS]
    targets = [[totals[label] * ratio for label in range(class_count)] for ratio in ratios]

    def choose(unit):
        added = Counter(r["label"] for r in unit)

        def cost(index):
            coverage = sum(counts[index][label] == 0 for label in added)
            change = sum(
                (
                    (counts[index][label] + added[label] - targets[index][label]) ** 2
                    - (counts[index][label] - targets[index][label]) ** 2
                )
                / targets[index][label]
                for label in range(class_count)
            )
            return (-coverage, change, index)

        return min(range(len(SPLITS)), key=cost)

    def assign(unit, chosen):
        for row in unit:
            row["split"] = SPLITS[chosen]
            row["path"] = f"{row['split']}/{row['source']}"
            counts[chosen][row["label"]] += 1

    for unit in units:
        assign(unit, choose(unit))
    coverage_repair = None
    if any(0 in count for count in counts):
        seeds, explored = _coverage_seeds(units, class_count)
        counts = [[0] * class_count for _ in SPLITS]
        for index, chosen in sorted(seeds.items()):
            assign(units[index], chosen)
        for index, unit in enumerate(units):
            if index not in seeds:
                assign(unit, choose(unit))
        coverage_repair = dict(
            method="deterministic_class_mask_search_then_greedy",
            states_explored=explored,
            maximum_states=COVERAGE_SEARCH_MAX_STATES,
        )
        if any(0 in count for count in counts):
            raise RuntimeError("Coverage repair failed its class-completeness invariant")
    manifest = dict(
        schema_version=1 if class_count == 2 else 2,
        seed=seed,
        class_names=classes,
        requested_ratios=dict(zip(SPLITS, ratios)),
        grouping="explicit" if groups is not None else "asserted_independent_images",
        duplicate_policy="exact_bytes_connected_with_groups",
        rows=rows,
        source_fingerprint=_fingerprint(rows),
    )
    if coverage_repair is not None:
        manifest["coverage_repair"] = coverage_repair
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
    class_count = validate_class_names(classes)
    if (
        manifest.get("schema_version") not in (1, 2)
        or (manifest["schema_version"] == 1 and class_count != 2)
        or not rows
    ):
        raise ValueError("Invalid dataset manifest schema or empty rows")
    paths, sources, groups, hashes, hash_labels, coverage = set(), set(), {}, {}, {}, set()
    for row in rows:
        label, split = row["label"], row["split"]
        if type(label) is not int or label not in range(class_count) or split not in SPLITS:
            raise ValueError("Invalid class index or split")
        if not isinstance(row["group"], str) or not row["group"]:
            raise ValueError("Invalid group ID")
        source_parts = PurePosixPath(row["source"]).parts
        if (
            not source_parts
            or source_parts[0] != classes[label]
            or row["path"] != f"{split}/{row['source']}"
        ):
            raise ValueError("Class/path mapping does not match the manifest")
        if row["path"] in paths or row["source"] in sources:
            raise ValueError("Repeated sample identity")
        paths.add(row["path"])
        sources.add(row["source"])
        coverage.add((split, label))
        for key, registry in ((row["group"], groups), (row["sha256"], hashes)):
            if key in registry and registry[key] != split:
                raise ValueError("Group or exact duplicate crosses a split boundary")
            registry[key] = split
        if row["sha256"] in hash_labels and hash_labels[row["sha256"]] != label:
            raise ValueError("Identical images have conflicting labels")
        hash_labels[row["sha256"]] = label
        path = safe_path(root, row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != row["size"]
            or digest(path) != row["sha256"]
        ):
            raise ValueError(f"Missing or changed sample: {row['path']}")
        dimensions = inspect_image(path)
        if "image_dimensions" in row and row["image_dimensions"] != dimensions:
            raise ValueError(f"Image dimension metadata mismatch: {row['path']}")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    if actual != paths or coverage != {(s, label) for s in SPLITS for label in range(class_count)}:
        raise ValueError("Untracked files or incomplete class coverage")
    if manifest["source_fingerprint"] != _fingerprint(rows):
        raise ValueError("Source fingerprint mismatch")
    return manifest


def load_rgb(path: Path, size: tuple[int, int]):
    """RGB float32 [0,255], bilinear resize; normalization lives in the model."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )


def dataset(
    root: Path,
    manifest: dict,
    split: str,
    size: tuple[int, int],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 42,
    categorical: bool = False,
):
    import numpy as np
    import tensorflow as tf

    if (
        type(batch_size) is not int
        or batch_size <= 0
        or split not in SPLITS
        or len(size) != 2
        or any(type(v) is not int or v <= 0 for v in size)
    ):
        raise ValueError("Invalid batch size or split")
    rows = [r for r in manifest["rows"] if r["split"] == split]
    if not rows:
        raise ValueError("No samples in the requested split")
    if not categorical and len(manifest["class_names"]) != 2:
        raise ValueError("Multiclass datasets require categorical=True")
    # Shuffle small indices, not the decoded float32 image tensors.
    ds = tf.data.Dataset.from_tensor_slices(np.arange(len(rows), dtype=np.int64))
    if shuffle:
        ds = ds.shuffle(len(rows), seed=seed)

    def load_index(index):
        row = rows[int(index)]
        label = (
            np.int32(row["label"]) if categorical else np.asarray([row["label"]], dtype=np.float32)
        )
        return load_rgb(Path(root) / row["path"], size), label

    def decode(index):
        pixels, label = tf.numpy_function(
            load_index, [index], [tf.float32, tf.int32 if categorical else tf.float32]
        )
        pixels.set_shape((*size, 3))
        label.set_shape(()) if categorical else label.set_shape((1,))
        return pixels, label

    ds = ds.map(decode, num_parallel_calls=1, deterministic=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    ds = ds.batch(batch_size).apply(
        tf.data.experimental.assert_cardinality(math.ceil(len(rows) / batch_size))
    )
    return ds.with_options(options).prefetch(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--groups", type=Path)
    group.add_argument("--independent-images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--validation", type=float, default=0.15)
    args = parser.parse_args()
    manifest = create_split(
        args.source,
        args.destination,
        groups=read_groups(args.groups) if args.groups else None,
        independent_images=args.independent_images,
        seed=args.seed,
        train=args.train,
        validation=args.validation,
    )
    print(
        json.dumps(
            {
                "source_fingerprint": manifest["source_fingerprint"],
                "counts": dict(Counter(r["split"] for r in manifest["rows"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
