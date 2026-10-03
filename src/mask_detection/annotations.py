"""Prepare traceable face crops from explicitly interpreted PASCAL VOC boxes.

This module does not infer coordinate semantics, train models or publish images.
The input must contain sibling ``images`` and ``annotations`` directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

CLASS_NAMES = ("mask_weared_incorrect", "with_mask", "without_mask")
COORDINATE_CONVENTIONS = ("voc-1-based-inclusive", "zero-based-half-open")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
MAX_XML_BYTES = 2 * 1024 * 1024
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unlinked(path: Path) -> None:
    """Reject symbolic links and Windows junction/reparse components."""
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError(f"Links are not accepted: {component}")
        if component.exists():
            attributes = getattr(component.stat(), "st_file_attributes", 0)
            if attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError(f"Reparse points are not accepted: {component}")


def _read(path: Path, maximum: int) -> bytes:
    _unlinked(path)
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"Missing, non-file or oversized input: {path}")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError(f"Oversized input: {path}")
    return data


def _inventory(directory: Path, suffixes: set[str]) -> list[Path]:
    _unlinked(directory)
    if not directory.is_dir():
        raise ValueError(f"Required input directory missing: {directory.name}")
    files = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        _unlinked(path)
        if not path.is_file() or path.suffix.lower() not in suffixes:
            raise ValueError(f"Unexpected dataset entry: {path}")
        files.append(path)
    if len({p.name.casefold() for p in files}) != len(files):
        raise ValueError("Case-ambiguous input filenames are not accepted")
    return files


def _one(parent: ET.Element, name: str) -> ET.Element:
    children = parent.findall(name)
    if len(children) != 1:
        raise ValueError(f"Expected exactly one {name} in {parent.tag}")
    return children[0]


def _text(parent: ET.Element, name: str) -> str:
    node = _one(parent, name)
    if len(node) or not node.text or not node.text.strip():
        raise ValueError(f"Expected nonempty text in {name}")
    return node.text.strip()


def _integer(parent: ET.Element, name: str) -> int:
    value = _text(parent, name)
    if not re.fullmatch(r"-?\d+", value, flags=re.ASCII):
        raise ValueError(f"Expected integer {name}")
    return int(value)


def _parse(data: bytes, annotation: Path) -> dict:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Annotation XML must be UTF-8") from error
    if re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, flags=re.IGNORECASE):
        raise ValueError("DTD and entity declarations are forbidden")
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise ValueError(f"Invalid XML: {annotation.name}") from error
    if root.tag != "annotation":
        raise ValueError("Expected annotation XML root")
    filename = _text(root, "filename")
    if (
        filename in (".", "..")
        or any(char in filename for char in "/\\:")
        or Path(filename).name != filename
        or Path(filename).stem != annotation.stem
    ):
        raise ValueError(f"Unsafe or mismatched annotation filename: {filename!r}")
    size = _one(root, "size")
    width, height = _integer(size, "width"), _integer(size, "height")
    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
        raise ValueError("Invalid or oversized annotated image dimensions")
    if len(size.findall("depth")) > 1:
        raise ValueError("Repeated image depth")
    if size.find("depth") is not None and _integer(size, "depth") not in (1, 3, 4):
        raise ValueError("Unsupported image depth")
    objects = []
    for index, node in enumerate(root.findall("object")):
        name = _text(node, "name")
        if name not in CLASS_NAMES:
            raise ValueError(f"Unknown annotated class: {name!r}")
        bounds = _one(node, "bndbox")
        box = [_integer(bounds, key) for key in ("xmin", "ymin", "xmax", "ymax")]
        objects.append({"object_index": index, "class_name": name, "original_box": box})
    # Every object must be a direct child; no nested records may be ignored.
    if len(list(root.iter("object"))) != len(objects):
        raise ValueError("Nested object annotations are not accepted")
    return {"filename": filename, "width": width, "height": height, "objects": objects}


def _image(data: bytes, width: int, height: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (width, height):
                raise ValueError("Decoded image dimensions differ from annotation")
            if image.format not in ("PNG", "JPEG") or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Expected a single-frame PNG or JPEG")
            image.load()
            return image.convert("RGB")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Image cannot be safely decoded") from error


def _bounds(
    box: list[int], width: int, height: int, convention: str, policy: str
) -> tuple[list[int] | None, str | None]:
    left, top, right, bottom = box
    if convention == "voc-1-based-inclusive":
        left -= 1
        top -= 1
    effective = [left, top, right, bottom]
    if left >= right or top >= bottom:
        return None, "empty_or_reversed_box"
    if left < 0 or top < 0 or right > width or bottom > height:
        if policy == "reject":
            return None, "out_of_bounds"
        effective = [max(0, left), max(0, top), min(width, right), min(height, bottom)]
        if effective[0] >= effective[2] or effective[1] >= effective[3]:
            return None, "empty_after_clipping"
        return effective, "clipped_to_image"
    return effective, None


def prepare_annotations(
    source: Path, destination: Path, *, coordinate_convention: str, bounds_policy: str = "reject"
) -> dict:
    """Create a new crop dataset and receipt, without changing the raw corpus.

    Exact image bytes with differing normalized object multisets are quarantined
    in their entirety. Identical annotations use the lexical filename winner.
    Bounds errors exclude the affected object; unsafe schemas/images abort.
    ``clip`` must be requested explicitly and every adjustment is recorded.
    """
    if coordinate_convention not in COORDINATE_CONVENTIONS:
        raise ValueError("Choose an explicit supported coordinate convention")
    if bounds_policy not in ("reject", "clip"):
        raise ValueError("bounds_policy must be reject or clip")
    source, destination = Path(source).absolute(), Path(destination).absolute()
    _unlinked(source)
    _unlinked(destination)
    if os.path.lexists(destination):
        raise FileExistsError("Destination exists; use a new exclusive directory")
    source = source.resolve()
    target = destination.resolve()
    if target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Source and destination must not contain each other")
    images = _inventory(source / "images", IMAGE_SUFFIXES)
    annotations = _inventory(source / "annotations", {".xml"})
    if not images or not annotations:
        raise ValueError("Images and annotations must be nonempty")
    image_paths = {path.name: path for path in images}
    paired = set()
    records = []
    inputs = []
    for path in annotations:
        xml_bytes = _read(path, MAX_XML_BYTES)
        record = _parse(xml_bytes, path)
        filename = record["filename"]
        if filename not in image_paths or filename in paired:
            raise ValueError(f"Missing or multiply paired image: {filename}")
        paired.add(filename)
        image_bytes = _read(image_paths[filename], MAX_IMAGE_BYTES)
        _image(image_bytes, record["width"], record["height"]).close()
        record.update(
            source_image=f"images/{filename}",
            source_annotation=f"annotations/{path.name}",
            source_image_sha256=_sha(image_bytes),
            source_annotation_sha256=_sha(xml_bytes),
        )
        records.append(record)
        inputs.extend(
            [
                {"path": record["source_image"], "sha256": record["source_image_sha256"]},
                {"path": record["source_annotation"], "sha256": record["source_annotation_sha256"]},
            ]
        )
    if paired != set(image_paths):
        raise ValueError("Every image must have exactly one annotation")
    by_hash = defaultdict(list)
    for record in records:
        by_hash[record["source_image_sha256"]].append(record)
    exclusions, duplicates, chosen = [], [], []
    for digest, group in sorted(by_hash.items()):
        group.sort(key=lambda r: r["source_image"])
        signatures = {
            tuple(sorted((obj["class_name"], *obj["original_box"]) for obj in record["objects"]))
            for record in group
        }
        members = [
            {
                key: record[key]
                for key in (
                    "source_image",
                    "source_annotation",
                    "source_annotation_sha256",
                    "objects",
                )
            }
            for record in group
        ]
        if len(signatures) != 1:
            exclusions.append(
                {
                    "reason": "conflicting_duplicate_annotations",
                    "source_image_sha256": digest,
                    "members": members,
                }
            )
            continue
        chosen.append(group[0])
        if len(group) > 1:
            duplicates.append(
                {
                    "source_image_sha256": digest,
                    "representative": group[0]["source_image"],
                    "reason": "identical_image_and_normalized_object_multiset",
                    "members": members,
                }
            )
    receipt = {
        "schema_version": 1,
        "class_names": list(CLASS_NAMES),
        "coordinate_convention": coordinate_convention,
        "bounds_policy": bounds_policy,
        "duplicate_policy": "quarantine_conflicting_groups; lexical_representative_for_identical_annotations",
        "group_definition": "source_image_sha256; not verified person identity",
        "group_csv_paths_relative_to": "crops",
        "inputs": sorted(inputs, key=lambda row: row["path"]),
        "exclusions": exclusions,
        "duplicate_groups": duplicates,
        "rows": [],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _unlinked(destination.parent)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        for class_name in CLASS_NAMES:
            (staging / "crops" / class_name).mkdir(parents=True)
        for record in chosen:
            data = _read(source / record["source_image"], MAX_IMAGE_BYTES)
            if _sha(data) != record["source_image_sha256"]:
                raise ValueError("Source image changed during preparation")
            if not record["objects"]:
                exclusions.append(
                    {
                        "reason": "no_annotated_objects",
                        "source_image": record["source_image"],
                        "source_image_sha256": record["source_image_sha256"],
                    }
                )
            with _image(data, record["width"], record["height"]) as image:
                for obj in record["objects"]:
                    effective, action = _bounds(
                        obj["original_box"],
                        record["width"],
                        record["height"],
                        coordinate_convention,
                        bounds_policy,
                    )
                    provenance = {
                        key: record[key]
                        for key in (
                            "source_image",
                            "source_annotation",
                            "source_image_sha256",
                            "source_annotation_sha256",
                        )
                    }
                    provenance.update(obj)
                    if effective is None:
                        exclusions.append({**provenance, "reason": action})
                        continue
                    relative = f"{obj['class_name']}/{record['source_image_sha256']}-{obj['object_index']:05d}.png"
                    crop_path = staging / "crops" / relative
                    with image.crop(effective) as crop:
                        crop.save(crop_path, format="PNG")
                    receipt["rows"].append(
                        {
                            **provenance,
                            "path": relative,
                            "group": record["source_image_sha256"],
                            "label": CLASS_NAMES.index(obj["class_name"]),
                            "effective_box_half_open": effective,
                            "bounds_action": action or "unchanged",
                            "sha256": _sha(crop_path.read_bytes()),
                            "size": crop_path.stat().st_size,
                        }
                    )
        receipt["counts"] = {
            "source_images": len(images),
            "source_annotations": len(annotations),
            "source_objects": sum(len(record["objects"]) for record in records),
            "unique_source_image_hashes": len(by_hash),
            "crops": len(receipt["rows"]),
            "crops_by_class": dict(
                sorted(Counter(row["class_name"] for row in receipt["rows"]).items())
            ),
            "excluded_image_hash_groups": sum(
                item["reason"] == "conflicting_duplicate_annotations" for item in exclusions
            ),
            "quarantined_source_images": sum(
                len(item["members"])
                for item in exclusions
                if item["reason"] == "conflicting_duplicate_annotations"
            ),
            "quarantined_source_objects": sum(
                len(member["objects"])
                for item in exclusions
                if item["reason"] == "conflicting_duplicate_annotations"
                for member in item["members"]
            ),
            "excluded_objects": sum(
                item["reason"] != "conflicting_duplicate_annotations" and "object_index" in item
                for item in exclusions
            ),
        }
        with (staging / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["path", "group"])
            writer.writeheader()
            writer.writerows(
                {key: row[key] for key in ("path", "group")} for row in receipt["rows"]
            )
        (staging / "preparation.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        # Detect changed, removed or added raw files before publishing output.
        if (
            _inventory(source / "images", IMAGE_SUFFIXES) != images
            or _inventory(source / "annotations", {".xml"}) != annotations
        ):
            raise ValueError("Source inventory changed during preparation")
        for item in receipt["inputs"]:
            limit = MAX_XML_BYTES if item["path"].startswith("annotations/") else MAX_IMAGE_BYTES
            if _sha(_read(source / item["path"], limit)) != item["sha256"]:
                raise ValueError("Source input changed during preparation")
        _unlinked(destination)
        if os.path.lexists(destination):
            raise FileExistsError("Destination appeared during preparation")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source", type=Path, help="Raw directory containing images/ and annotations/"
    )
    parser.add_argument("destination", type=Path, help="New exclusive output directory")
    parser.add_argument("--coordinate-convention", required=True, choices=COORDINATE_CONVENTIONS)
    parser.add_argument("--bounds-policy", choices=("reject", "clip"), default="reject")
    args = parser.parse_args()
    receipt = prepare_annotations(
        args.source,
        args.destination,
        coordinate_convention=args.coordinate_convention,
        bounds_policy=args.bounds_policy,
    )
    print(json.dumps(receipt["counts"], indent=2))


if __name__ == "__main__":
    main()
