"""Create the included schematic input fixture in a new exclusive directory.

This is deterministic illustration and test-data generation, not model output.
The image generator uses Pillow drawing primitives and contains no photographs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image, ImageDraw

CLASS_NAMES = ("mask_weared_incorrect", "with_mask", "without_mask")
WIDTH, HEIGHT = 480, 224
BOXES = ((18, 42, 146, 194), (176, 42, 304, 194), (334, 42, 462, 194))


def _json(path: Path, value: object) -> None:
    # Raw hashes must reproduce on both Windows and Unix checkouts.
    path.write_bytes((json.dumps(value, indent=2) + "\n").encode("utf-8"))


def _scene(index: int) -> Image.Image:
    """Nine distinct scenes with three deliberately schematic heads apiece."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "#eef4fa")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((6, 6, 474, 218), radius=16, fill="#ffffff")
    draw.text((20, 18), f"SCHEMATIC DEMO / SCENE {index:02d}", fill="#23334d")
    labels = ("MASK BELOW NOSE", "MASK WORN", "NO MASK")
    for label, box in enumerate(BOXES):
        x, y, right, bottom = box
        card = (230 - index * 3, 237 - label * 4, 245 - index * 2)
        draw.rounded_rectangle(box, radius=12, fill=card)
        centre = x + 64
        skin = (239 - index * 4, 195 - label * 9, 156 + index * 3)
        draw.rounded_rectangle((x + 23, y + 111, right - 23, bottom), radius=19, fill="#506780")
        draw.ellipse((x + 24, y + 13, right - 24, y + 120), fill=skin)
        draw.arc((x + 23, y + 11, right - 23, y + 95), 178, 359, fill="#34445d", width=13)
        draw.ellipse((centre - 21, y + 55, centre - 16, y + 60), fill="#23334d")
        draw.ellipse((centre + 16, y + 55, centre + 21, y + 60), fill="#23334d")
        draw.line((centre, y + 66, centre - 3, y + 78, centre + 4, y + 78), fill="#9d675d", width=2)
        if label < 2:
            top = y + (88 if label == 0 else 68)
            draw.line((x + 27, top + 4, x + 13, top - 2), fill="#5989a8", width=2)
            draw.line((right - 27, top + 4, right - 13, top - 2), fill="#5989a8", width=2)
            draw.rounded_rectangle((x + 28, top, right - 28, y + 108), radius=7, fill="#80c4df")
            draw.line((x + 37, top + 8, right - 37, top + 8), fill="#d8f3ff", width=2)
            draw.line((x + 39, top + 15, right - 39, top + 15), fill="#d8f3ff", width=2)
        else:
            draw.arc((centre - 13, y + 80, centre + 13, y + 97), 0, 180, fill="#995f54", width=2)
        # This visible badge also guarantees different bytes for every crop.
        draw.rounded_rectangle((x + 8, bottom - 21, x + 53, bottom - 5), radius=4, fill="#ffffff")
        draw.text((x + 13, bottom - 19), f"{index:02d}.{label}", fill="#23334d")
        draw.text((x + 5, 201), labels[label], fill="#506780")
    return image


def _annotation(path: Path, filename: str, objects: list[tuple[str, tuple[int, ...]]]) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    for key, value in (("width", WIDTH), ("height", HEIGHT), ("depth", 3)):
        ET.SubElement(size, key).text = str(value)
    for name, box in objects:
        item = ET.SubElement(root, "object")
        ET.SubElement(item, "name").text = name
        bounds = ET.SubElement(item, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bounds, key).text = str(value)
    ET.indent(root, space="  ")
    path.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n")


README = """# Included schematic software fixture

These images are original illustrations drawn with Pillow shapes. They depict
no real people and contain no source-dataset photographs. They test software
behaviour; they are not evidence of mask recognition performance.

`raw/images` contains nine distinct three-head scenes, one exact duplicate of
scene-00, and two same-byte sources with conflicting annotations. Scene-08 also
has one deliberately out-of-bounds object. The annotations explicitly use
**zero-based, half-open** boxes: `[xmin, ymin, xmax, ymax)`.

The real preparation code must keep 27 crops (nine per documented class), retain
scene-00 as the exact-duplicate representative, quarantine both conflicting
sources, and reject the out-of-bounds object. Every retained crop has distinct
bytes. A source-image SHA-256 is the grouping unit; it is not a person identity.

`predictions.json` contains 27 hand-authored probability vectors, keyed by
`source_image` and `object_index`. These vectors are **synthetic, not model
output**. For scenes 00, 03 and 06 the predicted class deliberately rotates to
the next class. Other scenes predict the annotated class. Each vector assigns
0.8 to its selected class and 0.1 to the other classes. The errors exercise the
confusion matrix and per-class report; their scores must never be reported as
model accuracy. All vectors use this exact class order:

1. `mask_weared_incorrect`
2. `with_mask`
3. `without_mask`

`manifest.json` lists SHA-256 hashes for every raw input, `predictions.json`,
and `expected.json`. Paths are relative to this directory. The manifest itself
and this README are outside its hash registry. `expected.json` holds independently
specified counts and the all-27 confusion matrix; these are fixture expectations,
not estimates learned from the images. Split-specific results depend on the
real grouped splitter and are calculated by the demo from the retained rows.

To regenerate into a NEW directory (Pillow required):

```powershell
# Repository working directory
python tools/generate_demo_samples.py --output sample-data/demo-new
```

Generation never overwrites an existing directory. Pixel determinism is tested
in the installed environment; a different Pillow version may encode PNG bytes
differently. The included files and their hashes are the executable fixture.
"""


def generate(destination: Path) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    images = destination / "raw" / "images"
    annotations = destination / "raw" / "annotations"
    images.mkdir(parents=True)
    annotations.mkdir()
    objects = list(zip(CLASS_NAMES, BOXES))
    predictions = []
    for index in range(9):
        name = f"scene-{index:02d}"
        with _scene(index) as image:
            image.save(images / f"{name}.png", format="PNG")
        annotated = objects
        if index == 8:
            annotated = [*objects, ("with_mask", (450, 42, 486, 194))]
        _annotation(annotations / f"{name}.xml", f"{name}.png", annotated)
        for label in range(3):
            chosen = (label + 1) % 3 if index % 3 == 0 else label
            predictions.append(
                {
                    "source_image": f"images/{name}.png",
                    "object_index": label,
                    "probabilities": [0.8 if category == chosen else 0.1 for category in range(3)],
                }
            )

    shutil.copyfile(images / "scene-00.png", images / "z-identical.png")
    _annotation(annotations / "z-identical.xml", "z-identical.png", list(reversed(objects)))
    # A distinct scene ensures the intentionally conflicting pair does not
    # accidentally quarantine one of the nine retained source scenes.
    with _scene(9) as image:
        image.save(images / "conflict-a.png", format="PNG")
    shutil.copyfile(images / "conflict-a.png", images / "conflict-b.png")
    _annotation(annotations / "conflict-a.xml", "conflict-a.png", objects)
    _annotation(
        annotations / "conflict-b.xml",
        "conflict-b.png",
        [("with_mask", BOXES[0]), *objects[1:]],
    )
    _json(destination / "predictions.json", predictions)
    _json(
        destination / "expected.json",
        {
            "schema_version": 1,
            "fixture_kind": "synthetic-software-demo",
            "class_names": list(CLASS_NAMES),
            "counts": {
                "source_images": 12,
                "source_annotations": 12,
                "source_objects": 37,
                "unique_source_image_hashes": 10,
                "crops": 27,
                "crops_by_class": dict.fromkeys(CLASS_NAMES, 9),
                "excluded_image_hash_groups": 1,
                "quarantined_source_images": 2,
                "quarantined_source_objects": 6,
                "excluded_objects": 1,
            },
            "duplicate_groups": 1,
            "duplicate_representative": "images/scene-00.png",
            "quarantined_sources": ["images/conflict-a.png", "images/conflict-b.png"],
            "excluded_object": {"source_image": "images/scene-08.png", "object_index": 3},
            "prediction_rows": 27,
            "confusion_matrix_scope": "all 27 retained fixture crops; synthetic vectors, not model output",
            "confusion_matrix": [[6, 3, 0], [0, 6, 3], [3, 0, 6]],
        },
    )
    files = {
        path.relative_to(destination).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "fixture_kind": "synthetic-software-demo",
        "coordinate_convention": "zero-based-half-open",
        "bounds_policy": "reject",
        "class_names": list(CLASS_NAMES),
        "raw_directory": "raw",
        "prediction_file": "predictions.json",
        "expectation_file": "expected.json",
        "prediction_origin": "hand-authored synthetic probabilities; not model output",
        "group_definition": "source image SHA-256, not person identity",
        "hash_scope": "all raw inputs, predictions.json and expected.json; excludes manifest and README",
        "files": files,
    }
    _json(destination / "manifest.json", manifest)
    (destination / "README.md").write_bytes(README.encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New exclusive fixture directory"
    )
    args = parser.parse_args()
    manifest = generate(args.output)
    print(f"Created {len(manifest['files'])} hashed fixture files in {args.output}")


if __name__ == "__main__":
    main()
