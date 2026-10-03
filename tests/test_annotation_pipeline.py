"""Synthetic raw annotations through crop preparation and grouped splitting."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mask_detection.annotations import CLASS_NAMES, prepare_annotations
from mask_detection.data import create_split, read_groups, validate_manifest


def hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def write_annotation(path, image_name, objects):
    object_xml = []
    for name, box in objects:
        coordinates = "".join(
            f"<{key}>{value}</{key}>" for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box)
        )
        object_xml.append(f"<object><name>{name}</name><bndbox>{coordinates}</bndbox></object>")
    path.write_text(
        f"<annotation><filename>{image_name}</filename>"
        "<size><width>12</width><height>4</height><depth>3</depth></size>"
        f"{''.join(object_xml)}</annotation>",
        encoding="utf-8",
    )


def raw_fixture(root, *, conflicting_crop=False):
    """Small coloured rectangles, never photographs or camera captures."""
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir()
    objects = [(name, (label * 4, 0, label * 4 + 4, 4)) for label, name in enumerate(CLASS_NAMES)]
    for group in range(9):
        image = Image.new("RGB", (12, 4))
        for label, (_, box) in enumerate(objects):
            # A repeated same-label crop connects two different source images.
            colour_group = 0 if group == 1 and label == 0 else group
            colour = (30 + label * 70, 10 + colour_group * 20, 42)
            if conflicting_crop and group == 2 and label == 1:
                colour = (30, 10, 42)
            image.paste(colour, box)
        name = f"image-{group:02}"
        image.save(root / "images" / f"{name}.png")
        annotated_objects = objects
        if group == 8:
            annotated_objects = objects + [("with_mask", (10, 0, 13, 4))]
        write_annotation(root / "annotations" / f"{name}.xml", f"{name}.png", annotated_objects)

    shutil.copyfile(root / "images/image-00.png", root / "images/z-identical.png")
    write_annotation(
        root / "annotations/z-identical.xml", "z-identical.png", list(reversed(objects))
    )

    image = Image.new("RGB", (12, 4), (245, 231, 219))
    image.save(root / "images/conflict-a.png")
    shutil.copyfile(root / "images/conflict-a.png", root / "images/conflict-b.png")
    write_annotation(root / "annotations/conflict-a.xml", "conflict-a.png", objects)
    changed_objects = [("with_mask", objects[0][1]), *objects[1:]]
    write_annotation(root / "annotations/conflict-b.xml", "conflict-b.png", changed_objects)


class AnnotationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"

    def test_raw_preparation_split_preserves_groups_exclusions_and_provenance(self):
        raw_fixture(self.raw)
        before = hashes(self.raw)
        prepared = self.root / "prepared"
        receipt = prepare_annotations(
            self.raw, prepared, coordinate_convention="zero-based-half-open"
        )
        self.assertEqual(12, receipt["counts"]["source_images"])
        self.assertEqual(37, receipt["counts"]["source_objects"])
        self.assertEqual(27, receipt["counts"]["crops"])
        self.assertEqual(2, receipt["counts"]["quarantined_source_images"])
        self.assertEqual(6, receipt["counts"]["quarantined_source_objects"])
        self.assertEqual(1, receipt["counts"]["excluded_objects"])
        self.assertEqual("images/image-00.png", receipt["duplicate_groups"][0]["representative"])
        conflicts = [
            r for r in receipt["exclusions"] if r["reason"] == "conflicting_duplicate_annotations"
        ]
        self.assertEqual(1, len(conflicts))
        self.assertEqual(
            {"images/conflict-a.png", "images/conflict-b.png"},
            {r["source_image"] for r in conflicts[0]["members"]},
        )
        self.assertNotIn(conflicts[0]["source_image_sha256"], {r["group"] for r in receipt["rows"]})
        self.assertEqual(
            receipt, json.loads((prepared / "preparation.json").read_text(encoding="utf-8"))
        )
        for item in receipt["inputs"]:
            self.assertEqual(before[item["path"]], item["sha256"])
        prepared_hashes = hashes(prepared)
        groups = read_groups(prepared / "groups.csv")
        self.assertEqual({r["path"]: r["source_image_sha256"] for r in receipt["rows"]}, groups)
        split = self.root / "split"
        manifest = create_split(prepared / "crops", split, groups=groups, seed=17)
        self.assertEqual(manifest, validate_manifest(split))
        self.assertEqual(list(CLASS_NAMES), manifest["class_names"])
        self.assertEqual(27, len(manifest["rows"]))
        by_path = {r["path"]: r for r in receipt["rows"]}
        split_hashes = hashes(split)
        for row in manifest["rows"]:
            original = by_path[row["source"]]
            self.assertEqual(original["sha256"], row["sha256"])
            self.assertEqual(original["label"], row["label"])
            self.assertEqual(before[original["source_image"]], row["group"])
            self.assertEqual(original["sha256"], split_hashes[row["path"]])
            expected_label = CLASS_NAMES.index(original["class_name"])
            expected_box = [expected_label * 4, 0, expected_label * 4 + 4, 4]
            self.assertEqual(expected_label, row["label"])
            self.assertEqual(expected_box, original["original_box"])
            with Image.open(self.raw / original["source_image"]) as source_image:
                with source_image.crop(expected_box) as expected:
                    with Image.open(split / row["path"]) as actual:
                        self.assertEqual(expected.size, actual.size)
                        self.assertEqual(expected.tobytes(), actual.tobytes())
        for group in set(groups.values()):
            self.assertEqual(1, len({r["split"] for r in manifest["rows"] if r["group"] == group}))
        connected = {before["images/image-00.png"], before["images/image-01.png"]}
        self.assertEqual(2, len(connected))
        self.assertEqual(1, len({r["split"] for r in manifest["rows"] if r["group"] in connected}))
        self.assertEqual(
            {
                (split_name, label)
                for split_name in ("train", "validation", "test")
                for label in range(3)
            },
            {(r["split"], r["label"]) for r in manifest["rows"]},
        )
        self.assertEqual(before, hashes(self.raw))
        self.assertEqual(prepared_hashes, hashes(prepared))

    def test_conflicting_generated_crop_labels_abort_split_without_changing_inputs(self):
        raw_fixture(self.raw, conflicting_crop=True)
        before = hashes(self.raw)
        prepared = self.root / "prepared"
        receipt = prepare_annotations(
            self.raw, prepared, coordinate_convention="zero-based-half-open"
        )
        # These images differ as whole sources but contain identical pixels with
        # different labels. Source-level quarantine cannot resolve this conflict.
        same_crop = [
            r
            for r in receipt["rows"]
            if r["source_image"] == "images/image-00.png" and r["class_name"] == CLASS_NAMES[0]
        ][0]
        conflicting = [
            r
            for r in receipt["rows"]
            if r["source_image"] == "images/image-02.png" and r["class_name"] == CLASS_NAMES[1]
        ][0]
        self.assertEqual(same_crop["sha256"], conflicting["sha256"])
        self.assertNotEqual(same_crop["group"], conflicting["group"])
        prepared_hashes = hashes(prepared)
        with self.assertRaisesRegex(ValueError, "conflicting class labels"):
            create_split(
                prepared / "crops", self.root / "split", groups=read_groups(prepared / "groups.csv")
            )
        self.assertFalse((self.root / "split").exists())
        self.assertEqual(before, hashes(self.raw))
        self.assertEqual(prepared_hashes, hashes(prepared))


if __name__ == "__main__":
    unittest.main()
