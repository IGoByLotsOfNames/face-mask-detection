import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mask_detection.annotations import CLASS_NAMES, prepare_annotations


def annotation(filename, objects, width=8, height=6):
    rows = []
    for name, box in objects:
        coordinates = "".join(
            f"<{key}>{value}</{key}>" for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box)
        )
        rows.append(f"<object><name>{name}</name><bndbox>{coordinates}</bndbox></object>")
    return (
        f"<annotation><filename>{filename}</filename><size><width>{width}</width>"
        f"<height>{height}</height><depth>3</depth></size>{''.join(rows)}</annotation>"
    )


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "raw"
        (self.source / "images").mkdir(parents=True)
        (self.source / "annotations").mkdir()

    def add(self, name="one", objects=None, colour=(30, 60, 90)):
        objects = objects if objects is not None else [("with_mask", [0, 0, 4, 4])]
        image = self.source / "images" / f"{name}.png"
        Image.new("RGB", (8, 6), colour).save(image)
        xml = self.source / "annotations" / f"{name}.xml"
        xml.write_text(annotation(image.name, objects), encoding="utf-8")
        return image, xml

    def run_prepare(self, name="prepared", **kwargs):
        return prepare_annotations(
            self.source,
            self.root / name,
            coordinate_convention=kwargs.pop("coordinate_convention", "zero-based-half-open"),
            **kwargs,
        )

    def test_mixed_class_source_group_and_full_provenance(self):
        image, xml = self.add(
            objects=[(name, [i, 0, i + 2, 3]) for i, name in enumerate(CLASS_NAMES)]
        )
        before = {
            path.relative_to(self.source).as_posix(): path.read_bytes()
            for path in self.source.rglob("*")
            if path.is_file()
        }
        report = self.run_prepare()
        self.assertEqual(3, len(report["rows"]))
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        self.assertEqual({digest}, {row["group"] for row in report["rows"]})
        self.assertEqual(list(CLASS_NAMES), report["class_names"])
        with (self.root / "prepared/groups.csv").open(newline="") as stream:
            groups = list(csv.DictReader(stream))
        self.assertEqual([{"path": row["path"], "group": digest} for row in report["rows"]], groups)
        for index, row in enumerate(report["rows"]):
            self.assertEqual("images/one.png", row["source_image"])
            self.assertEqual("annotations/one.xml", row["source_annotation"])
            self.assertEqual(
                hashlib.sha256(xml.read_bytes()).hexdigest(), row["source_annotation_sha256"]
            )
            self.assertEqual(index, row["object_index"])
            with Image.open(self.root / "prepared/crops" / row["path"]) as crop:
                self.assertEqual((2, 3), crop.size)
        self.assertEqual(
            before,
            {
                path.relative_to(self.source).as_posix(): path.read_bytes()
                for path in self.source.rglob("*")
                if path.is_file()
            },
        )
        self.assertEqual(report, json.loads((self.root / "prepared/preparation.json").read_text()))

    def test_conflicting_duplicate_group_quarantines_every_member(self):
        self.add("a", [("with_mask", [0, 0, 4, 4])])
        self.add("b", [("without_mask", [0, 0, 4, 4])])
        self.add("c", [("mask_weared_incorrect", [0, 0, 4, 4])], (99, 10, 2))
        report = self.run_prepare()
        self.assertEqual(1, len(report["rows"]))
        (exclusion,) = report["exclusions"]
        self.assertEqual("conflicting_duplicate_annotations", exclusion["reason"])
        self.assertEqual(
            ["images/a.png", "images/b.png"], [r["source_image"] for r in exclusion["members"]]
        )
        self.assertEqual(1, report["counts"]["excluded_image_hash_groups"])
        self.assertEqual("images/c.png", report["rows"][0]["source_image"])

    def test_object_order_is_normalized_but_multiplicity_is_not(self):
        objects = [("with_mask", [0, 0, 4, 4]), ("without_mask", [4, 0, 8, 4])]
        self.add("b", objects)
        self.add("a", list(reversed(objects)))
        report = self.run_prepare()
        self.assertEqual(2, len(report["rows"]))
        self.assertEqual("images/a.png", report["duplicate_groups"][0]["representative"])
        self.assertEqual([], report["exclusions"])
        self.add("c", objects + objects[:1])
        report = self.run_prepare("with-extra-object")
        self.assertEqual([], report["rows"])
        self.assertEqual(3, len(report["exclusions"][0]["members"]))

    def test_bad_bounds_are_excluded_per_object_and_clip_is_explicit(self):
        self.add(
            objects=[
                ("with_mask", [0, 0, 9, 6]),
                ("without_mask", [1, 1, 3, 3]),
                ("mask_weared_incorrect", [5, 2, 3, 4]),
            ]
        )
        rejected = self.run_prepare()
        self.assertEqual(1, len(rejected["rows"]))
        self.assertEqual(
            {"out_of_bounds", "empty_or_reversed_box"},
            {row["reason"] for row in rejected["exclusions"]},
        )
        clipped = self.run_prepare("clipped", bounds_policy="clip")
        self.assertEqual(2, len(clipped["rows"]))
        changed = next(row for row in clipped["rows"] if row["bounds_action"] == "clipped_to_image")
        self.assertEqual([0, 0, 9, 6], changed["original_box"])
        self.assertEqual([0, 0, 8, 6], changed["effective_box_half_open"])
        self.assertEqual("clip", clipped["bounds_policy"])

    def test_two_coordinate_conventions_have_expected_pixel_extent(self):
        self.add(objects=[("with_mask", [1, 1, 4, 4])])
        half_open = self.run_prepare("half")
        inclusive = self.run_prepare("inclusive", coordinate_convention="voc-1-based-inclusive")
        self.assertEqual([1, 1, 4, 4], half_open["rows"][0]["effective_box_half_open"])
        self.assertEqual([0, 0, 4, 4], inclusive["rows"][0]["effective_box_half_open"])
        self.assertNotEqual(half_open["rows"][0]["sha256"], inclusive["rows"][0]["sha256"])
        self.assertEqual(half_open["rows"][0]["group"], inclusive["rows"][0]["group"])
        with self.assertRaises(TypeError):
            prepare_annotations(self.source, self.root / "unspecified")
        with self.assertRaises(ValueError):
            self.run_prepare("unknown", coordinate_convention="guess")
        with self.assertRaises(ValueError):
            self.run_prepare("policy", bounds_policy="silently-fix")

    def test_invalid_or_unpaired_filenames_abort_without_output(self):
        image, xml = self.add()
        for filename in ("../one.png", r"..\one.png", "C:one.png", "/one.png", "two.png"):
            xml.write_text(annotation(filename, [("with_mask", [0, 0, 3, 3])]), encoding="utf-8")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.run_prepare()
            self.assertFalse((self.root / "prepared").exists())
        xml.write_text(annotation(image.name, [("with_mask", [0, 0, 3, 3])]), encoding="utf-8")
        Image.new("RGB", (8, 6)).save(self.source / "images" / "extra.png")
        with self.assertRaises(ValueError):
            self.run_prepare()

    def test_corrupt_image_and_dimension_mismatch_abort(self):
        image, xml = self.add()
        image.write_bytes(b"not an image")
        with self.assertRaises(ValueError):
            self.run_prepare()
        Image.new("RGB", (7, 6)).save(image)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.run_prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_unsafe_xml_unknown_labels_and_schema_errors_abort(self):
        _, xml = self.add()
        ordinary = annotation("one.png", [("with_mask", [0, 0, 3, 3])])
        cases = [
            '<!DOCTYPE annotation [<!ENTITY x "with_mask">]>'
            + ordinary.replace("with_mask", "&x;"),
            ordinary.replace("with_mask", "unknown"),
            ordinary.replace("<width>8</width>", "<width>8</width><width>8</width>"),
            ordinary.replace("<xmin>0</xmin>", "<xmin>0.5</xmin>"),
            ordinary.replace("<bndbox>", "<other>").replace("</bndbox>", "</other>"),
            ordinary.replace("<annotation>", "<wrong>").replace("</annotation>", "</wrong>"),
        ]
        for content in cases:
            xml.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.run_prepare()
            self.assertFalse((self.root / "prepared").exists())

    def test_existing_and_nested_destination_refused(self):
        self.add()
        self.run_prepare()
        original = (self.root / "prepared/preparation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_prepare()
        self.assertEqual(original, (self.root / "prepared/preparation.json").read_bytes())
        with self.assertRaises(ValueError):
            prepare_annotations(
                self.source, self.source / "out", coordinate_convention="zero-based-half-open"
            )

    def test_exact_crop_duplicates_retain_raw_source_hashes(self):
        first, _ = self.add("a", [("with_mask", [0, 0, 2, 2])])
        second, _ = self.add("b", [("with_mask", [0, 0, 2, 2])])
        with Image.open(second) as im:
            changed = im.copy()
        changed.putpixel((7, 5), (201, 15, 77))
        changed.save(second)
        report = self.run_prepare()
        self.assertEqual(2, len({row["group"] for row in report["rows"]}))
        self.assertEqual(1, len({row["sha256"] for row in report["rows"]}))
        self.assertNotEqual(
            hashlib.sha256(first.read_bytes()).hexdigest(),
            hashlib.sha256(second.read_bytes()).hexdigest(),
        )

    def test_source_changed_during_preparation_aborts_and_cleans_staging(self):
        image, _ = self.add()
        original_save = Image.Image.save
        modified = False

        def save_then_modify(im, target, *args, **kwargs):
            nonlocal modified
            result = original_save(im, target, *args, **kwargs)
            if not modified:
                modified = True
                image.write_bytes(image.read_bytes() + b"changed")
            return result

        with (
            patch.object(Image.Image, "save", save_then_modify),
            self.assertRaisesRegex(ValueError, "Source input changed"),
        ):
            self.run_prepare()
        self.assertFalse((self.root / "prepared").exists())
        self.assertEqual([], list(self.root.glob(".prepare-*")))

    def test_symlink_source_is_rejected_when_platform_allows_creation(self):
        image, _ = self.add()
        target = self.root / "external.png"
        shutil.copyfile(image, target)
        image.unlink()
        try:
            image.symlink_to(target)
        except OSError:
            self.skipTest("Platform does not permit symlink creation")
        with self.assertRaisesRegex(ValueError, "Links|Reparse"):
            self.run_prepare()

    def test_cli_help_and_missing_coordinate_are_safe(self):
        command = [sys.executable, "-m", "mask_detection.annotations"]
        help_result = subprocess.run(command + ["--help"], capture_output=True, text=True)
        self.assertEqual(0, help_result.returncode, help_result.stderr)
        self.assertIn("--coordinate-convention", help_result.stdout)
        self.add()
        result = subprocess.run(
            command + [str(self.source), str(self.root / "out")], capture_output=True, text=True
        )
        self.assertEqual(2, result.returncode)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
