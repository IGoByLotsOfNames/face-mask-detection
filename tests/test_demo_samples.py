"""Included illustrations exercise the real preparation and grouped split path."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from mask_detection.annotations import CLASS_NAMES, prepare_annotations
from mask_detection.data import create_split, read_groups, validate_manifest

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "sample-data" / "demo"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


class IncludedDemoSamplesTests(unittest.TestCase):
    def test_fixture_hashes_and_prediction_contract(self):
        manifest = read_json(SAMPLES / "manifest.json")
        self.assertEqual("synthetic-software-demo", manifest["fixture_kind"])
        self.assertEqual(list(CLASS_NAMES), manifest["class_names"])
        self.assertEqual("zero-based-half-open", manifest["coordinate_convention"])
        self.assertEqual("reject", manifest["bounds_policy"])
        actual = hashes(SAMPLES)
        self.assertEqual(set(actual) - {"manifest.json", "README.md"}, set(manifest["files"]))
        for name, digest in manifest["files"].items():
            self.assertEqual(digest, actual[name], name)
        predictions = read_json(SAMPLES / "predictions.json")
        self.assertEqual(27, len(predictions))
        self.assertEqual(27, len({(r["source_image"], r["object_index"]) for r in predictions}))
        for row in predictions:
            self.assertEqual(3, len(row["probabilities"]))
            self.assertAlmostEqual(1, sum(row["probabilities"]))

    def test_real_pipeline_matches_independent_expectations(self):
        expected = read_json(SAMPLES / "expected.json")
        before = hashes(SAMPLES)
        with tempfile.TemporaryDirectory() as directory:
            prepared, split = Path(directory) / "prepared", Path(directory) / "split"
            receipt = prepare_annotations(
                SAMPLES / "raw", prepared, coordinate_convention="zero-based-half-open"
            )
            self.assertEqual(expected["counts"], receipt["counts"])
            self.assertEqual(
                expected["duplicate_representative"],
                receipt["duplicate_groups"][0]["representative"],
            )
            self.assertEqual(27, len({row["sha256"] for row in receipt["rows"]}))
            group_map = read_groups(prepared / "groups.csv")
            manifest = create_split(prepared / "crops", split, groups=group_map, seed=17)
            self.assertEqual(manifest, validate_manifest(split))
            self.assertEqual(9, len({row["group"] for row in manifest["rows"]}))
            for group in set(group_map.values()):
                self.assertEqual(
                    1, len({row["split"] for row in manifest["rows"] if row["group"] == group})
                )
            self.assertEqual(
                {(name, label) for name in ("train", "validation", "test") for label in range(3)},
                {(row["split"], row["label"]) for row in manifest["rows"]},
            )
            predictions = {
                (row["source_image"], row["object_index"]): row["probabilities"]
                for row in read_json(SAMPLES / "predictions.json")
            }
            matrix = [[0] * 3 for _ in range(3)]
            self.assertEqual(
                set(predictions), {(r["source_image"], r["object_index"]) for r in receipt["rows"]}
            )
            for row in receipt["rows"]:
                values = predictions[(row["source_image"], row["object_index"])]
                matrix[row["label"]][max(range(3), key=values.__getitem__)] += 1
            self.assertEqual(expected["confusion_matrix"], matrix)
        self.assertEqual(before, hashes(SAMPLES))

    def test_generator_reproduces_included_files_and_refuses_overwrite(self):
        path = ROOT / "tools" / "generate_demo_samples.py"
        spec = importlib.util.spec_from_file_location("fixture_generator", path)
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "samples"
            generator.generate(destination)
            self.assertEqual(hashes(SAMPLES), hashes(destination))
            with self.assertRaises(FileExistsError):
                generator.generate(destination)


if __name__ == "__main__":
    unittest.main()
