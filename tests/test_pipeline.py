import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mask_detection.artifacts import load_metadata, save_metadata
from mask_detection.data import create_split, load_rgb, read_groups, safe_path, validate_manifest
from mask_detection.metrics import binary_metrics, evaluation_report


def fixture(root, count=12):
    groups = {}
    for label, name in enumerate(("class_a", "class_b")):
        (root / name).mkdir(parents=True)
        for i in range(count):
            path = root / name / f"{i:02}.png"
            Image.new("RGB", (9, 7), (label * 180, i * 7, i + 10)).save(path)
            groups[path.relative_to(root).as_posix()] = f"{name}-case-{i // 2}"
    return groups


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.groups = fixture(self.source)

    def split(self, name="split", **kwargs):
        return create_split(self.source, self.root / name, groups=self.groups, **kwargs)

    def test_deterministic_and_source_preserved(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*.png")}
        a, b = self.split("a"), self.split("b")
        self.assertEqual(a, b)
        self.assertEqual(a, validate_manifest(self.root / "a"))
        self.assertEqual(
            before, {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob("*.png")}
        )
        for group in self.groups.values():
            self.assertEqual(1, len({r["split"] for r in a["rows"] if r["group"] == group}))

    def test_reject_rerun_and_nested_output(self):
        self.split()
        with self.assertRaises(FileExistsError):
            self.split(seed=999)
        with self.assertRaises(ValueError):
            create_split(self.source, self.source / "output", groups=self.groups)

    def test_explicit_group_policy_required(self):
        with self.assertRaises(ValueError):
            create_split(self.source, self.root / "bad")
        with self.assertRaises(ValueError):
            create_split(
                self.source, self.root / "bad", groups=self.groups, independent_images=True
            )
        self.assertEqual(
            "asserted_independent_images",
            create_split(self.source, self.root / "ok", independent_images=True)["grouping"],
        )

    def test_duplicate_components_join_distinct_groups(self):
        shutil.copyfile(self.source / "class_a/00.png", self.source / "class_a/10.png")
        m = self.split()
        rows = {r["source"]: r for r in m["rows"]}
        self.assertEqual(rows["class_a/00.png"]["split"], rows["class_a/11.png"]["split"])

    def test_conflicting_duplicate_labels_rejected(self):
        shutil.copyfile(self.source / "class_a/00.png", self.source / "class_b/00.png")
        with self.assertRaises(ValueError):
            self.split()

    def test_tiny_and_invalid_ratios_rejected(self):
        for value in (0, 1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.split(train=value)
        self.groups = {p: "one-group" for p in self.groups}
        with self.assertRaises(ValueError):
            self.split()

    def test_missing_or_extra_group_rows_rejected(self):
        key = next(iter(self.groups))
        group = self.groups.pop(key)
        with self.assertRaises(ValueError):
            self.split()
        self.groups[key] = group
        self.groups["class_a/absent.png"] = "extra"
        with self.assertRaises(ValueError):
            self.split()

    def test_mixed_class_patient_groups_stay_together(self):
        self.groups = {p: p.split("/")[1][:2] for p in self.groups}
        m = self.split()
        for group in self.groups.values():
            self.assertEqual(1, len({r["split"] for r in m["rows"] if r["group"] == group}))

    def test_tamper_extra_and_missing_files_detected(self):
        m = self.split()
        path = self.root / "split" / m["rows"][0]["path"]
        original = path.read_bytes()
        path.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            validate_manifest(self.root / "split")
        path.write_bytes(original)
        extra = self.root / "split/extra.txt"
        extra.write_text("untracked")
        with self.assertRaises(ValueError):
            validate_manifest(self.root / "split")
        extra.unlink()
        path.unlink()
        with self.assertRaises(ValueError):
            validate_manifest(self.root / "split")

    def test_manifest_leakage_and_path_traversal_rejected(self):
        m = self.split()
        row = next(r for r in m["rows"] if r["split"] == "test")
        row["group"] = next(r["group"] for r in m["rows"] if r["split"] == "train")
        (self.root / "split/manifest.json").write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            validate_manifest(self.root / "split")
        for path in ("../outside.png", "/root.png", "a\\b.png", "a/./b", "a//b"):
            with self.assertRaises(ValueError):
                safe_path(self.root, path)

    def test_group_csv_duplicate_rejected(self):
        csv = self.root / "groups.csv"
        csv.write_text("path,group\na.png,id1\na.png,id2\n")
        with self.assertRaises(ValueError):
            read_groups(csv)

    def test_rgb_and_resize_contract(self):
        path = self.root / "red.png"
        Image.new("RGB", (4, 5), (255, 0, 0)).save(path)
        arr = load_rgb(path, (3, 2))
        self.assertEqual((3, 2, 3), arr.shape)
        self.assertEqual(np.float32, arr.dtype)
        np.testing.assert_array_equal(arr[0, 0], [255, 0, 0])

    def test_metadata_hash_class_order_and_tamper(self):
        m = self.split()
        model = self.root / "fake.keras"
        model.write_bytes(b"synthetic artifact, not a Keras model")
        metadata = save_metadata(model, m, size=(10, 20), architecture="fixture", seed=42)
        self.assertEqual(metadata, load_metadata(model))
        self.assertEqual(m["class_names"], metadata["class_names"])
        model.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load_metadata(model)


class MetricTests(unittest.TestCase):
    def test_known_auc_and_confusion(self):
        report = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
        self.assertEqual([[2, 0], [1, 1]], report["confusion_matrix"])
        self.assertAlmostEqual(0.75, report["roc_auc"])
        self.assertAlmostEqual(0.75, report["accuracy"])
        self.assertAlmostEqual((0.01 + 0.16 + 0.65**2 + 0.2**2) / 4, report["brier_score"])

    def test_auc_ties_and_single_class(self):
        self.assertEqual(0.5, binary_metrics([0, 1], [0.5, 0.5])["roc_auc"])
        report = binary_metrics([1, 1], [0.9, 0.8])
        self.assertIsNone(report["roc_auc"])
        self.assertIsNone(report["specificity"])
        json.dumps(report, allow_nan=False)

    def test_invalid_inputs(self):
        for probabilities in ([], [float("nan")], [float("inf")], [-0.1], [1.1], [0.2, 0.4]):
            with self.assertRaises(ValueError):
                binary_metrics([0], probabilities)
        with self.assertRaises(ValueError):
            binary_metrics([2], [0.1])
        with self.assertRaises(ValueError):
            binary_metrics([0], [0.1], threshold=2)

    def test_auc_pairwise_oracle(self):
        import random

        rng = random.Random(7)
        for _ in range(50):
            labels = [0] * 5 + [1] * 7
            scores = [rng.randrange(5) / 4 for _ in labels]
            expected = sum((p > n) + 0.5 * (p == n) for p in scores[5:] for n in scores[:5]) / 35
            self.assertAlmostEqual(expected, binary_metrics(labels, scores)["roc_auc"])

    def test_aligned_ensemble_and_length_rejection(self):
        rows = [dict(source=f"{i}.png", group=str(i), sha256=str(i), label=i) for i in (0, 1)]
        r = evaluation_report(rows, {"a": [0.1, 0.9], "b": [0.3, 0.7]}, ["negative", "positive"])
        self.assertAlmostEqual(0.2, r["samples"][0]["probabilities"]["ensemble"])
        self.assertEqual("positive", r["positive_class"])
        self.assertEqual(1, r["metrics"]["ensemble"]["roc_auc"])
        with self.assertRaises(ValueError):
            evaluation_report(rows, {"a": [0.2]}, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
