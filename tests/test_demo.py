"""Exercise the actual reviewer demo, integrity checks and exclusive output."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mask_detection.demo import read_fixture, run_demo

SAMPLES = Path(__file__).resolve().parents[1] / "sample-data/demo"


def tree_hashes(root):
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_real_pipeline_counts_reports_grouping_and_source_unchanged(self):
        before = tree_hashes(SAMPLES)
        result = run_demo(SAMPLES, self.root / "complete")
        self.assertEqual("synthetic-software-demo", result["kind"])
        self.assertEqual(12, result["counts"]["source_images"])
        self.assertEqual(27, result["counts"]["crops"])
        self.assertEqual(1, result["counts"]["duplicate_groups"])
        self.assertEqual(2, result["counts"]["quarantined_source_images"])
        self.assertEqual(1, result["counts"]["excluded_objects"])
        self.assertEqual([[6, 3, 0], [0, 6, 3], [3, 0, 6]], result["confusion_matrix"])
        self.assertEqual(27, len(result["samples"]))
        self.assertEqual(6, len(result["checks"]))
        self.assertEqual({"passed"}, {row["status"] for row in result["checks"]})
        self.assertIn("scripted predictions", result["notice"])
        report = (self.root / "complete/report.html").read_text(encoding="utf-8")
        self.assertIn("data:image/png;base64,", report)
        self.assertEqual(
            result, json.loads((self.root / "complete/summary.json").read_text(encoding="utf-8"))
        )
        self.assertFalse((self.root / "complete/failure.json").exists())
        self.assertEqual(before, tree_hashes(SAMPLES))

    def test_existing_output_is_untouched(self):
        output = self.root / "old"
        output.mkdir()
        (output / "keep.txt").write_text("user output", encoding="utf-8")
        before = tree_hashes(output)
        with self.assertRaises(FileExistsError):
            run_demo(SAMPLES, output)
        self.assertEqual(before, tree_hashes(output))

    def test_output_cannot_contain_input_or_be_nested_in_input(self):
        with self.assertRaisesRegex(ValueError, "must not contain"):
            run_demo(SAMPLES, SAMPLES / "new-run")
        with self.assertRaisesRegex(ValueError, "must not contain"):
            run_demo(SAMPLES, SAMPLES.parent)

    def test_changed_input_is_rejected_before_output_creation(self):
        with patch("mask_detection.demo.digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "input changed"):
                run_demo(SAMPLES, self.root / "not-created")
        self.assertFalse((self.root / "not-created").exists())

    def test_bad_probability_rows_leave_failure_receipt_not_success(self):
        fixture, expected, rows = read_fixture(SAMPLES)
        rows[0]["probabilities"] = [0.8, 0.8, 0.8]
        with patch("mask_detection.demo.read_fixture", return_value=(fixture, expected, rows)):
            with self.assertRaisesRegex(ValueError, "sum to one"):
                run_demo(SAMPLES, self.root / "bad-probabilities")
        error = json.loads((self.root / "bad-probabilities/failure.json").read_text())
        self.assertEqual("failed", error["status"])
        self.assertEqual("ValueError", error["type"])
        self.assertFalse((self.root / "bad-probabilities/report.html").exists())
        self.assertFalse((self.root / "bad-probabilities/summary.json").exists())

    def test_repeated_identity_cannot_silently_overwrite_prediction(self):
        fixture, expected, rows = read_fixture(SAMPLES)
        rows.append(dict(rows[0]))
        with patch("mask_detection.demo.read_fixture", return_value=(fixture, expected, rows)):
            with self.assertRaisesRegex(ValueError, "Repeated scripted"):
                run_demo(SAMPLES, self.root / "duplicate-prediction")
        self.assertTrue((self.root / "duplicate-prediction/failure.json").is_file())

    def test_incomplete_predictions_are_not_aligned_by_position(self):
        fixture, expected, rows = read_fixture(SAMPLES)
        rows.pop()
        with patch("mask_detection.demo.read_fixture", return_value=(fixture, expected, rows)):
            with self.assertRaisesRegex(ValueError, "cover exactly"):
                run_demo(SAMPLES, self.root / "missing-prediction")

    def test_shuffled_predictions_preserve_identity_alignment(self):
        fixture, expected, rows = read_fixture(SAMPLES)
        rows.reverse()
        with patch("mask_detection.demo.read_fixture", return_value=(fixture, expected, rows)):
            result = run_demo(SAMPLES, self.root / "reordered")
        self.assertEqual([[6, 3, 0], [0, 6, 3], [3, 0, 6]], result["confusion_matrix"])


if __name__ == "__main__":
    unittest.main()
