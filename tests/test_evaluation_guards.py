"""Evaluation boundaries with real synthetic split/bundle records and fake models.

The .keras files here contain marker bytes, not serialized neural networks. Only
TensorFlow loading and its dataset batches are mocked; provenance checks use the
actual metadata, bundle hashes and validated on-disk split.
"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from mask_detection.artifacts import save_metadata, write_bundle_manifest
from mask_detection.data import create_split, digest, load_rgb
from mask_detection.evaluate import evaluate_models
from mask_detection.inference import Predictor
from mask_detection.metrics import categorical_metrics


class BundleEvaluationBoundaryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        source = self.root / "source"
        for label, name in enumerate(("incorrect", "mask", "no_mask")):
            directory = source / name
            directory.mkdir(parents=True)
            for index in range(4):
                Image.new("RGB", (4, 4), (40 + label * 60, 10 + index * 20, 15)).save(
                    directory / f"{index}.png"
                )
        self.split = self.root / "split"
        self.manifest = create_split(source, self.split, independent_images=True)
        self.rows = [row for row in self.manifest["rows"] if row["split"] == "test"]
        self.bundle = self.make_bundle("bundle")
        self.model = Mock(input_shape=(None, 2, 2, 3), output_shape=(None, 3))
        self.model.side_effect = self.classify_synthetic_batch
        self.loader = Mock(return_value=self.model)
        tensorflow = SimpleNamespace(
            keras=SimpleNamespace(models=SimpleNamespace(load_model=self.loader))
        )
        self.tf_patch = patch.dict("sys.modules", tensorflow=tensorflow)
        self.tf_patch.start()
        self.addCleanup(self.tf_patch.stop)
        self.data_patch = patch(
            "mask_detection.evaluate.dataset", side_effect=self.make_test_batches
        )
        self.dataset = self.data_patch.start()
        self.addCleanup(self.data_patch.stop)

    def make_bundle(self, name, manifest=None):
        bundle = self.root / name
        bundle.mkdir()
        model = bundle / "model.keras"
        model.write_bytes(b"synthetic marker bytes; TensorFlow loader is mocked")
        (bundle / "model.history.json").write_text('{"loss": [1.0]}', encoding="utf-8")
        save_metadata(
            model,
            self.manifest if manifest is None else manifest,
            size=(2, 2),
            architecture="synthetic-boundary-fixture",
            seed=42,
        )
        write_bundle_manifest(bundle)
        return bundle

    @staticmethod
    def classify_synthetic_batch(images, *, training):
        if training is not False:
            raise AssertionError("Evaluation must disable training mode")
        labels = [int((image[0, 0, 0] - 40) // 60) for image in images]
        return np.eye(3)[labels]

    def make_test_batches(self, root, manifest, split, size, batch_size, *, categorical):
        self.assertEqual(self.split, root)
        self.assertEqual("test", split)
        self.assertTrue(categorical)
        self.assertEqual((2, 2), size)
        rows = [row for row in manifest["rows"] if row["split"] == split]
        for start in range(0, len(rows), batch_size):
            selected = rows[start : start + batch_size]
            yield (
                np.asarray([load_rgb(root / row["path"], size) for row in selected]),
                np.asarray([row["label"] for row in selected]),
            )

    def test_only_held_out_rows_reported_in_manifest_order_across_batches(self):
        report = evaluate_models([self.bundle], self.split, batch_size=2)
        self.dataset.assert_called_once_with(
            self.split, self.manifest, "test", (2, 2), 2, categorical=True
        )
        self.assertEqual(
            [row["source"] for row in self.rows], [row["sample_id"] for row in report["samples"]]
        )
        excluded = {row["source"] for row in self.manifest["rows"] if row["split"] != "test"}
        self.assertFalse(excluded & {row["sample_id"] for row in report["samples"]})
        self.assertEqual(math.ceil(len(self.rows) / 2), self.model.call_count)
        for expected, actual in zip(self.rows, report["samples"]):
            self.assertEqual(expected["label"], actual["predicted_class_indices"]["model_1"])
            self.assertEqual(expected["sha256"], actual["sha256"])
        self.assertEqual(digest(self.bundle / "model.keras"), report["models"]["model_1"]["sha256"])
        self.assertEqual(digest(self.split / "manifest.json"), report["manifest_sha256"])

    def test_valid_bundles_with_incompatible_class_source_or_split_are_refused(self):
        cases = (
            (
                "classes",
                "class order",
                lambda m: m.update(class_names=list(reversed(m["class_names"]))),
            ),
            (
                "source",
                "source fingerprint",
                lambda m: m.update(source_fingerprint="different-source"),
            ),
            ("split", "recorded dataset manifest", lambda m: m.update(seed=m["seed"] + 1)),
        )
        for name, reason, change in cases:
            manifest = copy.deepcopy(self.manifest)
            change(manifest)
            bundle = self.make_bundle(f"incompatible-{name}", manifest)
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, reason):
                    evaluate_models([bundle], self.split)
        self.loader.assert_not_called()
        self.dataset.assert_not_called()

    def test_recorded_test_overlap_refused_even_with_valid_bundle_hashes(self):
        metadata_path = self.bundle / "model.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["trained_split_hashes"].append(self.rows[0]["sha256"])
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        (self.bundle / "bundle.json").unlink()
        write_bundle_manifest(self.bundle)
        with self.assertRaisesRegex(ValueError, "overlaps model training/validation"):
            evaluate_models([self.bundle], self.split)
        self.loader.assert_not_called()
        self.dataset.assert_not_called()

    def test_changed_bundle_model_refused_by_inference_and_evaluation_before_loading(self):
        (self.bundle / "model.keras").write_bytes(b"changed marker bytes")
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            Predictor.load(self.bundle)
        with self.assertRaisesRegex(ValueError, "recorded hash"):
            evaluate_models([self.bundle], self.split)
        self.loader.assert_not_called()
        self.dataset.assert_not_called()

    def test_wrong_batch_cardinality_or_output_width_is_refused(self):
        for output in (np.ones((1, 3)) / 3, np.ones((2, 1)), np.ones((2, 4)) / 4):
            self.model.side_effect = None
            self.model.return_value = output
            with self.subTest(shape=output.shape):
                with self.assertRaisesRegex(ValueError, "batch shape"):
                    evaluate_models([self.bundle], self.split, batch_size=2)

    def test_invalid_probabilities_rejected_through_evaluation_and_inference(self):
        invalid = (
            [float("nan"), 0.5, 0.5],
            [float("inf"), 0, 0],
            [-0.1, 0.5, 0.6],
            [0.2, 0.2, 0.2],
        )
        for row in invalid:
            self.model.side_effect = lambda images, training, row=row: np.tile(
                row, (len(images), 1)
            )
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    evaluate_models([self.bundle], self.split, batch_size=2)
                with self.assertRaises(ValueError):
                    Predictor.load(self.bundle).predict_rgb(np.zeros((2, 2, 3), dtype=np.float32))

    def test_ensemble_vectors_keep_sample_alignment_across_models_and_batches(self):
        second_bundle = self.make_bundle("second")
        second = Mock(input_shape=(None, 2, 2, 3), output_shape=(None, 3))
        second.side_effect = lambda images, training: np.full((len(images), 3), 1 / 3)
        self.loader.side_effect = [self.model, second]
        report = evaluate_models([self.bundle, second_bundle], self.split, batch_size=2)
        self.assertEqual(2, self.dataset.call_count)
        for row in report["samples"]:
            probabilities = row["probabilities"]
            expected = (np.asarray(probabilities["model_1"]) + probabilities["model_2"]) / 2
            np.testing.assert_allclose(expected, probabilities["ensemble"])
            self.assertEqual(row["true_label"], row["predicted_class_indices"]["ensemble"])


class IndependentCategoricalOracleTests(unittest.TestCase):
    def test_probability_scores_match_rational_and_log_oracles_under_permutations(self):
        labels = [0, 2, 1, 0, 2]
        rational = [
            [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)],
            [Fraction(1, 5), Fraction(1, 5), Fraction(3, 5)],
            [Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)],
            [Fraction(1, 8), Fraction(3, 8), Fraction(1, 2)],
            [Fraction(1, 10), Fraction(2, 5), Fraction(1, 2)],
        ]
        probabilities = [[float(value) for value in row] for row in rational]
        # E[||p||^2 - 2*p(true) + 1], evaluated in exact rational arithmetic.
        brier = sum(
            sum(value * value for value in row) - 2 * row[label] + 1
            for row, label in zip(rational, labels)
        ) / len(labels)
        likelihood = math.prod(float(row[label]) for row, label in zip(rational, labels))
        log_loss = -math.log(likelihood) / len(labels)
        report = categorical_metrics(labels, probabilities)
        self.assertAlmostEqual(float(brier), report["brier_score"], places=14)
        self.assertAlmostEqual(log_loss, report["log_loss"], places=14)
        reversed_report = categorical_metrics(list(reversed(labels)), list(reversed(probabilities)))
        for metric in ("accuracy", "macro_recall", "macro_f1", "brier_score", "log_loss"):
            self.assertAlmostEqual(report[metric], reversed_report[metric], places=14)
        # Renaming classes preserves probability scores even though an exact
        # argmax tie can select a different semantic label after reordering.
        permutation = [2, 0, 1]
        renamed_labels = [permutation.index(label) for label in labels]
        renamed_probabilities = [[row[index] for index in permutation] for row in probabilities]
        renamed = categorical_metrics(renamed_labels, renamed_probabilities)
        self.assertAlmostEqual(report["brier_score"], renamed["brier_score"], places=14)
        self.assertAlmostEqual(report["log_loss"], renamed["log_loss"], places=14)


if __name__ == "__main__":
    unittest.main()
