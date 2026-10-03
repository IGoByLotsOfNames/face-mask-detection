"""Hand-calculated categorical oracles and mocked inference/evaluation guards."""

import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from mask_detection.evaluate import evaluate_models
from mask_detection.inference import Predictor
from mask_detection.metrics import (
    DECISION_POLICY,
    binary_metrics,
    categorical_metrics,
    categorical_predictions,
    evaluation_report,
    validate_probability_matrix,
)
from mask_detection.webcam import run_camera

CLASSES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def sample_rows(labels):
    return [
        dict(
            source=f"{index}.png",
            sha256=f"hash-{index}",
            group=f"group-{index}",
            label=label,
            split="test",
        )
        for index, label in enumerate(labels)
    ]


def categorical_metadata():
    return dict(
        schema_version=2,
        class_names=CLASSES,
        image_size=[2, 2],
        output_mode="categorical_softmax",
        decision_policy=dict(DECISION_POLICY),
    )


class CategoricalMetricTests(unittest.TestCase):
    def test_hand_calculated_confusion_precision_recall_f1(self):
        probabilities = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.3, 0.6],
            [0.7, 0.1, 0.2],
        ]
        report = categorical_metrics([0, 0, 1, 1, 2, 2], probabilities)
        self.assertEqual([[1, 1, 0], [0, 1, 1], [1, 0, 1]], report["confusion_matrix"])
        self.assertEqual(0.5, report["accuracy"])
        self.assertEqual(0.5, report["balanced_accuracy"])
        self.assertEqual(0.5, report["macro_recall"])
        self.assertEqual(0.5, report["macro_f1"])
        for row in report["by_class"]:
            self.assertEqual(
                dict(precision=0.5, recall=0.5, f1=0.5, support=2, predicted_count=2), row
            )
        expected_loss = -sum(math.log(value) for value in [0.8, 0.1, 0.6, 0.2, 0.6, 0.2]) / 6
        self.assertAlmostEqual(expected_loss, report["log_loss"])
        # Explicit per-example sums over the three coordinates, then mean.
        self.assertAlmostEqual((0.06 + 1.46 + 0.24 + 1.14 + 0.26 + 1.14) / 6, report["brier_score"])
        json.dumps(report, allow_nan=False)

    def test_perfect_and_maximum_brier_oracles(self):
        perfect = categorical_metrics([0, 1, 2], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(1, perfect["accuracy"])
        self.assertEqual(0, perfect["log_loss"])
        self.assertEqual(0, perfect["brier_score"])
        wrong = categorical_metrics([0, 1, 2], [[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.assertEqual(2, wrong["brier_score"])
        self.assertAlmostEqual(-math.log(1e-15), wrong["log_loss"])

    def test_missing_true_class_has_strict_null_macros(self):
        report = categorical_metrics([0, 1], [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        self.assertIsNone(report["by_class"][2]["recall"])
        self.assertIsNone(report["by_class"][2]["f1"])
        self.assertEqual(0, report["by_class"][2]["precision"])
        self.assertIsNone(report["macro_recall"])
        self.assertIsNone(report["balanced_accuracy"])
        self.assertIsNone(report["macro_f1"])
        json.dumps(report, allow_nan=False)

    def test_unpredicted_supported_class_f1_zero(self):
        report = categorical_metrics([0, 1, 2], [[1, 0, 0]] * 3)
        self.assertIsNone(report["by_class"][1]["precision"])
        self.assertEqual(0, report["by_class"][1]["recall"])
        self.assertEqual(0, report["by_class"][1]["f1"])
        self.assertAlmostEqual(1 / 3, report["macro_recall"])
        self.assertAlmostEqual(1 / 6, report["macro_f1"])

    def test_argmax_ties_use_lowest_recorded_index(self):
        self.assertEqual(
            [0, 1, 0],
            categorical_predictions([[0.5, 0.5, 0], [0, 0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]], 3),
        )

    def test_invalid_matrices_rejected(self):
        invalid = [
            [],
            [0, 1, 0],
            [[0.5, 0.5]],
            [[1, 0, 0], [0.5, 0.5]],
            [[0.5, 0.5, 0.1]],
            [[0, 0, 0]],
            [[-0.1, 0.5, 0.6]],
            [[1.1, 0, -0.1]],
            [[float("nan"), 0, 1]],
            [[float("inf"), 0, 0]],
            [[None, 0, 1]],
            ["100"],
            "100",
        ]
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    validate_probability_matrix(probabilities, 3)

    def test_sum_tolerance_accepts_float32_without_renormalization(self):
        values = np.asarray([[0.1, 0.2, 0.7]], dtype=np.float32)
        output = validate_probability_matrix(values, 3)
        self.assertEqual(float(values[0, 0]), output[0][0])
        self.assertNotEqual(1.0, sum(output[0]))
        with self.assertRaises(ValueError):
            validate_probability_matrix([[0.1, 0.2, 0.70001]], 3)

    def test_invalid_labels_rejected(self):
        for labels in ([3], [-1], [True], [1.0], [], [0, 1]):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    categorical_metrics(labels, [[1, 0, 0]])

    def test_report_preserves_order_and_averages_vectors_before_decision(self):
        rows = sample_rows([2, 0, 1])
        report = evaluation_report(
            rows,
            {
                "first": [[0.2, 0.3, 0.5], [0.8, 0.1, 0.1], [0, 0.6, 0.4]],
                "second": [[0.2, 0.7, 0.1], [0.6, 0.2, 0.2], [0, 0.4, 0.6]],
            },
            CLASSES,
        )
        self.assertEqual(2, report["schema_version"])
        self.assertNotIn("positive_class", report)
        self.assertEqual(
            ["0.png", "1.png", "2.png"], [row["sample_id"] for row in report["samples"]]
        )
        np.testing.assert_allclose(
            [0.2, 0.5, 0.3], report["samples"][0]["probabilities"]["ensemble"]
        )
        self.assertEqual(1, report["samples"][0]["predicted_class_indices"]["ensemble"])
        self.assertEqual(1, report["samples"][2]["predicted_class_indices"]["ensemble"])
        self.assertEqual(DECISION_POLICY, report["decision_policy"])

    def test_report_rejects_threshold_and_bad_alignment(self):
        rows = sample_rows([0])
        for threshold in (0, 0.5, 1, float("nan")):
            with self.assertRaises(ValueError):
                evaluation_report(rows, {"a": [[1, 0, 0]]}, CLASSES, threshold=threshold)
        for predictions in ({}, {"ensemble": [[1, 0, 0]]}, {"a": [[1, 0, 0], [0, 1, 0]]}):
            with self.assertRaises(ValueError):
                evaluation_report(rows, predictions, CLASSES)
        with self.assertRaises(ValueError):
            evaluation_report(rows * 2, {"a": [[1, 0, 0], [0, 1, 0]]}, CLASSES)

    def test_legacy_binary_report_preserved(self):
        rows = sample_rows([0, 1])
        report = evaluation_report(rows, {"a": [0.1, 0.9]}, ["a", "b"])
        self.assertEqual(1, report["schema_version"])
        self.assertEqual("b", report["positive_class"])
        self.assertEqual(binary_metrics([0, 1], [0.1, 0.9]), report["metrics"]["a"])
        self.assertEqual(0.1, report["samples"][0]["probabilities"]["a"])
        self.assertEqual(
            0.7,
            evaluation_report(rows, {"a": [0.1, 0.9]}, ["a", "b"], threshold=0.7)["metrics"]["a"][
                "threshold"
            ],
        )


class CategoricalInferenceTests(unittest.TestCase):
    def predictor(self, output):
        model = Mock(
            return_value=np.asarray(output), input_shape=(None, 2, 2, 3), output_shape=(None, 3)
        )
        return Predictor(model, categorical_metadata())

    def test_inference_and_evaluation_share_tie_policy(self):
        probabilities = [[0.5, 0.5, 0]]
        result = self.predictor(probabilities).bgr(np.zeros((2, 2, 3), dtype=np.uint8))
        report = evaluation_report(sample_rows([0]), {"a": probabilities}, CLASSES)
        self.assertEqual(0, result["class_index"])
        self.assertEqual(CLASSES[0], result["class_name"])
        self.assertEqual(probabilities[0], result["probabilities"])
        self.assertEqual(CLASSES, result["class_names"])
        self.assertEqual(0.5, result["probability_predicted_class"])
        self.assertEqual(report["decision_policy"], result["decision_policy"])
        self.assertNotIn("threshold", result)
        self.assertNotIn("probability_class_1", result)

    def test_four_class_contract_is_categorical_end_to_end(self):
        classes = CLASSES + ["other"]
        metadata = categorical_metadata()
        metadata["class_names"] = classes
        model = Mock(
            return_value=np.array([[0.1, 0.2, 0.3, 0.4]]),
            input_shape=(None, 2, 2, 3),
            output_shape=(None, 4),
        )
        result = Predictor(model, metadata).bgr(np.zeros((2, 2, 3), dtype=np.uint8))
        report = evaluation_report(sample_rows([3]), {"a": [result["probabilities"]]}, classes)
        self.assertEqual(3, result["class_index"])
        self.assertEqual(3, report["samples"][0]["predicted_class_indices"]["a"])
        self.assertEqual(4, len(report["metrics"]["a"]["by_class"]))

    def test_inference_rejects_wrong_shapes_and_invalid_probabilities(self):
        for output in (
            [[0.7]],
            [0.2, 0.3, 0.5],
            [[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]],
            [[0.3, 0.3, 0.3]],
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    self.predictor(output).bgr(np.zeros((2, 2, 3), dtype=np.uint8))

    def test_direct_predictor_rejects_mismatched_categorical_policy(self):
        model = Mock(input_shape=(None, 2, 2, 3), output_shape=(None, 3))
        for key, value in (
            ("schema_version", 1),
            ("output_mode", "binary_sigmoid"),
            ("decision_policy", {"kind": "threshold"}),
        ):
            metadata = categorical_metadata()
            metadata[key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    Predictor(model, metadata)

    def test_predictor_load_uses_bundle_resolved_model(self):
        model = Mock(
            return_value=np.array([[0, 0, 1]]), input_shape=(None, 2, 2, 3), output_shape=(None, 3)
        )
        loader = Mock(return_value=model)
        tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=loader)))
        resolved = Path("bundle/model.keras")
        with (
            patch.dict("sys.modules", tensorflow=tf),
            patch("mask_detection.inference.resolve_model_path", return_value=resolved) as resolver,
            patch("mask_detection.inference.load_metadata", return_value=categorical_metadata()),
        ):
            result = Predictor.load(Path("bundle")).bgr(np.zeros((2, 2, 3), dtype=np.uint8))
        resolver.assert_called_once_with(Path("bundle"))
        loader.assert_called_once_with(resolved, compile=False)
        self.assertEqual(CLASSES[2], result["class_name"])

    def test_three_classes_draw_distinct_colours_and_preserve_crops(self):
        camera, cv, cascade, predictor = Mock(), Mock(), Mock(), Mock()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame), (False, None)]
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = [(0, 0, 6, 6), (2, 2, 6, 6), (3, 3, 6, 6)]
        crops = []

        def predict(crop):
            index = len(crops)
            crops.append(crop.copy())
            return dict(
                class_index=index, class_name=CLASSES[index], probability_predicted_class=0.8
            )

        predictor.bgr.side_effect = predict

        def draw(image, start, end, colour, width):
            image[start[1] : end[1], start[0] : end[0]] = colour

        cv.rectangle.side_effect = draw
        cv.waitKey.return_value = -1
        run_camera(predictor, cv=cv, capture=camera, cascade=cascade)
        self.assertEqual(3, len({call.args[3] for call in cv.rectangle.call_args_list}))
        self.assertEqual(3, len(crops))
        self.assertTrue(all(np.count_nonzero(crop) == 0 for crop in crops))
        for index, call in enumerate(cv.putText.call_args_list):
            self.assertIn(CLASSES[index], call.args[1])
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()


class EvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(
            class_names=CLASSES, rows=sample_rows([0, 1, 2]), source_fingerprint="source"
        )
        self.metadata = dict(
            categorical_metadata(),
            trained_split_hashes=["train-hash"],
            split_manifest_fingerprint="manifest",
            source_fingerprint="source",
            architecture="synthetic",
        )
        self.model = Mock(
            input_shape=(None, 2, 2, 3),
            output_shape=(None, 3),
            return_value=np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )
        self.loader = Mock(return_value=self.model)
        self.tf = SimpleNamespace(
            keras=SimpleNamespace(models=SimpleNamespace(load_model=self.loader))
        )
        self.patchers = [
            patch.dict("sys.modules", tensorflow=self.tf),
            patch("mask_detection.evaluate.validate_manifest", return_value=self.manifest),
            patch("mask_detection.evaluate.load_metadata", return_value=self.metadata),
            patch(
                "mask_detection.evaluate.resolve_model_path",
                return_value=Path("bundle/model.keras"),
            ),
            patch("mask_detection.evaluate.manifest_fingerprint", return_value="manifest"),
            patch("mask_detection.evaluate.digest", return_value="digest"),
            patch(
                "mask_detection.evaluate.dataset",
                return_value=[(np.zeros((3, 2, 2, 3)), np.arange(3))],
            ),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_matrix_is_not_flattened_and_bundle_identity_is_recorded(self):
        report = evaluate_models([Path("bundle")], Path("dataset"))
        self.assertEqual(1, report["metrics"]["model_1"]["accuracy"])
        self.assertEqual([0.0, 1.0, 0.0], report["samples"][1]["probabilities"]["model_1"])
        self.assertEqual("model.keras", report["models"]["model_1"]["file"])
        self.loader.assert_called_once_with(Path("bundle/model.keras"), compile=False)

    def test_overlap_fingerprint_and_class_order_guards_precede_loading(self):
        for key, bad_value in (
            ("trained_split_hashes", ["hash-0"]),
            ("split_manifest_fingerprint", "other"),
            ("source_fingerprint", "other"),
            ("class_names", list(reversed(CLASSES))),
        ):
            original = self.metadata[key]
            self.metadata[key] = bad_value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    evaluate_models([Path("bundle")], Path("dataset"))
            self.metadata[key] = original
        self.loader.assert_not_called()

    def test_explicit_categorical_threshold_is_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            evaluate_models([Path("bundle")], Path("dataset"), threshold=0.5)
        self.loader.assert_not_called()

    def test_bad_batch_shape_is_rejected(self):
        self.model.return_value = np.asarray([1, 0, 0])
        with self.assertRaises(ValueError):
            evaluate_models([Path("bundle")], Path("dataset"))


if __name__ == "__main__":
    unittest.main()
