"""Opt-in real TensorFlow integration using generated colours, never face images."""

import gc
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from mask_detection.data import create_split, dataset


@unittest.skipUnless(
    os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for TensorFlow smoke tests"
)
class MulticlassRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tensorflow as tf

        # Other runtime suites may already have initialized this same process.
        try:
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(2)
        except RuntimeError:
            if (
                tf.config.threading.get_inter_op_parallelism_threads() != 1
                or tf.config.threading.get_intra_op_parallelism_threads() != 2
            ):
                raise
        cls.tf = tf

    def tearDown(self):
        self.tf.keras.backend.clear_session()
        gc.collect()

    def test_three_class_bundle_training_reload_and_evaluation(self):
        from mask_detection.artifacts import load_metadata, resolve_model_path
        from mask_detection.evaluate import evaluate_models
        from mask_detection.inference import Predictor
        from mask_detection.train import train_bundle

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            groups = {}
            for label, name in enumerate(("synthetic_a", "synthetic_b", "synthetic_c")):
                (source / name).mkdir(parents=True)
                for index in range(9):
                    path = source / name / f"{index}.png"
                    Image.new("RGB", (16, 16), (label * 90, index * 20, 40)).save(path)
                    groups[path.relative_to(source).as_posix()] = f"photo-{index}"
            manifest = create_split(source, root / "split", groups=groups)
            train_bundle(root / "split", root / "bundle", epochs=1, batch_size=4)
            metadata = load_metadata(root / "bundle")
            self.assertEqual(2, metadata["schema_version"])
            self.assertEqual(manifest["class_names"], metadata["class_names"])
            self.assertEqual("categorical_softmax", metadata["output_mode"])
            self.assertTrue(resolve_model_path(root / "bundle").is_file())
            predictor = Predictor.load(root / "bundle")
            report = evaluate_models([root / "bundle"], root / "split", batch_size=2)
            self.assertEqual(3, len(report["metrics"]["model_1"]["by_class"]))
            by_source = {row["source"]: row for row in manifest["rows"]}
            for sample in report["samples"]:
                row = by_source[sample["sample_id"]]
                actual = predictor.image(root / "split" / row["path"])
                self.assertEqual(
                    sample["predicted_class_indices"]["model_1"], actual["class_index"]
                )
                np.testing.assert_allclose(
                    sample["probabilities"]["model_1"], actual["probabilities"], atol=1e-6
                )
                self.assertEqual(report["decision_policy"], actual["decision_policy"])
            with self.assertRaises(FileExistsError):
                train_bundle(root / "split", root / "bundle", epochs=1)
            with self.assertRaises(ValueError):
                evaluate_models([root / "bundle"], root / "split", threshold=0.5)

            # Rebuild with the same seed and compare complete sample orders.
            orders = []
            for _ in range(2):
                self.tf.keras.utils.set_random_seed(17)
                ds = dataset(
                    root / "split",
                    manifest,
                    "train",
                    (8, 8),
                    4,
                    shuffle=True,
                    seed=17,
                    categorical=True,
                )
                order = []
                for images, labels in ds:
                    self.assertEqual(self.tf.int32, labels.dtype)
                    self.assertEqual(1, len(labels.shape))
                    order.extend(
                        (tuple(map(int, image[0, 0])), int(label))
                        for image, label in zip(images.numpy(), labels.numpy())
                    )
                orders.append(order)
            self.assertEqual(orders[0], orders[1])
            self.assertEqual(
                sum(row["split"] == "train" for row in manifest["rows"]), len(orders[0])
            )


if __name__ == "__main__":
    unittest.main()
