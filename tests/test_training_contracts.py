"""Publication and metadata regressions use synthetic bytes, not trained-model evidence."""

from __future__ import annotations

import builtins
import copy
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mask_detection.artifacts import (
    BUNDLE_FILES,
    load_metadata,
    metadata_path,
    resolve_model_path,
    save_metadata,
    validate_classes,
    validate_model,
)
from mask_detection.model import build_model, validate_model_config
from mask_detection.train import train_bundle, train_model


def manifest(classes=None):
    return {
        "schema_version": 2,
        "class_names": classes or ["mask", "no_mask", "incorrect_mask"],
        "source_fingerprint": "fixture-source",
        "rows": [
            {"split": "train", "sha256": "train-hash"},
            {"split": "validation", "sha256": "validation-hash"},
            {"split": "test", "sha256": "test-hash"},
        ],
    }


def stage_fixture(root, model_path, **options):
    model_path.write_bytes(b"synthetic-model-bytes")
    model_path.with_suffix(".history.json").write_text('{"loss": [1.0]}', encoding="utf-8")
    save_metadata(model_path, manifest(), size=(46, 46), architecture="fixture", seed=42)
    return "synthetic-fitted-model"


class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fit = patch("mask_detection.train._fit_and_save", side_effect=stage_fixture)
        self.mock_fit = self.fit.start()
        self.addCleanup(self.fit.stop)

    def test_invalid_options_rejected_before_data_or_fit(self):
        bad = [
            dict(architecture="mobilenet"),
            dict(pretrained=True),
            dict(pretrained=0),
            dict(epochs=0),
            dict(epochs=True),
            dict(epochs=1.5),
            dict(batch_size=-1),
            dict(seed=-1),
            dict(seed=True),
            dict(size=(45, 128)),
            dict(size=(128, 45)),
            dict(size=(128,)),
            dict(size=(128.0, 128)),
        ]
        for trainer in (train_bundle, train_model):
            for options in bad:
                with self.subTest(trainer=trainer.__name__, options=options):
                    with self.assertRaises(ValueError):
                        trainer(self.root / "absent", self.root / "model.keras", **options)
        self.mock_fit.assert_not_called()
        self.assertEqual([], list(self.root.iterdir()))

    def test_invalid_model_config_before_tensorflow_import(self):
        invalid = [
            ((45, 128, 3), 3),
            ((128, 128, 1), 3),
            ((128, 128, 3), 1),
            ((128, 128, 3), True),
            ((128, 128, 3), 3.0),
            ((128, 128), 3),
        ]
        real_import = builtins.__import__

        def guard_import(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise AssertionError("Invalid configuration must not import TensorFlow")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard_import):
            for shape, count in invalid:
                with self.assertRaises(ValueError):
                    build_model(shape, count)
        validate_model_config((46, 46, 3), 3)
        validate_model_config((46, 128, 3), 4)

    def test_class_list_validation(self):
        for classes in (
            "abc",
            ("a", "b"),
            [],
            ["a"],
            ["a", "a"],
            ["a", ""],
            ["a", "  "],
            ["a", 1],
            ["a", ["b"]],
        ):
            with self.assertRaises(ValueError):
                validate_classes(classes)

    def test_schema_one_preserved_and_schema_two_binds_class_order(self):
        path = self.root / "fixture.keras"
        path.write_bytes(b"fixture")
        binary = save_metadata(
            path, manifest(["a", "b"]), size=(10, 20), architecture="fixture", seed=42
        )
        self.assertEqual(1, binary["schema_version"])
        self.assertNotIn("output_mode", binary)
        self.assertEqual(binary, load_metadata(path))
        three = save_metadata(path, manifest(), size=(46, 46), architecture="fixture", seed=42)
        self.assertEqual(2, three["schema_version"])
        self.assertEqual("categorical_softmax", three["output_mode"])
        self.assertEqual(
            {"kind": "argmax", "tie_break": "lowest_class_index"}, three["decision_policy"]
        )
        self.assertEqual(three, load_metadata(path))
        self.assertEqual(["train-hash", "validation-hash"], three["trained_split_hashes"])
        for changed in (
            dict(class_names=list(reversed(three["class_names"]))),
            dict(decision_policy={"kind": "argmax", "tie_break": "random"}),
            dict(output_mode="independent_sigmoids"),
            dict(schema_version=1),
        ):
            broken = copy.deepcopy(three)
            broken.update(changed)
            metadata_path(path).write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metadata(path)

    def test_four_classes_use_same_categorical_contract(self):
        path = self.root / "fixture.keras"
        path.write_bytes(b"fixture")
        metadata = save_metadata(
            path, manifest(["a", "b", "c", "d"]), size=(46, 46), architecture="fixture", seed=42
        )
        self.assertEqual(metadata, load_metadata(path))
        validate_model(
            SimpleNamespace(input_shape=(None, 46, 46, 3), output_shape=(None, 4)), metadata
        )
        with self.assertRaises(ValueError):
            validate_model(
                SimpleNamespace(input_shape=(None, 46, 46, 3), output_shape=(None, 1)), metadata
            )

    def test_bundle_publishes_complete_directory_and_rejects_overwrite(self):
        destination = self.root / "run"
        with patch("mask_detection.train.os.rename", wraps=os.rename) as rename:
            self.assertEqual(
                "synthetic-fitted-model", train_bundle(self.root / "data", destination)
            )
        rename.assert_called_once()
        self.assertEqual(
            set(BUNDLE_FILES) | {"bundle.json"}, {p.name for p in destination.iterdir()}
        )
        self.assertEqual(destination / "model.keras", resolve_model_path(destination))
        self.assertEqual(3, len(load_metadata(destination)["class_names"]))
        before = {p.name: p.read_bytes() for p in destination.iterdir()}
        with self.assertRaises(FileExistsError):
            train_bundle(self.root / "data", destination)
        self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})
        self.assertEqual([destination], list(self.root.iterdir()))

    def test_bundle_cleans_partial_staging_after_fit_failure(self):
        def fail(root, path, **options):
            path.write_bytes(b"partial")
            raise OSError("simulated save failure")

        self.mock_fit.side_effect = fail
        with self.assertRaisesRegex(OSError, "save failure"):
            train_bundle(self.root / "data", self.root / "run")
        self.assertEqual([], list(self.root.iterdir()))

    def test_bundle_cleans_staging_after_manifest_or_publish_failure(self):
        for target in ("write_bundle_manifest", "os.rename"):
            with self.subTest(target=target):
                with patch(
                    f"mask_detection.train.{target}", side_effect=OSError("simulated failure")
                ):
                    with self.assertRaises(OSError):
                        train_bundle(self.root / "data", self.root / "run")
                self.assertEqual([], list(self.root.iterdir()))

    def test_reservation_blocks_second_writer_and_preserves_lock(self):
        lock = self.root / ".run.training.lock"
        lock.write_text("another trainer", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            train_bundle(self.root / "data", self.root / "run")
        self.assertEqual("another trainer", lock.read_text(encoding="utf-8"))
        self.mock_fit.assert_not_called()

    def test_reservation_write_or_close_failure_cleans_only_owned_lock(self):
        actual_open = Path.open

        class FailingReservation:
            def __init__(self, stream, failure):
                self.stream, self.failure = stream, failure

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.stream.close()
                if self.failure == "close":
                    raise OSError("simulated reservation close failure")

            def write(self, value):
                if self.failure == "write":
                    raise OSError("simulated reservation write failure")
                return self.stream.write(value)

        for failure in ("write", "close"):
            with self.subTest(failure=failure):

                def failing_open(path, *args, **kwargs):
                    stream = actual_open(path, *args, **kwargs)
                    if path.name.endswith(".training.lock"):
                        return FailingReservation(stream, failure)
                    return stream

                with patch.object(Path, "open", failing_open):
                    with self.assertRaisesRegex(OSError, f"reservation {failure} failure"):
                        train_bundle(self.root / "data", self.root / "run")
                self.assertEqual([], list(self.root.iterdir()))
        self.mock_fit.assert_not_called()

        # Exclusive-open failure occurs before ownership, so its existing
        # reservation must survive even under the same injected write failure.
        lock = self.root / ".run.training.lock"
        lock.write_text("another trainer", encoding="utf-8")
        with patch.object(Path, "open", failing_open):
            with self.assertRaises(FileExistsError):
                train_bundle(self.root / "data", self.root / "run")
        self.assertEqual("another trainer", lock.read_text(encoding="utf-8"))
        self.mock_fit.assert_not_called()

    def test_bundle_detects_changes_missing_files_and_path_escape(self):
        destination = self.root / "run"
        train_bundle(self.root / "data", destination)
        for name in BUNDLE_FILES:
            path = destination / name
            before = path.read_bytes()
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                resolve_model_path(destination)
            path.unlink()
            with self.assertRaises(ValueError):
                resolve_model_path(destination)
            path.write_bytes(before)
        manifest_path = destination / "bundle.json"
        contract = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract["files"]["../outside.keras"] = contract["files"].pop("model.keras")
        manifest_path.write_text(json.dumps(contract), encoding="utf-8")
        with self.assertRaises(ValueError):
            resolve_model_path(destination)

    def test_legacy_metadata_last_and_ordinary_failure_rollback(self):
        path = self.root / "legacy.keras"
        links = []
        actual_link = os.link

        def tracked_link(source, destination):
            links.append(destination.name)
            if destination.suffixes[-2:] == [".metadata", ".json"]:
                raise OSError("simulated metadata publication failure")
            return actual_link(source, destination)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with patch("mask_detection.train.os.link", side_effect=tracked_link):
                with self.assertRaises(OSError):
                    train_model(self.root / "data", path)
        self.assertEqual(["legacy.keras", "legacy.history.json", "legacy.metadata.json"], links)
        self.assertEqual([], list(self.root.iterdir()))

    def test_legacy_success_and_existing_sidecar_preserved(self):
        path = self.root / "legacy.keras"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual("synthetic-fitted-model", train_model(self.root / "data", path))
        self.assertEqual(2, load_metadata(path)["schema_version"])
        metadata = metadata_path(path)
        original = metadata.read_bytes()
        path.unlink()
        path.with_suffix(".history.json").unlink()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(FileExistsError):
                train_model(self.root / "data", path)
        self.assertEqual(original, metadata.read_bytes())

    def test_bundle_existing_empty_directory_is_never_replaced(self):
        destination = self.root / "run"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            train_bundle(self.root / "data", destination)
        self.mock_fit.assert_not_called()
        self.assertTrue(destination.is_dir())


if __name__ == "__main__":
    unittest.main()
