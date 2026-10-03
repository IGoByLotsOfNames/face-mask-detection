"""Opt-in subprocess CLI contract using synthetic colours, never real face data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SOURCE = Path(__file__).resolve().parents[1] / "src"


def strict_json(text):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


def hashes(directory):
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.rglob("*")
        if path.is_file()
    }


@unittest.skipUnless(
    os.environ.get("RUN_ML_TESTS") == "1", "Set RUN_ML_TESTS=1 for TensorFlow CLI smoke tests"
)
class RuntimeCliTests(unittest.TestCase):
    def run_cli(self, module, *arguments, success=True, timeout=180):
        environment = os.environ.copy()
        environment.update(
            PYTHONPATH=str(SOURCE),
            PYTHONUTF8="1",
            PYTHONDONTWRITEBYTECODE="1",
            TF_NUM_INTRAOP_THREADS="2",
            TF_NUM_INTEROP_THREADS="1",
            TF_CPP_MIN_LOG_LEVEL="2",
            CUDA_VISIBLE_DEVICES="-1",
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", f"mask_detection.{module}", *map(str, arguments)],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            check=False,
        )
        evidence = f"{module} returned {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if success:
            self.assertEqual(0, result.returncode, evidence)
        else:
            self.assertNotEqual(0, result.returncode, evidence)
        return result

    def test_grouped_data_to_bundle_inference_and_saved_report_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            self.root = Path(temp)
            source = self.root / "source"
            group_rows = []
            for label, name in enumerate(("synthetic_a", "synthetic_b", "synthetic_c")):
                (source / name).mkdir(parents=True)
                for index in range(6):
                    path = source / name / f"{index}.png"
                    Image.new("RGB", (12, 10), (label * 90, index * 30, 40)).save(path)
                    group_rows.append((path.relative_to(source).as_posix(), f"photo-{index}"))
            group_file = self.root / "groups.csv"
            with group_file.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("path", "group"))
                writer.writerows(group_rows)

            split = self.root / "split"
            prepared = strict_json(
                self.run_cli("data", source, split, "--groups", group_file).stdout
            )
            manifest = strict_json((split / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_fingerprint"], prepared["source_fingerprint"])
            self.assertEqual(18, sum(prepared["counts"].values()))
            before_split = hashes(split)
            rejected = self.run_cli("data", source, split, "--groups", group_file, success=False)
            self.assertIn("FileExistsError", rejected.stderr)
            self.assertEqual(before_split, hashes(split))

            # Omitting --output exercises the new canonical bundle default from
            # an unrelated working directory, not a source-tree import accident.
            self.run_cli("train", split, "--epochs", "1", "--batch-size", "4", "--seed", "17")
            bundle = self.root / "artifacts" / "run"
            self.assertEqual(
                {"model.keras", "model.metadata.json", "model.history.json", "bundle.json"},
                {path.name for path in bundle.iterdir()},
            )
            sealed = strict_json((bundle / "bundle.json").read_text(encoding="utf-8"))
            for filename, expected in sealed["files"].items():
                self.assertEqual(
                    expected, hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
                )
            before_bundle = hashes(bundle)
            rejected = self.run_cli("train", split, "--epochs", "1", success=False)
            self.assertIn("FileExistsError", rejected.stderr)
            self.assertEqual(before_bundle, hashes(bundle))

            row = next(row for row in manifest["rows"] if row["split"] == "test")
            prediction = strict_json(self.run_cli("inference", bundle, split / row["path"]).stdout)
            self.assertEqual(manifest["class_names"], prediction["class_names"])
            self.assertEqual(
                manifest["class_names"][prediction["class_index"]], prediction["class_name"]
            )
            self.assertEqual(3, len(prediction["probabilities"]))

            report_path = self.root / "report.json"
            evaluated = self.run_cli(
                "evaluate", bundle, split, "--batch-size", "2", "--output", report_path
            )
            printed_metrics = strict_json(evaluated.stdout)
            report = strict_json(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["metrics"], printed_metrics)
            self.assertEqual(manifest["class_names"], report["class_names"])
            self.assertEqual(
                sum(row["split"] == "test" for row in manifest["rows"]), len(report["samples"])
            )
            matching = next(
                sample for sample in report["samples"] if sample["sample_id"] == row["source"]
            )
            self.assertEqual(
                prediction["class_index"], matching["predicted_class_indices"]["model_1"]
            )
            self.assertEqual(
                hashlib.sha256((bundle / "model.keras").read_bytes()).hexdigest(),
                report["models"]["model_1"]["sha256"],
            )
            before_report = report_path.read_bytes()
            rejected = self.run_cli(
                "evaluate", bundle, split, "--output", report_path, success=False
            )
            self.assertIn("FileExistsError", rejected.stderr)
            self.assertEqual(before_report, report_path.read_bytes())

            invalid_report = self.root / "invalid-threshold.json"
            rejected = self.run_cli(
                "evaluate",
                bundle,
                split,
                "--threshold",
                ".5",
                "--output",
                invalid_report,
                success=False,
            )
            self.assertIn("argmax", rejected.stderr)
            self.assertFalse(invalid_report.exists())
            self.assertEqual(before_bundle, hashes(bundle))


if __name__ == "__main__":
    unittest.main()
