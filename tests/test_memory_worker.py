"""Worker input and OS-counter contracts; no TensorFlow training or measurement run."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

WORKER_PATH = Path(__file__).resolve().parents[1] / "tools" / "input_memory_worker.py"
SPEC = importlib.util.spec_from_file_location("input_memory_worker_under_test", WORKER_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class MemoryWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "results").mkdir()
        self.runtime = {"python": "fixture", "platform": "fixture", "packages": {}}
        self.runtime_patch = patch.object(worker, "runtime_identity", return_value=self.runtime)
        self.runtime_patch.start()
        self.addCleanup(self.runtime_patch.stop)
        self.source = self.root / "data.py"
        self.source.write_text("# synthetic source fixture", encoding="utf-8")
        images = []
        for identity in range(2):
            path = self.root / f"{identity}.png"
            path.write_bytes(f"preflight hash fixture {identity}".encode())
            images.append(
                {
                    "id": identity,
                    "path": path.name,
                    "label": identity % 2,
                    "sha256": worker.digest(path),
                }
            )
        self.inputs = {
            "schema_version": 1,
            "class_names": ["synthetic_a", "synthetic_b"],
            "rows": images,
        }
        self.inputs_path = self.root / "inputs.json"
        self.inputs_path.write_text(json.dumps(self.inputs), encoding="utf-8")
        record = {"path": "data.py", "sha256": worker.digest(self.source)}
        self.plan = {
            "schema_version": 1,
            "cases": [2],
            "seeds": [17],
            "image_size": [128, 128],
            "batch_size": 32,
            "runtime": self.runtime,
            "sources": {"baseline": record, "indexed": record},
            "inputs_path": "inputs.json",
            "inputs_sha256": worker.digest(self.inputs_path),
        }
        self.plan_path = self.root / "plan.json"
        self.output = self.root / "results" / "worker.json"
        self.save_plan()

    def save_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def request(self, **options):
        arguments = dict(
            plan_path=self.plan_path, implementation="indexed", rows=2, seed=17, output=self.output
        )
        arguments.update(options)
        return worker.validate_request(**arguments)

    def test_preflight_accepts_declared_request_and_binds_bytes(self):
        request = self.request()
        self.assertEqual(worker.digest(self.plan_path), request["plan_sha256"])
        self.assertEqual(worker.digest(self.source), request["source_sha256"])
        self.assertEqual(2, len(request["selected"]))
        self.assertFalse(self.output.exists())

    def test_unknown_case_seed_implementation_and_bool_rejected(self):
        for options in (
            {"rows": 3},
            {"rows": True},
            {"seed": 1},
            {"seed": True},
            {"implementation": "other"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.request(**options)

    def test_runtime_shape_batch_and_invalid_plan_lists_rejected(self):
        for field, value in (
            ("runtime", {}),
            ("image_size", [64, 64]),
            ("batch_size", 1),
            ("cases", [True]),
            ("seeds", [[17]]),
            ("cases", [2, 2]),
            ("seeds", [-1]),
            ("seeds", [2**32]),
        ):
            before = self.plan[field]
            self.plan[field] = value
            self.save_plan()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.request()
            self.plan[field] = before
        self.save_plan()

    def test_source_manifest_and_image_tampering_rejected(self):
        for path in (self.source, self.inputs_path, self.root / "0.png"):
            before = path.read_bytes()
            path.write_bytes(b"changed")
            with (
                self.subTest(path=path.name),
                self.assertRaises((ValueError, json.JSONDecodeError)),
            ):
                self.request()
            path.write_bytes(before)

    def test_ids_labels_and_repeated_image_paths_rejected(self):
        for field, value in (("id", 0), ("label", 0), ("path", "0.png")):
            inputs = json.loads(json.dumps(self.inputs))
            inputs["rows"][1][field] = value
            self.inputs_path.write_text(json.dumps(inputs), encoding="utf-8")
            self.plan["inputs_sha256"] = worker.digest(self.inputs_path)
            self.save_plan()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.request()

    def test_containment_and_existing_output_rejected(self):
        for path in ("../outside", "/absolute", "C:/absolute", "a\\b", "a//b", "./file"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                worker.contained_path(self.root, path)
        with self.assertRaises(ValueError):
            self.request(output=self.root / "outside-results.json")
        with self.assertRaises(FileExistsError):
            self.request(output=self.root / "results" / "missing-parent" / "worker.json")
        self.output.write_text("prior receipt", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.request()
        self.assertEqual("prior receipt", self.output.read_text(encoding="utf-8"))

    def test_resource_units_and_invalid_counter_values(self):
        self.assertEqual(123 * 1024, worker.normalize_resource_peak(123, "Linux"))
        self.assertEqual(123, worker.normalize_resource_peak(123, "Darwin"))
        for value in (0, -1, True, 1.2, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                worker.normalize_resource_peak(value, "Linux")
        with self.assertRaises(RuntimeError):
            worker.normalize_resource_peak(123, "Unknown")

    def test_os_metric_selection_and_errors(self):
        with patch.object(worker.platform, "system", return_value="Windows"):
            with patch.object(worker, "_windows_peak_bytes", return_value=123456):
                self.assertEqual((123456, worker.WINDOWS_METHOD), worker.peak_memory())
            with patch.object(
                worker, "_windows_peak_bytes", side_effect=OSError("OS counter failed")
            ):
                with self.assertRaisesRegex(OSError, "OS counter failed"):
                    worker.peak_memory()
            with patch.object(worker, "_windows_peak_bytes", return_value=0):
                with self.assertRaises(ValueError):
                    worker.peak_memory()
        with patch.object(worker.platform, "system", return_value="Unsupported"):
            with self.assertRaises(RuntimeError):
                worker.peak_memory()

    @unittest.skipUnless(platform.system() == "Windows", "Windows native counter smoke check")
    def test_native_windows_counter_returns_positive_bytes(self):
        # Counter API compatibility only: this test is not a benchmark receipt.
        self.assertGreater(worker._windows_peak_bytes(), 0)

    def test_worker_import_is_standard_library_only(self):
        code = """
import importlib.abc
import importlib.util
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'numpy', 'PIL', 'cv2'}:
            raise AssertionError('Worker eagerly imported ' + fullname)
sys.meta_path.insert(0, Guard())
spec = importlib.util.spec_from_file_location('worker', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('stdlib import passed')
"""
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code, str(WORKER_PATH)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertEqual("stdlib import passed", result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
