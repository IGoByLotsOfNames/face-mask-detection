"""Frozen runner contracts; all worker launches are mocked, never measured."""

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def load_runner(path):
    spec = importlib.util.spec_from_file_location("memory_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_runner(TOOLS / "benchmark_input_memory.py")
FAKE_RUNTIME = {
    "python": "fixture-python",
    "platform": "win32",
    "packages": {name: "fixture-version" for name in ("tensorflow", "keras", "numpy", "Pillow")},
}


class MemoryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TOOLS.parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.baseline = self.root / "fixture-baseline.py"
        self.baseline.write_text("# Test fixture; never execute this source.\n", encoding="utf-8")
        self.baseline_hash = runner.digest(self.baseline)
        self.experiment = self.root / "experiment"
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(runner, "BASELINE_SHA", self.baseline_hash))
        self.patches.enter_context(
            patch.object(runner, "runtime_record", return_value=FAKE_RUNTIME)
        )

    def freeze(self):
        plan = runner.prepare(self.baseline, self.experiment, smoke=True)
        frozen = load_runner(self.experiment / "benchmark_input_memory.py")
        self.patches.enter_context(patch.object(frozen, "BASELINE_SHA", self.baseline_hash))
        self.patches.enter_context(
            patch.object(frozen, "runtime_record", return_value=FAKE_RUNTIME)
        )
        self.patches.enter_context(
            patch.object(frozen, "available_ram_bytes", return_value=64 * 1024**3)
        )
        for name in ("platform", "machine", "processor"):
            self.patches.enter_context(
                patch.object(frozen.platform, name, return_value=f"fixture-{name}")
            )
        # The application refuses real measurement under instrumentation. These
        # tests launch no process; clear that guard only to exercise failure logic.
        self.patches.enter_context(patch.object(frozen.sys, "gettrace", return_value=None))
        clean = {key: value for key, value in os.environ.items() if not key.startswith("COVERAGE_")}
        self.patches.enter_context(patch.dict(os.environ, clean, clear=True))
        return plan, frozen

    def sole_run(self):
        results = list((self.experiment / "results").iterdir())
        self.assertEqual(1, len(results))
        return results[0]

    def assert_failed_without_summary(self, run_root, failure_type):
        failure = json.loads((run_root / "failure.json").read_text(encoding="utf-8"))
        self.assertEqual("failed", failure["status"])
        self.assertEqual(failure_type, failure["type"])
        self.assertFalse((run_root / "summary.json").exists())
        self.assertFalse((run_root / "SUMMARY.txt").exists())
        self.assertFalse((self.experiment / ".run.lock").exists())

    def test_full_schedule_has_unique_adjacent_pairs_and_balanced_orientation(self):
        cases, seeds = [256, 1024, 4096], [17, 29, 43, 71, 101]
        schedule = runner.make_schedule(cases, seeds)
        self.assertEqual(schedule, runner.make_schedule(cases, seeds))
        self.assertEqual(30, len(schedule))
        self.assertEqual(
            {
                (rows, seed, implementation)
                for rows in cases
                for seed in seeds
                for implementation in ("baseline", "indexed")
            },
            {(slot["rows"], slot["seed"], slot["implementation"]) for slot in schedule},
        )
        first = Counter()
        for index in range(0, len(schedule), 2):
            a, b = schedule[index : index + 2]
            self.assertEqual((a["rows"], a["seed"]), (b["rows"], b["seed"]))
            self.assertEqual({"baseline", "indexed"}, {a["implementation"], b["implementation"]})
            first[a["rows"], a["implementation"]] += 1
        for rows in cases:
            self.assertEqual([2, 3], sorted(first[rows, name] for name in ("baseline", "indexed")))
        self.assertEqual(
            [7, 8],
            sorted(sum(first[rows, name] for rows in cases) for name in ("baseline", "indexed")),
        )

    def test_prepare_freezes_sources_inputs_and_refuses_overwrite(self):
        baseline_before = self.baseline.read_bytes()
        plan, _ = self.freeze()
        self.assertEqual(plan, runner.validate_experiment(self.experiment))
        self.assertEqual("harness-smoke-not-a-benchmark-result", plan["purpose"])
        self.assertEqual(8, len(json.loads((self.experiment / "inputs.json").read_text())["rows"]))
        for relative, digest in plan["files"].items():
            self.assertEqual(digest, runner.digest(self.experiment / relative))
        self.assertEqual(
            runner.digest(self.experiment / "plan.json"),
            (self.experiment / "plan.sha256").read_text().strip(),
        )
        before = {
            path.relative_to(self.experiment): runner.digest(path)
            for path in self.experiment.rglob("*")
            if path.is_file()
        }
        with self.assertRaises(FileExistsError):
            runner.prepare(self.baseline, self.experiment, smoke=True)
        self.assertEqual(
            before,
            {
                path.relative_to(self.experiment): runner.digest(path)
                for path in self.experiment.rglob("*")
                if path.is_file()
            },
        )
        self.assertEqual(baseline_before, self.baseline.read_bytes())
        self.assertFalse((self.experiment / "results").exists())

    def test_plan_source_and_runtime_tampering_are_rejected(self):
        plan, _ = self.freeze()
        plan_path = self.experiment / "plan.json"
        original = plan_path.read_bytes()
        plan_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "seal"):
            runner.validate_experiment(self.experiment)
        plan_path.write_bytes(original)
        source = self.experiment / plan["sources"]["indexed"]["path"]
        source_before = source.read_bytes()
        source.write_bytes(source_before + b"\n# changed fixture\n")
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            runner.validate_experiment(self.experiment)
        source.write_bytes(source_before)
        with patch.object(
            runner, "runtime_record", return_value={**FAKE_RUNTIME, "python": "different"}
        ):
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                runner.validate_experiment(self.experiment)
        self.assertEqual(plan, runner.validate_experiment(self.experiment))

    def test_failed_worker_is_not_retried_or_summarized(self):
        _, frozen = self.freeze()

        def failed(command, **kwargs):
            kwargs["stdout"].write("fixture worker stdout\n")
            kwargs["stderr"].write("fixture failure detail\n")
            return subprocess.CompletedProcess(command, 7)

        with patch.object(frozen.subprocess, "run", side_effect=failed) as launch:
            with patch.object(frozen, "_load_stats") as stats:
                with self.assertRaisesRegex(RuntimeError, "Worker failed"):
                    frozen.run(self.experiment)
                self.assertEqual(1, launch.call_count)
                stats.assert_not_called()
        run_root = self.sole_run()
        self.assert_failed_without_summary(run_root, "RuntimeError")
        (command,) = run_root.glob("*.command.json")
        self.assertIn("--output", json.loads(command.read_text())["command"])
        (exit_receipt,) = run_root.glob("*.exit.json")
        self.assertEqual(7, json.loads(exit_receipt.read_text())["exit_code"])
        (stderr,) = run_root.glob("*.stderr.log")
        self.assertIn("fixture failure detail", stderr.read_text())
        self.assertTrue((run_root / "context.json").exists())
        self.assertTrue((run_root / "schedule.json").exists())

    def test_timed_out_worker_retains_command_and_failure_without_retry(self):
        _, frozen = self.freeze()

        def timed_out(command, **kwargs):
            kwargs["stderr"].write("fixture timeout detail\n")
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

        with patch.object(frozen.subprocess, "run", side_effect=timed_out) as launch:
            with self.assertRaises(subprocess.TimeoutExpired):
                frozen.run(self.experiment)
            self.assertEqual(1, launch.call_count)
        run_root = self.sole_run()
        self.assert_failed_without_summary(run_root, "TimeoutExpired")
        (command,) = run_root.glob("*.command.json")
        receipt = json.loads(command.read_text())
        self.assertIn("--plan", receipt["command"])
        self.assertIn(str(self.experiment / "plan.json"), receipt["command"])
        (stderr,) = run_root.glob("*.stderr.log")
        self.assertIn("fixture timeout detail", stderr.read_text())

    def test_worker_source_provenance_mismatch_stops_before_statistics(self):
        plan, frozen = self.freeze()

        def wrong_source(command, **kwargs):
            def value(flag):
                return command[command.index(flag) + 1]

            rows, seed = int(value("--rows")), int(value("--seed"))
            record = {
                "schema_version": 1,
                "implementation": value("--implementation"),
                "rows": rows,
                "seed": seed,
                "metric": "process_lifetime_peak_rss_bytes",
                "peak_rss_bytes": 1024 * 1024,
                "metric_method": "Windows GetProcessMemoryInfo PeakWorkingSetSize",
                "plan_sha256": runner.digest(self.experiment / "plan.json"),
                "source_sha256": "0" * 64,
                "inputs_sha256": plan["inputs_sha256"],
                "image_size": [128, 128],
                "batch_size": 32,
                "batches": (rows + 31) // 32,
                "processed_rows": rows,
                "membership_sha256": hashlib.sha256(
                    b"".join(i.to_bytes(4, "little") for i in range(rows))
                ).hexdigest(),
                "correctness": "all constant RGB pixels encode each ID exactly once; label=id%2",
                "runtime": FAKE_RUNTIME,
                "tensorflow_devices": ["CPU"],
                "threads": {"intra_op": 2, "inter_op": 1},
            }
            frozen.write_json(Path(value("--output")), record)
            return subprocess.CompletedProcess(command, 0)

        with patch.object(frozen.subprocess, "run", side_effect=wrong_source) as launch:
            with patch.object(frozen, "_load_stats") as stats:
                with self.assertRaisesRegex(ValueError, "source|provenance"):
                    frozen.run(self.experiment)
                self.assertEqual(1, launch.call_count)
                stats.assert_not_called()
        self.assert_failed_without_summary(self.sole_run(), "ValueError")

    def test_existing_run_lock_is_preserved_without_worker_launch(self):
        _, frozen = self.freeze()
        lock = self.experiment / ".run.lock"
        lock.write_text("fixture owner; do not remove\n")
        with patch.object(frozen.subprocess, "run") as launch:
            with self.assertRaises(FileExistsError):
                frozen.run(self.experiment)
            launch.assert_not_called()
        self.assertEqual("fixture owner; do not remove\n", lock.read_text())
        self.assertFalse((self.experiment / "results").exists())


if __name__ == "__main__":
    unittest.main()
