"""Freeze and run a paired, fresh-process input-pipeline peak-memory comparison."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import distributions, version
from pathlib import Path

BASELINE_SHA = "270ca39a449477078cca3d2aa72a40cb7ecb8c1c914d1ad48eec5ddaf20da4cc"
BASELINE_COMMIT = "5f2e2ba9796b00255a3d93808f7ecdefdb6c5ecc"
SCRIPTS = ("benchmark_input_memory.py", "input_memory_worker.py", "input_memory_stats.py")
CASES = [256, 1024, 4096]
SEEDS = [17, 29, 43, 71, 101]
MIB = 2**20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def runtime_record():
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": {name: version(name) for name in ("tensorflow", "keras", "numpy", "Pillow")},
    }


def installed_packages():
    return dict(
        sorted(
            (item.metadata["Name"].lower().replace("_", "-"), item.version)
            for item in distributions()
        )
    )


def make_schedule(cases, seeds):
    pairs = []
    for case_index, rows in enumerate(cases):
        for seed_index, seed in enumerate(seeds):
            order = (
                ("baseline", "indexed")
                if (case_index + seed_index) % 2 == 0
                else ("indexed", "baseline")
            )
            pairs.append([{"rows": rows, "seed": seed, "implementation": item} for item in order])
    random.Random(1729).shuffle(pairs)
    return [item for pair in pairs for item in pair]


def contained(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Expected a nonempty relative POSIX path")
    path = root / relative
    if Path(relative).is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe experiment path")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Experiment path escapes its root")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Experiment links are not accepted")
    return path


def prepare(baseline, destination, *, smoke=False):
    """Copy immutable implementations and generate fixed tiny PNG inputs."""
    from PIL import Image

    baseline, destination = Path(baseline), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a new experiment directory")
    if digest(baseline) != BASELINE_SHA:
        raise ValueError("Baseline is not the pinned upstream data.py")
    code_root = Path(__file__).resolve().parent
    indexed = code_root.parent / "src/mask_detection/data.py"
    cases, seeds = ([8], [17, 29]) if smoke else (CASES, SEEDS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memory-prepare-", dir=destination.parent) as temp:
        stage = Path(temp)
        for name, source in (("baseline", baseline), ("indexed", indexed)):
            target = stage / "sources" / name / "data.py"
            target.parent.mkdir(parents=True)
            shutil.copy2(source, target)
        for name in SCRIPTS:
            shutil.copy2(code_root / name, stage / name)
        corpus = stage / "corpus/images"
        corpus.mkdir(parents=True)
        rows = []
        for index in range(max(cases)):
            relative = f"corpus/images/{index:06d}.png"
            colour = (index % 256, (index // 256) % 256, (index // 65536) % 256)
            with Image.new("RGB", (16, 16), colour) as image:
                image.save(stage / relative, format="PNG")
            rows.append(
                {
                    "id": index,
                    "path": relative,
                    "label": index % 2,
                    "sha256": digest(stage / relative),
                }
            )
        write_json(
            stage / "inputs.json",
            {"schema_version": 1, "class_names": ["synthetic_a", "synthetic_b"], "rows": rows},
        )
        files = {name: digest(stage / name) for name in SCRIPTS}
        files.update(
            {
                f"sources/{name}/data.py": digest(stage / "sources" / name / "data.py")
                for name in ("baseline", "indexed")
            }
        )
        files["inputs.json"] = digest(stage / "inputs.json")
        plan = {
            "schema_version": 1,
            "purpose": "harness-smoke-not-a-benchmark-result"
            if smoke
            else "input-pipeline-memory-v1",
            "metric": "process_lifetime_peak_rss_bytes",
            "measurement_scope": "Fresh process lifetime through identical one-epoch streaming checks; includes Python, TensorFlow, manifest/image verification and pipeline; excludes fixture generation and JSON serialization",
            "baseline_commit": BASELINE_COMMIT,
            "cases": cases,
            "seeds": seeds,
            "image_size": [128, 128],
            "batch_size": 32,
            "schedule_seed": 1729,
            "schedule": make_schedule(cases, seeds),
            "runtime": runtime_record(),
            "installed_packages": installed_packages(),
            "sources": {
                name: {
                    "path": f"sources/{name}/data.py",
                    "sha256": files[f"sources/{name}/data.py"],
                }
                for name in ("baseline", "indexed")
            },
            "inputs_path": "inputs.json",
            "inputs_sha256": files["inputs.json"],
            "files": files,
            "timeout_seconds_per_worker": 300,
            "minimum_available_ram_bytes": 3 * 1024**3 if not smoke else 0,
            "comparison_contract": "Both real dataset() functions use their shared binary label mode; this isolates a supported common workload, not three-class accuracy or CNN training memory",
            "env": {
                "CUDA_VISIBLE_DEVICES": "-1",
                "TF_ENABLE_ONEDNN_OPTS": "0",
                "TF_NUM_INTRAOP_THREADS": "2",
                "TF_NUM_INTEROP_THREADS": "1",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "PYTHONUTF8": "1",
            },
        }
        write_json(stage / "plan.json", plan)
        (stage / "plan.sha256").write_text(digest(stage / "plan.json") + "\n", encoding="ascii")
        if destination.exists():
            raise FileExistsError("Destination appeared during preparation")
        stage.rename(destination)
    return plan


def validate_experiment(root):
    root = Path(root).resolve()
    plan_file = root / "plan.json"
    if digest(plan_file) != (root / "plan.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Plan seal mismatch")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("metric") != "process_lifetime_peak_rss_bytes":
        raise ValueError("Unsupported measurement plan")
    smoke = plan.get("purpose") == "harness-smoke-not-a-benchmark-result"
    if plan.get("purpose") not in (
        "harness-smoke-not-a-benchmark-result",
        "input-pipeline-memory-v1",
    ):
        raise ValueError("Unknown plan purpose")
    cases, seeds = ([8], [17, 29]) if smoke else (CASES, SEEDS)
    if (
        plan.get("cases") != cases
        or plan.get("seeds") != seeds
        or plan.get("schedule") != make_schedule(cases, seeds)
    ):
        raise ValueError("Unexpected cases, seeds or frozen schedule")
    expected = {*SCRIPTS, "inputs.json", "sources/baseline/data.py", "sources/indexed/data.py"}
    if set(plan["files"]) != expected:
        raise ValueError("Incomplete frozen source registry")
    for relative, expected_hash in plan["files"].items():
        if digest(contained(root, relative)) != expected_hash:
            raise ValueError(f"Frozen file changed: {relative}")
    if plan["sources"]["baseline"]["sha256"] != BASELINE_SHA:
        raise ValueError("Unexpected baseline provenance")
    if runtime_record() != plan["runtime"]:
        raise ValueError("Python/platform/runtime versions differ from the frozen plan")
    if installed_packages() != plan["installed_packages"]:
        raise ValueError("Installed package inventory differs from the frozen plan")
    return plan


def available_ram_bytes():
    """Headroom guard, not a promise of a hard memory cap."""
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong)
                for name in (
                    "total_physical",
                    "available_physical",
                    "total_page",
                    "available_page",
                    "total_virtual",
                    "available_virtual",
                    "available_extended",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        function = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        function.argtypes = [ctypes.POINTER(MemoryStatus)]
        function.restype = ctypes.c_int
        if not function(ctypes.byref(status)):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(status.available_physical)
    if sys.platform.startswith("linux"):
        values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return int(values["MemAvailable"].split()[0]) * 1024
    raise ValueError("Runner headroom guard supports Windows/Linux only; do not pool platforms")


def _load_stats(root):
    spec = importlib.util.spec_from_file_location(
        "frozen_memory_stats", root / "input_memory_stats.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_worker_record(plan, plan_hash, slot, record):
    if (
        any(record.get(key) != value for key, value in slot.items())
        or record.get("plan_sha256") != plan_hash
    ):
        raise ValueError("Worker identity does not match its scheduled plan slot")
    expected_membership = hashlib.sha256(
        b"".join(index.to_bytes(4, "little") for index in range(slot["rows"]))
    ).hexdigest()
    required = {
        "runtime": plan["runtime"],
        "source_sha256": plan["sources"][slot["implementation"]]["sha256"],
        "inputs_sha256": plan["inputs_sha256"],
        "image_size": plan["image_size"],
        "batch_size": plan["batch_size"],
        "processed_rows": slot["rows"],
        "batches": (slot["rows"] + 31) // 32,
        "membership_sha256": expected_membership,
        "threads": {"intra_op": 2, "inter_op": 1},
        "correctness": "all constant RGB pixels encode each ID exactly once; label=id%2",
    }
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("Worker input, source, runtime or correctness provenance mismatch")
    if not record.get("tensorflow_devices") or any(
        device != "CPU" for device in record["tensorflow_devices"]
    ):
        raise ValueError("Worker did not report CPU-only TensorFlow execution")
    methods = {
        "win32": "Windows GetProcessMemoryInfo PeakWorkingSetSize",
        "linux": "resource.getrusage(RUSAGE_SELF).ru_maxrss; KiB converted to bytes",
    }
    if record.get("metric_method") != methods.get(plan["runtime"]["platform"]):
        raise ValueError("Worker memory API does not match the frozen platform")


def run(root, *, power_mode="not recorded", power_source="not recorded", background="not recorded"):
    root = Path(root).resolve()
    plan = validate_experiment(root)
    if digest(Path(__file__)) != plan["files"]["benchmark_input_memory.py"]:
        raise ValueError("Run the benchmark's frozen runner copy")
    plan_hash = digest(root / "plan.json")
    # Profilers change the workload. Never mix an instrumented run into results.
    if sys.gettrace() or any(key.startswith("COVERAGE_") for key in os.environ):
        raise ValueError("Run outside Coverage.py or a tracing debugger")
    lock = root / ".run.lock"
    stream = lock.open("x", encoding="utf-8")
    destination = None
    try:
        with stream:
            stream.write(
                f"Process {os.getpid()}; inspect an abandoned run before removing this lock.\n"
            )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        destination = root / "results" / run_id
        destination.mkdir(parents=True)
        context = {
            "run_id": run_id,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan_hash,
            "purpose": plan["purpose"],
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "runtime": runtime_record(),
            "installed_packages": installed_packages(),
            "power_mode_user_reported": power_mode,
            "power_source_user_reported": power_source,
            "background_user_reported": background,
            "environment": plan["env"],
        }
        write_json(destination / "context.json", context)
        write_json(destination / "schedule.json", plan["schedule"])
        records = []
        for index, slot in enumerate(plan["schedule"], 1):
            # Check all experiment source bytes before each independent process.
            validate_experiment(root)
            if digest(root / "plan.json") != plan_hash:
                raise ValueError("Plan changed during measurement")
            headroom = available_ram_bytes()
            if headroom < plan["minimum_available_ram_bytes"]:
                raise MemoryError(
                    "Insufficient free RAM for the declared workload; no remaining workers were launched"
                )
            prefix = f"{index:02d}-{slot['rows']}-{slot['seed']}-{slot['implementation']}"
            output = destination / (prefix + ".json")
            command = [
                sys.executable,
                "-B",
                str(root / "input_memory_worker.py"),
                "--plan",
                str(root / "plan.json"),
                "--implementation",
                slot["implementation"],
                "--rows",
                str(slot["rows"]),
                "--seed",
                str(slot["seed"]),
                "--output",
                str(output),
            ]
            environment = dict(os.environ, **plan["env"])
            environment.pop("PYTHONPATH", None)
            print(
                f"{index}/{len(plan['schedule'])}: rows={slot['rows']} seed={slot['seed']} {slot['implementation']}",
                flush=True,
            )
            stdout_path, stderr_path = (
                destination / (prefix + ".stdout.log"),
                destination / (prefix + ".stderr.log"),
            )
            write_json(
                destination / (prefix + ".command.json"),
                {
                    "command": command,
                    "available_ram_before_bytes": headroom,
                    "timeout_seconds": plan["timeout_seconds_per_worker"],
                },
            )
            with (
                stdout_path.open("x", encoding="utf-8") as stdout,
                stderr_path.open("x", encoding="utf-8") as stderr,
            ):
                child = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=plan["timeout_seconds_per_worker"],
                    check=False,
                )
            write_json(destination / (prefix + ".exit.json"), {"exit_code": child.returncode})
            if child.returncode != 0:
                raise RuntimeError(f"Worker failed; see {stderr_path}")
            record = json.loads(output.read_text(encoding="utf-8"))
            validate_worker_record(plan, plan_hash, slot, record)
            records.append(record)
        validate_experiment(root)
        if digest(root / "plan.json") != plan_hash:
            raise ValueError("Plan changed before summary")
        summary = _load_stats(root).summarize(plan, records)
        summary.update(purpose=plan["purpose"], status="complete", context=context)
        write_json(destination / "records.json", records)
        write_json(destination / "summary.json", summary)
        lines = [
            "Input-pipeline process peak resident memory (MiB)",
            "Includes interpreter/TensorFlow startup and streaming verification; not buffer-only memory.",
            "Five fresh process peaks per implementation/case; sample SD and variance use ddof=1.",
            "rows | implementation | n | mean MiB | median MiB | sample SD MiB | variance MiB^2",
        ]
        for case in summary["cases"]:
            for implementation, value in case["by_implementation"].items():
                lines.append(
                    f"{case['rows']} | {implementation} | {value['count']} | {value['mean']:.3f} | "
                    f"{value['median']:.3f} | {value['sample_sd']:.3f} | {value['sample_variance']:.6f}"
                )
            delta = case["paired_indexed_minus_baseline"]
            lines.append(
                f"  paired indexed-baseline mean: {delta['mean']:.3f} MiB; sample SD {delta['sample_sd']:.3f} MiB"
            )
        if plan["purpose"].startswith("harness-smoke"):
            lines = ["SMOKE CHECK ONLY: no performance comparison should be quoted from this run."]
        lines.extend(
            [
                "Synthetic shared binary input mode; no accuracy, training-memory or latency claim.",
                f"Plan SHA-256: {plan_hash}",
                f"Saved results to: {destination}",
            ]
        )
        (destination / "SUMMARY.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return destination
    except BaseException as error:
        if destination is not None:
            write_json(
                destination / "failure.json",
                {
                    "status": "failed",
                    "type": type(error).__name__,
                    "message": str(error),
                    "plan_sha256": plan_hash,
                    "note": "No incomplete run is summarized or automatically retried; keep these receipts.",
                },
            )
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "prepare", help="Create a new frozen experiment, without measuring"
    )
    freeze.add_argument("--baseline", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--smoke", action="store_true", help="Eight-row harness test, not a result")
    execute = commands.add_parser(
        "run", help="Run every planned pair in fresh sequential processes"
    )
    execute.add_argument("experiment", type=Path)
    execute.add_argument("--power-mode", default="not recorded")
    execute.add_argument("--power-source", default="not recorded")
    execute.add_argument("--background", default="not recorded")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.baseline, args.output, smoke=args.smoke)
        print(f"Frozen plan: {args.output / 'plan.json'}")
        print(f"Plan SHA-256: {digest(args.output / 'plan.json')}")
    else:
        run(
            args.experiment,
            power_mode=args.power_mode,
            power_source=args.power_source,
            background=args.background,
        )


if __name__ == "__main__":
    main()
