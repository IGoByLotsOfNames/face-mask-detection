"""One fresh-process input-pipeline memory observation; no model or timing metric.

The metric is the process-lifetime resident-memory high-water mark, including
interpreter/imports, validation, dataset construction and full-pass checks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path, PurePosixPath

WINDOWS_METHOD = "Windows GetProcessMemoryInfo PeakWorkingSetSize"
PACKAGES = ("tensorflow", "keras", "numpy", "Pillow")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            value.update(chunk)
    return value.hexdigest()


def _json(path, limit):
    with Path(path).open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("JSON input exceeds the worker size bound")
    return json.loads(content), hashlib.sha256(content).hexdigest()


def contained_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise ValueError("Expected a contained POSIX relative path")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe relative path")
    candidate = root.joinpath(*path.parts)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes the experiment")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink experiment members are not accepted")
    return candidate


def _check_hash(path, expected):
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or not path.is_file()
        or digest(path) != expected
    ):
        raise ValueError(f"Input hash mismatch: {path.name}")


def runtime_identity():
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
    }


def validate_request(plan_path, implementation, rows, seed, output):
    """Validate all requested inputs without importing TensorFlow or decoding pixels."""
    plan_path = Path(plan_path).absolute()
    if plan_path.is_symlink():
        raise ValueError("Plan must not be a symlink")
    root = plan_path.parent.resolve()
    plan, plan_sha256 = _json(plan_path, 1024 * 1024)
    if (
        not isinstance(plan, dict)
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("Unsupported worker plan")
    if implementation not in ("baseline", "indexed"):
        raise ValueError("Unknown implementation")
    for field in ("cases", "seeds"):
        values = plan.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(
                type(value) is not int
                or value < (1 if field == "cases" else 0)
                or value > (2**24 if field == "cases" else 2**32 - 1)
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"Invalid plan {field}")
    if (
        type(rows) is not int
        or rows not in plan["cases"]
        or type(seed) is not int
        or seed not in plan["seeds"]
    ):
        raise ValueError("Case or seed is not declared by the frozen plan")
    if (
        plan.get("image_size") != [128, 128]
        or type(plan.get("batch_size")) is not int
        or plan["batch_size"] != 32
    ):
        raise ValueError("Worker requires image_size [128,128] and batch_size 32")
    if plan.get("runtime") != runtime_identity():
        raise ValueError("Runtime differs from the frozen plan")
    source_record = plan["sources"][implementation]
    source = contained_path(root, source_record["path"])
    _check_hash(source, source_record["sha256"])
    inputs_path = contained_path(root, plan["inputs_path"])
    inputs, inputs_sha256 = _json(inputs_path, 16 * 1024 * 1024)
    if inputs_sha256 != plan["inputs_sha256"]:
        raise ValueError("Input manifest hash mismatch")
    if (
        not isinstance(inputs, dict)
        or type(inputs.get("schema_version")) is not int
        or inputs["schema_version"] != 1
        or inputs.get("class_names") != ["synthetic_a", "synthetic_b"]
        or not isinstance(inputs.get("rows"), list)
        or len(inputs["rows"]) < rows
    ):
        raise ValueError("Invalid synthetic input manifest")
    selected = inputs["rows"][:rows]
    paths = set()
    for identity, row in enumerate(selected):
        if (
            not isinstance(row, dict)
            or type(row.get("id")) is not int
            or row["id"] != identity
            or type(row.get("label")) is not int
            or row["label"] != identity % 2
        ):
            raise ValueError("Synthetic IDs must be sequential with label=id%2")
        image = contained_path(root, row["path"])
        if image in paths:
            raise ValueError("Repeated input path")
        paths.add(image)
        _check_hash(image, row["sha256"])
    output = Path(output).absolute()
    results = root / "results"
    if (
        not output.resolve().is_relative_to(results.resolve())
        or output.resolve() == results.resolve()
    ):
        raise ValueError("Output must be inside this experiment's results directory")
    contained_path(root, output.relative_to(root).as_posix())
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise FileExistsError("Output must be fresh with an existing parent directory")
    return {
        "root": root,
        "plan_path": plan_path,
        "plan": plan,
        "plan_sha256": plan_sha256,
        "source": source,
        "source_sha256": source_record["sha256"],
        "inputs_path": inputs_path,
        "inputs_sha256": inputs_sha256,
        "selected": selected,
        "output": output,
    }


def _windows_peak_bytes():
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return int(counters.PeakWorkingSetSize)


def normalize_resource_peak(value, system):
    if type(value) not in (int, float) or value <= 0 or not float(value).is_integer():
        raise ValueError("Invalid process peak resident-memory value")
    if system == "Linux":
        return int(value) * 1024
    if system == "Darwin":
        return int(value)
    raise RuntimeError("No defined ru_maxrss unit conversion for this platform")


def peak_memory():
    system = platform.system()
    if system == "Windows":
        value = _windows_peak_bytes()
        if type(value) is not int or value <= 0:
            raise ValueError("Invalid Windows peak working set")
        return value, WINDOWS_METHOD
    if system in ("Linux", "Darwin"):
        import resource

        return (
            normalize_resource_peak(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, system),
            "resource.getrusage(RUSAGE_SELF).ru_maxrss; "
            + ("KiB converted to bytes" if system == "Linux" else "bytes"),
        )
    raise RuntimeError("Peak resident-memory measurement is unsupported on this platform")


def run_worker(plan_path, implementation, rows, seed, output):
    request = validate_request(plan_path, implementation, rows, seed, output)
    os.environ.update(
        CUDA_VISIBLE_DEVICES="-1",
        TF_NUM_INTRAOP_THREADS="2",
        TF_NUM_INTEROP_THREADS="1",
        TF_CPP_MIN_LOG_LEVEL="2",
    )
    import numpy as np
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    specification = importlib.util.spec_from_file_location(
        "_frozen_input_pipeline", request["source"]
    )
    if specification is None or specification.loader is None:
        raise ValueError("Cannot load frozen dataset source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    manifest = {
        "class_names": ["synthetic_a", "synthetic_b"],
        "rows": [dict(row, split="train") for row in request["selected"]],
    }
    pipeline = module.dataset(
        request["root"], manifest, "train", (128, 128), 32, shuffle=True, seed=seed
    )
    seen, batches = set(), 0
    for images, labels in pipeline:
        pixels, targets = images.numpy(), labels.numpy()
        if (
            pixels.dtype != np.float32
            or pixels.ndim != 4
            or pixels.shape[1:] != (128, 128, 3)
            or targets.shape != (len(pixels), 1)
            or targets.dtype != np.float32
            or not 1 <= len(pixels) <= 32
        ):
            raise ValueError("Pipeline batch does not match the binary RGB float32 contract")
        for image, target in zip(pixels, targets):
            colour = image[0, 0]
            if not np.isfinite(colour).all() or (colour < 0).any() or (colour > 255).any():
                raise ValueError("Synthetic pixels are outside their byte range")
            identity = int(colour[0]) + (int(colour[1]) << 8) + (int(colour[2]) << 16)
            expected = np.asarray(
                [identity & 255, (identity >> 8) & 255, (identity >> 16) & 255], dtype=np.float32
            )
            if identity not in range(rows) or identity in seen or not np.all(image == expected):
                raise ValueError("Synthetic image ID, content or uniqueness check failed")
            if target[0] != identity % 2:
                raise ValueError("Synthetic label check failed")
            seen.add(identity)
        batches += 1
    if seen != set(range(rows)) or batches != (rows + 31) // 32:
        raise ValueError("Pipeline did not consume exactly the requested corpus once")
    _check_hash(request["plan_path"], request["plan_sha256"])
    _check_hash(request["source"], request["source_sha256"])
    _check_hash(request["inputs_path"], request["inputs_sha256"])
    for row in request["selected"]:
        _check_hash(contained_path(request["root"], row["path"]), row["sha256"])
    membership = hashlib.sha256(
        b"".join(identity.to_bytes(4, "little") for identity in sorted(seen))
    ).hexdigest()
    observed_runtime = runtime_identity()
    observed_system = platform.platform()
    observed_devices = [device.device_type for device in tf.config.list_logical_devices()]
    # Sample after all validation, and before serializing/writing the receipt.
    peak_bytes, method = peak_memory()
    result = {
        "schema_version": 1,
        "implementation": implementation,
        "rows": rows,
        "seed": seed,
        "metric": "process_lifetime_peak_rss_bytes",
        "peak_rss_bytes": peak_bytes,
        "metric_method": method,
        "plan_sha256": request["plan_sha256"],
        "source_sha256": request["source_sha256"],
        "inputs_sha256": request["inputs_sha256"],
        "image_size": [128, 128],
        "batch_size": 32,
        "batches": batches,
        "processed_rows": len(seen),
        "membership_sha256": membership,
        "correctness": "all constant RGB pixels encode each ID exactly once; label=id%2",
        "runtime": observed_runtime,
        "system": observed_system,
        "tensorflow_devices": observed_devices,
        "threads": {"intra_op": 2, "inter_op": 1},
        "scope": "process lifetime through imports, preflight, one complete dataset pass and correctness/provenance checks; excludes receipt writing; no model",
    }
    with request["output"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--implementation", required=True, choices=("baseline", "indexed"))
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run_worker(args.plan, args.implementation, args.rows, args.seed, args.output)
    print(
        json.dumps(
            {
                "implementation": result["implementation"],
                "rows": result["rows"],
                "seed": result["seed"],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
