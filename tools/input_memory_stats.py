"""Validate and summarize paired process-lifetime peak-memory measurements.

This module never runs a worker, trusts no timing or memory claim on its own,
and does not verify source/runtime/file provenance; the parent runner owns those
checks. It requires one completed worker record for every declared design slot.
"""

from __future__ import annotations

import math
import re
import statistics

MIB = 1024 * 1024
METRIC = "process_lifetime_peak_rss_bytes"
IMPLEMENTATIONS = ("baseline", "indexed")


def _identities(values, name, *, minimum_count, minimum_value):
    if (
        not isinstance(values, list)
        or len(values) < minimum_count
        or any(type(value) is not int or value < minimum_value for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must be a list of unique valid integers")
    return values


def _statistics(values, unit):
    """Sample dispersion across process values, retaining every accepted run."""
    try:
        variance = statistics.variance(values)
        result = dict(
            unit=unit,
            count=len(values),
            mean=statistics.mean(values),
            median=statistics.median(values),
            sample_sd=math.sqrt(variance),
            sample_variance=variance,
            min=min(values),
            max=max(values),
            ddof=1,
        )
    except (OverflowError, ValueError) as error:
        raise ValueError("Peak-memory values exceed finite summary precision") from error
    if any(
        not math.isfinite(result[key])
        for key in ("mean", "median", "sample_sd", "sample_variance", "min", "max")
    ):
        raise ValueError("Peak-memory values exceed finite summary precision")
    return result


def summarize(plan, records):
    """Return JSON-compatible, separate summaries for every planned row count.

    The paired difference and ratio match baseline/indexed runs by seed, then
    summarize those paired values. Neither is derived from a ratio/difference
    of aggregate means. Worker order is irrelevant; declared plan order is kept.
    """
    if not isinstance(plan, dict):
        raise ValueError("Plan must be an object")
    cases = _identities(plan.get("cases"), "cases", minimum_count=1, minimum_value=1)
    seeds = _identities(plan.get("seeds"), "seeds", minimum_count=2, minimum_value=0)
    expected = {
        (rows, seed, implementation)
        for rows in cases
        for seed in seeds
        for implementation in IMPLEMENTATIONS
    }
    if isinstance(records, (str, bytes, dict)):
        raise ValueError("Worker records must be an iterable of objects")
    try:
        records = list(records)
    except TypeError as error:
        raise ValueError("Worker records must be an iterable of objects") from error
    found = {}
    method = None
    plan_hash = None
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Worker record must be an object")
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("Unsupported worker schema version")
        rows, seed = record.get("rows"), record.get("seed")
        implementation = record.get("implementation")
        if (
            type(rows) is not int
            or type(seed) is not int
            or not isinstance(implementation, str)
            or implementation not in IMPLEMENTATIONS
        ):
            raise ValueError("Invalid worker slot identity")
        slot = (rows, seed, implementation)
        if slot not in expected:
            raise ValueError("Worker record has an unplanned slot")
        if slot in found:
            raise ValueError("Duplicate worker slot")
        if record.get("metric") != METRIC:
            raise ValueError("Unsupported worker metric")
        peak = record.get("peak_rss_bytes")
        if type(peak) is not int or peak <= 0:
            raise ValueError("Peak RSS must be a positive integer byte count")
        current_method = record.get("metric_method")
        if not isinstance(current_method, str) or not current_method.strip():
            raise ValueError("Worker metric method must be nonempty text")
        current_hash = record.get("plan_sha256")
        if not isinstance(current_hash, str) or re.fullmatch(r"[0-9a-f]{64}", current_hash) is None:
            raise ValueError("Worker plan hash must be a lowercase SHA-256 hex digest")
        if method is not None and current_method != method:
            raise ValueError("Cannot combine heterogeneous peak-memory metric methods")
        if plan_hash is not None and current_hash != plan_hash:
            raise ValueError("Cannot combine worker records from different plan hashes")
        method, plan_hash = current_method, current_hash
        found[slot] = peak
    if set(found) != expected:
        raise ValueError("Missing planned worker slots")

    summaries = []
    for rows in cases:
        try:
            baseline = [found[(rows, seed, "baseline")] / MIB for seed in seeds]
            indexed = [found[(rows, seed, "indexed")] / MIB for seed in seeds]
            # Subtract integers first, preserving the exact byte difference.
            difference = [
                (found[(rows, seed, "indexed")] - found[(rows, seed, "baseline")]) / MIB
                for seed in seeds
            ]
            ratio = [
                found[(rows, seed, "indexed")] / found[(rows, seed, "baseline")] for seed in seeds
            ]
        except OverflowError as error:
            raise ValueError("Peak-memory values exceed finite summary precision") from error
        summaries.append(
            dict(
                rows=rows,
                seeds=list(seeds),
                by_implementation={
                    "baseline": _statistics(baseline, "MiB"),
                    "indexed": _statistics(indexed, "MiB"),
                },
                paired_indexed_minus_baseline=_statistics(difference, "MiB"),
                paired_indexed_over_baseline=_statistics(ratio, "ratio"),
            )
        )
    return dict(
        schema_version=1,
        metric=METRIC,
        metric_method=method,
        plan_sha256=plan_hash,
        measurement_notes=[
            "Each value is one fresh process lifetime peak RSS, including TensorFlow initialization and other process memory; it is not isolated input-buffer memory.",
            "Dispersion is sample spread across the recorded process runs (ddof=1), not population dispersion or request latency.",
            "Baseline and indexed runs are paired by the same declared seed; differences and ratios are calculated per pair before summarizing.",
            "One MiB is 1048576 bytes. Sample variance has squared units: MiB^2 or ratio^2.",
            "Every planned record is retained; no outliers are discarded and row-count cases are never pooled.",
            "Plan hash consistency is checked here; the parent runner must verify the hash against the actual plan and verify source, runtime and worker-file provenance.",
        ],
        cases=summaries,
    )
