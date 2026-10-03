"""Run real data contracts on included synthetic samples, without a trained model."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import uuid
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .annotations import CLASS_NAMES, prepare_annotations
from .data import create_split, digest, read_groups, safe_path, validate_manifest
from .demo_report import render_report
from .metrics import evaluation_report

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTICE = (
    "Synthetic illustrations and scripted predictions; "
    "no trained face-mask model or accuracy claim."
)


def write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_fixture(source: Path) -> tuple[dict, dict, list]:
    """Validate the checked-in fixture before creating output or decoding images."""
    manifest_path = safe_path(source, "manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("fixture_kind") != "synthetic-software-demo"
        or manifest.get("coordinate_convention") != "zero-based-half-open"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("Unsupported synthetic demo fixture")
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name not in ("README.md", "manifest.json")
    }
    if actual != set(manifest["files"]):
        raise ValueError("Demo sample inventory differs from its manifest")
    for name, expected in manifest["files"].items():
        if digest(safe_path(source, name)) != expected:
            raise ValueError(f"Demo input changed: {name}")
    expected = json.loads((source / "expected.json").read_text(encoding="utf-8"))
    predictions = json.loads((source / "predictions.json").read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("Expected explicit scripted prediction rows")
    return manifest, expected, predictions


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def run_demo(source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).absolute()
    fixture, expected, scripted = read_fixture(source)
    _check(
        not output.resolve().is_relative_to(source) and not source.is_relative_to(output.resolve()),
        "Sample and output directories must not contain each other",
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("Demo output already exists; choose a new directory")
    output.mkdir(parents=True)
    try:
        prepared = output / "prepared"
        receipt = prepare_annotations(
            source / "raw", prepared, coordinate_convention=fixture["coordinate_convention"]
        )
        split = output / "split"
        manifest = create_split(
            prepared / "crops", split, groups=read_groups(prepared / "groups.csv"), seed=17
        )
        _check(manifest == validate_manifest(split), "Generated split validation failed")
        _check(manifest["class_names"] == list(CLASS_NAMES), "Class order changed")
        counts = dict(receipt["counts"], duplicate_groups=len(receipt["duplicate_groups"]))
        _check(
            counts["duplicate_groups"] == expected["duplicate_groups"], "Unexpected duplicate count"
        )
        for name, value in expected["counts"].items():
            _check(counts[name] == value, f"Unexpected fixture count: {name}")
        split_counts = dict(Counter(row["split"] for row in manifest["rows"]))
        if "split_counts" in expected:
            _check(split_counts == expected["split_counts"], "Unexpected fixture split counts")

        group_splits, hash_splits = defaultdict(set), defaultdict(set)
        for row in manifest["rows"]:
            group_splits[row["group"]].add(row["split"])
            hash_splits[row["sha256"]].add(row["split"])
        _check(all(len(v) == 1 for v in group_splits.values()), "Source group crosses splits")
        _check(all(len(v) == 1 for v in hash_splits.values()), "Exact duplicate crosses splits")
        _check(
            {(r["split"], r["label"]) for r in manifest["rows"]}
            == {(s, c) for s in ("train", "validation", "test") for c in range(3)},
            "Missing class in a split",
        )

        predictions = {}
        for row in scripted:
            key = (row["source_image"], row["object_index"])
            _check(key not in predictions, "Repeated scripted sample identity")
            predictions[key] = row["probabilities"]
        crops = {row["path"]: row for row in receipt["rows"]}
        _check(
            set(predictions) == {(r["source_image"], r["object_index"]) for r in receipt["rows"]},
            "Scripted predictions must cover exactly the retained crops",
        )
        probabilities = [
            predictions[
                (crops[row["source"]]["source_image"], crops[row["source"]]["object_index"])
            ]
            for row in manifest["rows"]
        ]
        # All splits are included only to exercise report arithmetic on scripted data.
        evaluation = evaluation_report(
            manifest["rows"], {"scripted_fixture": probabilities}, manifest["class_names"]
        )
        metrics = evaluation["metrics"]["scripted_fixture"]
        _check(
            metrics["confusion_matrix"] == expected["confusion_matrix"], "Fixture report differs"
        )
        evaluation.update(
            demo_notice=NOTICE, evaluation_scope="all synthetic crops; not a held-out model score"
        )
        write_json(output / "scripted-evaluation.json", evaluation)

        # Prove the existing-output guard on the actual splitter; no output is replaced.
        try:
            create_split(prepared / "crops", split, groups=read_groups(prepared / "groups.csv"))
        except FileExistsError:
            pass
        else:
            raise ValueError("Existing split was not protected")
        _check(validate_manifest(split) == manifest, "Overwrite attempt changed the split")
        _check(read_fixture(source)[0] == fixture, "Source fixture changed during demonstration")
        samples = []
        for row, prediction in zip(manifest["rows"], evaluation["samples"]):
            crop = crops[row["source"]]
            predicted_index = prediction["predicted_class_indices"]["scripted_fixture"]
            samples.append(
                {
                    "source_image": crop["source_image"],
                    "object_index": crop["object_index"],
                    "class_name": crop["class_name"],
                    "split": row["split"],
                    "probabilities": prediction["probabilities"]["scripted_fixture"],
                    "predicted_class": CLASS_NAMES[predicted_index],
                    "image_data_uri": "data:image/png;base64,"
                    + base64.b64encode((split / row["path"]).read_bytes()).decode("ascii"),
                }
            )
        checks = [
            {
                "name": "Input integrity",
                "status": "passed",
                "detail": "All included sample hashes match; inputs remain unchanged.",
            },
            {
                "name": "Annotation policies",
                "status": "passed",
                "detail": "Duplicate handled, conflicting pair quarantined, invalid box excluded.",
            },
            {
                "name": "Grouped splits",
                "status": "passed",
                "detail": "No source group or exact image copy crosses split boundaries.",
            },
            {
                "name": "Class coverage",
                "status": "passed",
                "detail": "All three categories appear in every split.",
            },
            {
                "name": "Scripted report",
                "status": "passed",
                "detail": "Probabilities align by sample identity and match the expected confusion matrix.",
            },
            {
                "name": "Overwrite protection",
                "status": "passed",
                "detail": "Attempting to reuse a split path was rejected; saved data stayed unchanged.",
            },
        ]
        summary = {
            "schema_version": 1,
            "kind": "synthetic-software-demo",
            "title": "Face Mask Detection · Pipeline demo",
            "notice": NOTICE,
            "class_names": list(CLASS_NAMES),
            "counts": counts,
            "split_counts": split_counts,
            "checks": checks,
            "confusion_matrix": metrics["confusion_matrix"],
            "per_class": [
                dict(name=name, **metric) for name, metric in zip(CLASS_NAMES, metrics["by_class"])
            ],
            "samples": samples,
            "run_summary_relative": "summary.json",
            "fixture_manifest_sha256": digest(source / "manifest.json"),
            "coordinate_convention": fixture["coordinate_convention"],
            "split_seed": 17,
            "evaluation_scope": "all synthetic crops; no trained model",
        }
        rendered = render_report(summary)
        write_json(output / "summary.json", summary)
        with (output / "report.html").open("x", encoding="utf-8") as stream:
            stream.write(rendered)
        return summary
    except Exception as error:
        write_json(
            output / "failure.json",
            {"status": "failed", "type": type(error).__name__, "message": str(error)},
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New output folder; existing paths are never overwritten"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the saved report in your default browser"
    )
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = args.output or PROJECT_ROOT / "demo-runs" / run_id
    try:
        result = run_demo(PROJECT_ROOT / "sample-data/demo", output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Demo failed: {error}\nChoose a new output path and keep any failure receipt.",
            file=sys.stderr,
        )
        return 1
    print("PASS: synthetic pipeline demo (no trained mask classifier)")
    print(
        f"Inputs: {result['counts']['source_images']} source images -> {result['counts']['crops']} retained crops"
    )
    print(
        f"Policies: {result['counts']['duplicate_groups']} duplicate group; {result['counts']['quarantined_source_images']} conflicting sources quarantined; {result['counts']['excluded_objects']} invalid box excluded"
    )
    print(
        "Splits: "
        + ", ".join(
            f"{name}={result['split_counts'][name]}" for name in ("train", "validation", "test")
        )
    )
    print("Checks: 6/6 passed; scripted matrix matches expected output")
    print(NOTICE)
    report = output.resolve() / "report.html"
    print(f"Report: {report}\nSummary: {output.resolve() / 'summary.json'}")
    if args.open:
        webbrowser.open(report.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
