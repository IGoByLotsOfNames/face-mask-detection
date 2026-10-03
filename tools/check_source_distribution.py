"""Check that an extracted source distribution can run the synthetic demo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def check_archive(archive: Path, source: Path) -> dict:
    expected_paths = {"demo.py", "LICENSE", "pyproject.toml", "README.md"}
    for directory in ("src/mask_detection", "sample-data/demo"):
        expected_paths.update(
            path.relative_to(source).as_posix()
            for path in (source / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    with tempfile.TemporaryDirectory(prefix="mask-source-check-") as temporary:
        extraction = Path(temporary) / "source"
        extraction.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir())
                ):
                    raise ValueError(f"Unexpected source archive member: {member.name}")
            bundle.extractall(extraction, filter="data")
        roots = list(extraction.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError("Expected one source-distribution root")
        root = roots[0]
        for relative in sorted(expected_paths):
            extracted = root / relative
            if (
                not extracted.is_file()
                or extracted.read_bytes() != (source / relative).read_bytes()
            ):
                raise ValueError(f"Missing or changed distribution input: {relative}")
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(root / "demo.py"),
                "--no-install",
                "--output",
                str(Path(temporary) / "demo"),
            ],
            cwd=temporary,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"Extracted demo failed:\n{result.stdout}\n{result.stderr}")
        summary = json.loads((Path(temporary) / "demo" / "summary.json").read_text("utf-8"))
        if "Checks: 6/6 passed" not in result.stdout:
            raise RuntimeError("Demo did not report its six expected pipeline checks")
        return {
            "archive": archive.name,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_files": sum(member.isfile() for member in members),
            "verified_source_and_sample_files": len(expected_paths),
            "demo_exit_code": result.returncode,
            "demo_checks": {"passed": 6, "total": 6},
            "demo_summary": {
                key: summary[key]
                for key in (
                    "kind",
                    "notice",
                    "counts",
                    "split_counts",
                    "fixture_manifest_sha256",
                    "evaluation_scope",
                )
            },
            "scope": "Extracted source demo using available Pillow; no model-accuracy claim",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = check_archive(args.archive.resolve(), args.source.resolve())
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
