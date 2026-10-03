"""Run the synthetic sample demo without installing the research stack globally."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

PILLOW_VERSION = "12.0.0"
OWNER_TEXT = "face-mask-detection sample demo environment v1\n"
PROBE = """
import json, sys
from PIL import Image, __version__
Image.new('RGB', (1, 1)).load()
print(json.dumps({'python': list(sys.version_info[:2]), 'pillow': __version__}))
"""


class LaunchError(Exception):
    """An actionable local setup error, with no destructive repair."""


def usable_runtime(executable):
    """Check actual imports in an isolated child, without trusting package metadata."""
    try:
        result = subprocess.run(
            [str(executable), "-I", "-B", "-c", PROBE],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            return False
        observed = json.loads(result.stdout)
        return observed["python"] in ([3, 11], [3, 12], [3, 13]) and (
            observed["pillow"] == PILLOW_VERSION
        )
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError):
        return False


def private_python(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def checked_setup(command, timeout, description):
    try:
        completed = subprocess.run(command, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f"{description} failed: {exc}") from exc
    if completed.returncode:
        raise LaunchError(f"{description} failed (exit {completed.returncode}).")


def select_runtime(root, *, no_install=False, wheelhouse=None):
    if usable_runtime(sys.executable):
        return Path(sys.executable)
    lock = root / ".demo-bootstrap.lock"
    lock_message = (
        f"Demo setup is already locked: {lock}. If an earlier setup stopped, verify no setup "
        "is running before removing that stale lock."
    )
    if lock.exists() or lock.is_symlink():
        raise LaunchError(lock_message)
    environment = root / ".demo-venv"
    owner = environment / ".demo-owner"
    executable = private_python(environment)
    if environment.exists():
        if environment.is_symlink() or getattr(environment, "is_junction", lambda: False)():
            raise LaunchError(f"Refusing redirected environment: {environment}")
        if not owner.is_file() or owner.read_text(encoding="utf-8") != OWNER_TEXT:
            raise LaunchError(
                f"Existing {environment} is not owned by this launcher; keep it intact."
            )
        if usable_runtime(executable):
            return executable
        # A concurrent bootstrap may have started after the initial lock check.
        if lock.exists() or lock.is_symlink():
            raise LaunchError(lock_message)
        raise LaunchError(
            f"The existing demo environment is incomplete or incompatible: {environment}. "
            "Rename it aside, then rerun to create a fresh one. No files were removed."
        )
    if no_install:
        raise LaunchError(
            "Pillow 12.0.0 is unavailable. Run again without --no-install to create a private "
            ".demo-venv, or provide --wheelhouse PATH containing a compatible Pillow 12.0.0 wheel."
        )
    token = uuid.uuid4().hex
    try:
        with lock.open("x", encoding="utf-8") as handle:
            handle.write(token)
    except FileExistsError as exc:
        raise LaunchError(lock_message) from exc
    try:
        environment.mkdir()  # Exclusive: never modify an environment created by another process.
        with owner.open("x", encoding="utf-8") as handle:
            handle.write(OWNER_TEXT)
        if wheelhouse is None:
            print(
                "First setup: downloading Pillow 12.0.0 into private .demo-venv (internet needed)."
            )
        else:
            print(f"First setup: installing Pillow 12.0.0 offline from {wheelhouse}.")
        checked_setup([sys.executable, "-I", "-m", "venv", str(environment)], 120, "venv creation")
        command = [
            str(executable),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "install",
            "--no-input",
            "--only-binary=:all:",
            "--no-deps",
            "--retries",
            "1",
            "--timeout",
            "30",
        ]
        if wheelhouse is not None:
            command += ["--no-index", "--find-links", str(wheelhouse)]
        command += [f"Pillow=={PILLOW_VERSION}"]
        checked_setup(command, 180, "Pillow installation")
        if not usable_runtime(executable):
            raise LaunchError("The new environment failed its Python/Pillow import check.")
        return executable
    except (OSError, LaunchError) as exc:
        raise LaunchError(
            f"{exc} Setup files were preserved at {environment}; if incomplete, rename that "
            "directory aside before trying again."
        ) from exc
    finally:
        if lock.is_file() and lock.read_text(encoding="utf-8") == token:
            lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the synthetic sample demo. It does not classify faces or use a camera."
    )
    parser.add_argument(
        "--output", type=Path, help="New output directory; relative to your current folder"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the generated report in your browser"
    )
    parser.add_argument(
        "--no-install", action="store_true", help="Use available dependencies only; never install"
    )
    parser.add_argument(
        "--wheelhouse", type=Path, help="Install offline from a local directory of wheels"
    )
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    try:
        if sys.version_info[:2] not in ((3, 11), (3, 12), (3, 13)):
            raise LaunchError("Use Python 3.11, 3.12 or 3.13 to run this launcher.")
        if not (root / "src" / "mask_detection" / "demo.py").is_file():
            raise LaunchError(
                "The demo source is missing. Keep demo.py beside the project's src folder."
            )
        wheelhouse = args.wheelhouse.resolve() if args.wheelhouse is not None else None
        if wheelhouse is not None and not wheelhouse.is_dir():
            raise LaunchError(f"Wheelhouse is not a directory: {wheelhouse}")
        executable = select_runtime(root, no_install=args.no_install, wheelhouse=wheelhouse)
        command = [str(executable), "-B", "-m", "mask_detection.demo"]
        if args.output is not None:
            command += ["--output", str(args.output.resolve())]
        if args.open:
            command += ["--open"]
        environment = os.environ.copy()
        source = str(root / "src")
        environment["PYTHONPATH"] = source + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
        environment["PYTHONNOUSERSITE"] = "1"
        print(
            "Running synthetic sample demo; no face classification, camera or training.", flush=True
        )
        return subprocess.run(command, cwd=root, env=environment, check=False).returncode
    except (LaunchError, OSError) as exc:
        print(f"Demo setup error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
