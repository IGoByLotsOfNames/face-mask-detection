"""Launcher boundary tests; never install packages or contact a package index."""

import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("demo_launcher_under_test", ROOT / "demo.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class DemoLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        # Windows runners may expose TEMP through its 8.3 alias.
        self.root = Path(self.temporary.name).resolve()

    def owned_environment(self):
        directory = self.root / ".demo-venv"
        directory.mkdir()
        (directory / ".demo-owner").write_text(launcher.OWNER_TEXT, encoding="utf-8")
        return directory

    def test_runtime_requires_actual_supported_import_receipt(self):
        for receipt, expected in [
            ({"python": [3, 12], "pillow": "12.0.0"}, True),
            ({"python": [3, 10], "pillow": "12.0.0"}, False),
            ({"python": [3, 12], "pillow": "11.0.0"}, False),
            ({}, False),
        ]:
            with self.subTest(receipt=receipt):
                completed = subprocess.CompletedProcess([], 0, json.dumps(receipt), "")
                with patch.object(launcher.subprocess, "run", return_value=completed) as run:
                    self.assertEqual(expected, launcher.usable_runtime("python"))
                    self.assertEqual(30, run.call_args.kwargs["timeout"])
                    self.assertIn("-I", run.call_args.args[0])

    def test_failed_or_timed_out_runtime_probe_is_not_accepted(self):
        with patch.object(
            launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)
        ):
            self.assertFalse(launcher.usable_runtime("python"))
        with patch.object(
            launcher.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 30)
        ):
            self.assertFalse(launcher.usable_runtime("python"))

    def test_reuse_current_runtime_without_setup(self):
        with (
            patch.object(launcher, "usable_runtime", return_value=True),
            patch.object(launcher, "checked_setup") as setup,
        ):
            self.assertEqual(Path(sys.executable), launcher.select_runtime(self.root))
            setup.assert_not_called()
        self.assertFalse((self.root / ".demo-venv").exists())

    def test_no_install_missing_dependency_preserves_directory(self):
        with (
            patch.object(launcher, "usable_runtime", return_value=False),
            patch.object(launcher, "checked_setup") as setup,
        ):
            with self.assertRaisesRegex(launcher.LaunchError, "Pillow 12.0.0 is unavailable"):
                launcher.select_runtime(self.root, no_install=True)
            setup.assert_not_called()
        self.assertEqual([], list(self.root.iterdir()))

    def test_existing_owned_environment_is_revalidated(self):
        environment = self.owned_environment()
        with patch.object(launcher, "usable_runtime", side_effect=[False, True]) as probe:
            result = launcher.select_runtime(self.root, no_install=True)
            self.assertEqual(launcher.private_python(environment), result)
            self.assertEqual(result, probe.call_args.args[0])

    def test_unowned_and_partial_environments_are_preserved(self):
        environment = self.root / ".demo-venv"
        environment.mkdir()
        keep = environment / "keep.txt"
        keep.write_text("user-owned", encoding="utf-8")
        with patch.object(launcher, "usable_runtime", return_value=False):
            with self.assertRaisesRegex(launcher.LaunchError, "not owned"):
                launcher.select_runtime(self.root)
            (environment / ".demo-owner").write_text(launcher.OWNER_TEXT, encoding="utf-8")
            with self.assertRaisesRegex(launcher.LaunchError, "incomplete or incompatible"):
                launcher.select_runtime(self.root)
        self.assertEqual("user-owned", keep.read_text(encoding="utf-8"))

    def test_offline_bootstrap_is_pinned_isolated_and_bounded(self):
        wheelhouse = self.root / "wheels"
        wheelhouse.mkdir()
        with (
            patch.object(launcher, "usable_runtime", side_effect=[False, True]),
            patch.object(launcher, "checked_setup") as setup,
        ):
            executable = launcher.select_runtime(self.root, wheelhouse=wheelhouse)
        creation, installation = setup.call_args_list
        self.assertIn("venv", creation.args[0])
        self.assertEqual(120, creation.args[1])
        command = installation.args[0]
        self.assertEqual(str(executable), command[0])
        for argument in (
            "--isolated",
            "--no-index",
            "--no-deps",
            "--only-binary=:all:",
            "Pillow==12.0.0",
        ):
            self.assertIn(argument, command)
        self.assertEqual(180, installation.args[1])
        self.assertFalse((self.root / ".demo-bootstrap.lock").exists())

    def test_failed_bootstrap_preserves_environment_and_releases_own_lock(self):
        with (
            patch.object(launcher, "usable_runtime", return_value=False),
            patch.object(
                launcher, "checked_setup", side_effect=launcher.LaunchError("failure fixture")
            ),
        ):
            with self.assertRaisesRegex(launcher.LaunchError, "Setup files were preserved"):
                launcher.select_runtime(self.root)
        self.assertTrue((self.root / ".demo-venv" / ".demo-owner").is_file())
        self.assertFalse((self.root / ".demo-bootstrap.lock").exists())

    def test_existing_lock_is_preserved(self):
        lock = self.root / ".demo-bootstrap.lock"
        lock.write_text("other process", encoding="utf-8")
        with patch.object(launcher, "usable_runtime", return_value=False):
            with self.assertRaisesRegex(launcher.LaunchError, "already locked"):
                launcher.select_runtime(self.root)
        self.assertEqual("other process", lock.read_text(encoding="utf-8"))
        self.assertFalse((self.root / ".demo-venv").exists())

    def test_active_partial_environment_gets_lock_guidance_and_stays_untouched(self):
        environment = self.owned_environment()
        lock = self.root / ".demo-bootstrap.lock"
        lock.write_text("active setup", encoding="utf-8")
        with patch.object(launcher, "usable_runtime", return_value=False) as probe:
            with self.assertRaisesRegex(launcher.LaunchError, "already locked") as caught:
                launcher.select_runtime(self.root)
        self.assertNotIn("rename", str(caught.exception).lower())
        self.assertEqual(1, probe.call_count)  # Do not probe an interpreter still being created.
        self.assertEqual("active setup", lock.read_text(encoding="utf-8"))
        self.assertEqual(
            launcher.OWNER_TEXT, (environment / ".demo-owner").read_text(encoding="utf-8")
        )
        self.assertEqual([".demo-owner"], [path.name for path in environment.iterdir()])

    def test_setup_exit_and_timeout_are_actionable(self):
        with patch.object(
            launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 19)
        ):
            with self.assertRaisesRegex(launcher.LaunchError, "exit 19"):
                launcher.checked_setup(["fake"], 5, "fixture")
        with patch.object(
            launcher.subprocess, "run", side_effect=subprocess.TimeoutExpired("fake", 5)
        ):
            with self.assertRaisesRegex(launcher.LaunchError, "fixture failed"):
                launcher.checked_setup(["fake"], 5, "fixture")

    def test_invocation_uses_script_root_and_propagates_exit_status(self):
        source = self.root / "src" / "mask_detection"
        source.mkdir(parents=True)
        (source / "demo.py").write_text("# fixture", encoding="utf-8")
        with (
            patch.object(launcher, "__file__", str(self.root / "demo.py")),
            patch.object(launcher, "select_runtime", return_value=Path(sys.executable)),
            patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 23)
            ) as run,
        ):
            self.assertEqual(
                23, launcher.main(["--no-install", "--output", "relative-output", "--open"])
            )
        command = run.call_args.args[0]
        self.assertEqual(
            str(Path("relative-output").resolve()), command[command.index("--output") + 1]
        )
        self.assertIn("--open", command)
        self.assertNotIn("--no-install", command)
        self.assertEqual(self.root, run.call_args.kwargs["cwd"])
        self.assertEqual(
            str(self.root / "src"),
            run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0],
        )

    def test_default_run_does_not_open_browser(self):
        with (
            patch.object(launcher, "select_runtime", return_value=Path(sys.executable)),
            patch.object(
                launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)
            ) as run,
            patch.object(launcher.Path, "is_file", return_value=True),
        ):
            self.assertEqual(0, launcher.main([]))
        self.assertNotIn("--open", run.call_args.args[0])

    def test_unsupported_python_fails_before_setup(self):
        errors = io.StringIO()
        with (
            patch.object(launcher.sys, "version_info", (3, 14)),
            patch.object(launcher.sys, "stderr", errors),
            patch.object(launcher, "select_runtime") as select,
        ):
            self.assertEqual(1, launcher.main([]))
        self.assertIn("Use Python 3.11, 3.12 or 3.13", errors.getvalue())
        select.assert_not_called()

    def test_invalid_wheelhouse_fails_before_setup(self):
        errors = io.StringIO()
        with (
            patch.object(launcher.Path, "is_file", return_value=True),
            patch.object(launcher.sys, "stderr", errors),
            patch.object(launcher, "select_runtime") as select,
        ):
            self.assertEqual(1, launcher.main(["--wheelhouse", str(self.root / "absent")]))
        self.assertIn("Wheelhouse is not a directory", errors.getvalue())
        select.assert_not_called()


if __name__ == "__main__":
    unittest.main()
