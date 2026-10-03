"""Core imports and CLI help must work without optional vision runtimes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src"
GUARD = """
import importlib.abc
import sys
class BlockOptionalRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tensorflow', 'cv2'}:
            raise AssertionError('Optional runtime import blocked: ' + fullname)
sys.meta_path.insert(0, BlockOptionalRuntime())
# Prove the guard is active in this fresh interpreter before checking imports.
for blocked in ('tensorflow', 'cv2'):
    try:
        __import__(blocked)
    except AssertionError as error:
        assert 'Optional runtime import blocked:' in str(error)
    else:
        raise AssertionError('Runtime import guard was not exercised')
"""


class CoreImportTests(unittest.TestCase):
    def run_guarded(self, body):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(SOURCE)
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-X", "utf8", "-c", GUARD + textwrap.dedent(body)],
                cwd=temp,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60,
                check=False,
            )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_core_modules_import_without_tensorflow_or_opencv(self):
        actual = self.run_guarded("""
            import importlib
            import json
            from pathlib import Path
            names = ['mask_detection', 'mask_detection.data', 'mask_detection.annotations',
                     'mask_detection.metrics', 'mask_detection.artifacts', 'mask_detection.model',
                     'mask_detection.train', 'mask_detection.evaluate', 'mask_detection.inference',
                     'mask_detection.webcam']
            modules = [importlib.import_module(name) for name in names]
            optional = [name for name in sys.modules
                        if name.split('.')[0] in {'tensorflow', 'cv2'}]
            print(json.dumps({'modules': names, 'optional': optional,
                              'source': str(Path(modules[0].__file__).resolve())}))
        """)
        self.assertEqual(10, len(actual["modules"]))
        self.assertEqual([], actual["optional"])
        self.assertEqual(SOURCE / "mask_detection" / "__init__.py", Path(actual["source"]))

    def test_every_cli_help_works_with_optional_runtimes_blocked(self):
        actual = self.run_guarded("""
            import contextlib
            import io
            import json
            import runpy
            help_text = {}
            for name in ('data', 'annotations', 'train', 'inference', 'evaluate', 'webcam'):
                module = 'mask_detection.' + name
                sys.argv = [module, '--help']
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    try:
                        runpy.run_module(module, run_name='__main__')
                    except SystemExit as result:
                        assert result.code == 0, (module, result.code)
                    else:
                        raise AssertionError('CLI help did not exit: ' + module)
                help_text[name] = output.getvalue()
            assert not any(name.split('.')[0] in {'tensorflow', 'cv2'} for name in sys.modules)
            print(json.dumps(help_text))
        """)
        self.assertEqual(
            {"data", "annotations", "train", "inference", "evaluate", "webcam"}, set(actual)
        )
        for module, help_text in actual.items():
            with self.subTest(module=module):
                self.assertIn("usage:", help_text.lower())
                self.assertIn("--help", help_text)


if __name__ == "__main__":
    unittest.main()
