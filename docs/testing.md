# Testing and automation

## Local commands

Use Python 3.12 for the complete CPU runtime. The core package supports Python
3.11 through 3.13; the CI matrix is configured to check those versions without ML dependencies.
From the repository root:

```powershell
# Repository root: face-mask-detection/
python -m pip install -e . -r requirements-dev.txt
python -m pip check
python -m ruff check .
python -m ruff format --check .
$env:RUN_ML_TESTS = '0'
python -m unittest discover -s tests -v
```

To format an intentional edit, run `python -m ruff format .`, then inspect the
diff and rerun lint/tests. Ruff's configured rules catch undefined names, unused
imports, import ordering and selected Python syntax/style errors. Formatting is
checked separately; CI never rewrites source files.

For the synthetic TensorFlow integration and subprocess CLI tests:

```powershell
# Repository root: face-mask-detection/
python -m pip install -r requirements-runtime.txt -r requirements-dev.txt -e .
$env:RUN_ML_TESTS = '1'
$env:TF_ENABLE_ONEDNN_OPTS = '0'
$env:TF_CPP_MIN_LOG_LEVEL = '2'
$env:TF_NUM_INTRAOP_THREADS = '2'
$env:TF_NUM_INTEROP_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
python -m coverage erase
python -m coverage run -m unittest discover -s tests -v
python -m coverage combine
python -m coverage json -o coverage.json
python -m coverage xml -o coverage.xml
python -m coverage report --fail-under=85
```

On Linux/macOS, set the equivalent variables using `export NAME=value`.
Coverage.py's subprocess support records child commands as well as the parent
test process. The 85% floor is its combined statement/branch coverage measure
for `src/mask_detection`; it is not a model accuracy score or a requirement that
every module individually reaches 85%. Tests, legacy wrappers and diagram
rendering are outside that measurement scope, though Ruff checks their Python.
The root-level demo launcher is also outside package coverage and has separate tests.
The source package's command-line entry points are included.

## Checks and their purpose

| Boundary | Check |
|---|---|
| Raw annotations to split | Generated XML/images produce traceable crops, exclusions and whole-group splits; original inputs remain unchanged |
| Split feasibility | A brute-force oracle enumerates small assignments independently of the optimized class-mask search |
| Model/data identity | Real manifests and sealed synthetic bundles reject class-order, provenance, split and overlap mismatches before inference |
| Probability mathematics | Known examples, rational Brier calculations, likelihood-derived log loss and row/class permutation invariants |
| Lightweight use | Core imports and six CLI help commands explicitly reject any TensorFlow/OpenCV import attempt |
| Complete CLI | Separate Python processes prepare a grouped split, train, predict and evaluate from an unrelated working directory |
| Runtime compatibility | Real TensorFlow covers both categorical bundles and legacy binary models |
| Filesystem failures | Injected reservation, fit, sealing and publication failures; existing outputs are preserved |
| Camera adapter | Fake devices check cleanup and ensure overlays cannot contaminate overlapping face crops |
| Reviewer demo | Included illustrations exercise real preparation/splitting/reporting; scripted vectors align by identity, inputs remain unchanged and existing outputs are refused |
| Demo bootstrap | Runtime probes, missing dependencies, preserved partial environments and setup-lock ownership; fresh offline installation and reuse checked separately |

The CLI tests exercise the source/installed-editable package. A separate
distribution check verifies source-archive contents and exercises the extracted
demo. Neither check establishes validation of a separately distributed wheel
or executable.

## CI configuration

The workflow runs on every push and pull request, with read-only repository
permissions and bounded job durations. Core jobs install, check dependencies,
lint, check formatting and test on Python 3.11/3.12/3.13. Runtime jobs use Python
3.12 on Ubuntu and Windows, run synthetic tests with subprocess-aware coverage,
enforce the 85% floor, and retain JSON/XML coverage reports for 14 days. The
reports contain coverage information, not images, datasets or trained models.
A sixth job builds the source distribution, verifies that its included inputs
match the checkout, then runs the demo from the extracted archive. Its checker
refuses unsafe archive paths and keeps output in a temporary directory.

Local execution and hosted results are separate records. Inspect the
[workflow run for the relevant commit](https://github.com/IGoByLotsOfNames/face-mask-detection/actions)
for remote job outcomes; the local results below do not establish a hosted pass.

## Coverage gaps and interpretation

- Synthetic integration verifies software contracts. Real mask recognition,
  calibration, class imbalance, person-independent generalization and subgroup
  performance require a separately justified data/evaluation protocol. These
  gaps limit conclusions about model quality.
- Hardware capture and Haar face-detector recall are not measured. Mock camera
  coverage cannot justify a real-camera reliability claim.
- Windows may not allow an unprivileged test to create symbolic links. That
  fixture then reports a skip. Linux CI is configured to exercise it; check
  the relevant workflow result for its outcome.
- Abrupt process termination, power loss, hostile concurrent filesystem writers
  and interrupted evaluation-report writes are not guaranteed by these tests.
  Model directory publication covers ordinary failures, not all crash modes.
- Controlled synthetic input-pipeline process memory was measured separately;
  see [the results and limits](measurement-results.md). Real-data/training memory,
  training speed and evaluation latency remain unmeasured. Coverage does not
  replace those studies.

## Recorded local verification

A complete Windows/Python 3.12 run on 3 October 2026 collected **157 tests:
156 passed and one Windows symlink fixture skipped**. Combined statement/branch
coverage of `src/mask_detection` was **89.39%**, above the configured 85% floor.
This includes all 15 launcher tests and the opt-in TensorFlow binary,
categorical and subprocess-CLI integration tests. The data in those integrations
are synthetic software fixtures, not a face-recognition study.

This fresh run covers the final launcher refinement. The earlier local record
of 155 passes and a later separate 15-test launcher check are superseded for
the current integrated checkout; their counts are not added together.
Application-package source still matches the measured revision. The
[dated verification record](evidence/integration-verification.json) records
the tested files and environment.

The launcher was also checked with a compatible local Pillow wheel for fresh
offline dependency setup, environment reuse and preservation of existing output.
That verifies a specific setup path, rather than every possible network or
platform. The [binary validation record](validation.md) describes an earlier
implementation and retains its own historical test count.

Configuration references: [Ruff](https://docs.astral.sh/ruff/configuration/),
[Coverage.py](https://coverage.readthedocs.io/en/latest/config.html),
[subprocess measurement](https://coverage.readthedocs.io/en/latest/subprocess.html),
[setup-python](https://github.com/actions/setup-python), and
[artifact uploads](https://github.com/actions/upload-artifact).
