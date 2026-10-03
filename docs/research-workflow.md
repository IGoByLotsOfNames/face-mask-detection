# Research workflow

These commands exercise the trainable pipeline. They require an
appropriate external dataset and, for prediction, a compatible model. No real
face dataset or trained mask classifier is included. For the included synthetic
software demonstration, start with [demo.md](demo.md).

## Environment

Use Python 3.12 for the pinned research runtime. From a source checkout on Windows:

```powershell
# face-mask-detection/pyproject.toml — repository root
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . -r requirements-runtime.txt
```

The commands below assume that environment's Python is selected. On PowerShell,
either activate it with `.\.venv\Scripts\Activate.ps1` or replace `python` with
`.\.venv\Scripts\python.exe`. On Linux/macOS, the environment interpreter is
`.venv/bin/python`; these platforms were not exercised during the local demo check.
The runtime requirements alone do not install this source package; keep `-e .`.

## Prepare annotated face crops

[The annotation adapter](../src/mask_detection/annotations.py)
accepts sibling `images/` and `annotations/` directories with PASCAL VOC XML.
It requires an explicit coordinate interpretation. Valid choices are
`voc-1-based-inclusive` and `zero-based-half-open`.

Dataset rights, labels, annotation coordinates and
appropriate person/session groups must be established for the chosen study.
The historical corpus's exact provenance and convention remain unresolved.
Do not copy the synthetic fixture's convention into a real-data study by default.

The following is a **template**, not a runnable default. Replace the placeholder
only after verifying the selected corpus's convention:

```text
# face-mask-detection/src/mask_detection/annotations.py — command template
python -m mask_detection.annotations raw-data prepared --coordinate-convention <verified-convention>
```

Default `--bounds-policy reject` excludes empty, reversed or
out-of-bounds boxes and records the reasons. Explicit `--bounds-policy clip`
records each adjustment; empty results remain excluded. Conflicting annotation
sets on byte-identical images quarantine the whole group. Matching duplicates
use one deterministic representative. Unsafe/malformed inputs abort preparation.

The new directory contains `crops/`, `groups.csv` and `preparation.json`, with
source hashes, original/effective boxes, crop hashes and exclusions. Source
images are not edited. A typical layout is:

```text
# face-mask-detection/prepared/ — generated output layout
prepared/
  crops/
    mask_weared_incorrect/crop001.png
    with_mask/crop002.png
    without_mask/crop003.png
  groups.csv
  preparation.json
```

## Grouped splitting, training and reserved-test evaluation

The adapter's initial group IDs are source-image hashes, not
verified person identities. When a person, session or derived image is known
to recur across source images, review the grouping CSV to merge those relations
before splitting. It must contain exactly `path,group` columns and cover every
crop, with paths relative to `crops/`.

The splitter joins declared groups and exact byte duplicates into components,
then assigns whole components to partitions. Every partition must contain every
class, so ratios are approximate. At least three independent components per
class are necessary, but not sufficient to guarantee a feasible joint split.
A bounded coverage search distinguishes proved infeasibility from exhausted search.

Run the following only with reviewed input/grouping files and fresh output paths:

```powershell
# face-mask-detection/src/mask_detection/ — research commands
# Substitute a reviewed grouping CSV when it merges known person/session relations.
python -m mask_detection.data prepared/crops split-data --groups prepared/groups.csv --seed 42
python -m mask_detection.train split-data --epochs 30 --output artifacts/mask-run
python -m mask_detection.evaluate artifacts/mask-run split-data --output artifacts/test-report.json
python -m mask_detection.inference artifacts/mask-run path/to/face-crop.png
```

Training uses the training and validation partitions, not the test
partition. It trains the CNN from scratch with Adam and early stopping; no
pretrained backbone is selected by these commands. Evaluation checks class order,
source/split identity and recorded train/validation overlap before using reserved
test samples. Exact hashes do not replace a sound study/grouping protocol.

Keep the complete `mask-run` directory together: `model.keras`,
`model.metadata.json`, `model.history.json` and `bundle.json`. The new bundle is
checked before directory publication. This handles ordinary failures, not all
power-loss or hostile-writer scenarios. Legacy `.keras` plus sidecar artifacts
remain supported; separate-file training output is deprecated.

## Model and output contracts

File and camera-crop inputs share RGB channel order, bilinear
resizing and float32 values in `[0,255]`. OpenCV's BGR input is converted before
prediction. Normalization resides inside the saved model. Sorted class names
define the recorded indices; metadata binds their order and output semantics.

Three-category models have a three-unit softmax head and sparse categorical
cross-entropy. Inference returns the ordered probability vector, selected class
index/name and selected probability. Argmax resolves exact ties to the lowest
recorded index. These probabilities are not calibrated confidence guarantees.

The two-class compatibility path has one sigmoid output interpreted as the
probability of class index 1. Headless and camera inference use threshold `0.5`.
For binary evaluation only, any alternative threshold must be chosen on
validation data before final test evaluation. Categorical evaluation rejects
`--threshold`, including an explicit `0.5`.

The [CNN schematic](visuals/cnn-architecture.png) shows the categorical head and
the separate legacy binary path. Both use the same four convolution/pool blocks,
global average pooling, dropout and 128-unit dense layer in the
[current model builder](../src/mask_detection/model.py).

Categorical reports retain sample identities and probability
vectors, ordered confusion matrices, per-class precision/recall/F1/support,
accuracy, macro recall (balanced accuracy), macro F1, log loss and multiclass
Brier score. Log loss uses natural logarithms and a `1e-15` true-class probability
floor. Brier score averages the sum of squared errors across classes, without
dividing by class count; its range is `[0,2]`. Probability rows must sum to one
within absolute `1e-6`; they are not silently renormalized.

Precision is `null` without predicted support. Recall and F1 are `null` without
true support. A supported but never-predicted class has F1 zero. Strict macro
recall/F1 are `null` if a required constituent is undefined. The API can average
aligned model probability vectors before applying the same decision rule.
Reports use fresh paths; an old result is not silently replaced.

## Optional camera adapter

After obtaining an appropriate compatible model, explicitly run:

```powershell
# face-mask-detection/src/mask_detection/webcam.py
python -m mask_detection.webcam artifacts/mask-run --device 0
```

The adapter uses OpenCV's Haar face detector, clips candidate face regions and
sends clean crops to the shared predictor. Overlays are drawn separately so one
box cannot contaminate another overlapping crop. Press Escape to exit. Cleanup
is in a `finally` path; tests use fake devices rather than a real camera.

Synthetic runtime tests establish software
contracts only. No real-data classification, camera reliability, calibration,
latency or subgroup result is claimed. Camera detection may fail before the
classifier receives any crop. Keep external face data, personal identifiers and
trained model artifacts excluded from version control. The included synthetic
illustration fixtures are intentionally tracked, with documented provenance.
