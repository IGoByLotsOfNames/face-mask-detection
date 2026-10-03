# One-command software demonstration

Run from a source checkout with Python 3.11, 3.12 or 3.13:

```powershell
# face-mask-detection/demo.py
python demo.py
```

The launcher uses an existing compatible Pillow 12.0.0 installation, or creates
its own `.demo-venv` and installs that pinned package. First setup needs internet
unless a compatible wheel is supplied with `--wheelhouse PATH`. It does not
install TensorFlow, OpenCV or NumPy, modify global packages, open a camera or
start a web server. Later runs reuse the local environment after checking it.

Open the printed `report.html` path in your browser. The report embeds its
illustrations, styles and script; it needs no server or remote assets. Keep
`summary.json` beside it for the machine-readable summary link. Use `--open`
to explicitly open the report in your default browser after a successful run.

```powershell
# face-mask-detection/demo.py
python demo.py --open
```

## What it demonstrates

This is the actual annotation, crop, grouped-split and evaluation
reporting code running on included synthetic illustrations. It demonstrates
three documented categories: mask worn, no mask, and mask worn incorrectly.
It is a software demonstration, **not a trained mask classifier**.

The inputs are nine illustrated scenes plus deliberately problematic examples:
one exact duplicate, two identical images with conflicting annotations, and
one out-of-bounds object. Their coordinate convention is explicitly
`zero-based-half-open`. The samples are original generated illustrations;
no historical photos or camera captures are included.

The included probability vectors are hand-authored fixtures with deliberate
mismatches. They exercise reporting and sample alignment. Their numeric metrics
are not evidence of classifier accuracy. All 27 retained crops, across all
splits, are used solely for that reporting check; this is not held-out model
evaluation. The real input-pipeline memory benchmark remains a separate result
in [measurement-results.md](measurement-results.md).

## Expected output

The launcher may print dependency setup messages on its first run. The pipeline
then prints this fixed result, followed by the generated file paths:

```text
# face-mask-detection/demo-runs/<run-id>/console-output
PASS: synthetic pipeline demo (no trained mask classifier)
Inputs: 12 source images -> 27 retained crops
Policies: 1 duplicate group; 2 conflicting sources quarantined; 1 invalid box excluded
Splits: train=21, validation=3, test=3
Checks: 6/6 passed; scripted matrix matches expected output
Synthetic illustrations and scripted predictions; no trained face-mask model or accuracy claim.
```

The fixture confusion matrix is `[[6,3,0],[0,6,3],[3,0,6]]`, ordered as
`mask_weared_incorrect`, `with_mask`, `without_mask`; rows are fixture labels and
columns are scripted predictions. This expected arithmetic is fixed by the
fixture specification, rather than learned by a model.

The report shows preparation counts, six checks, the fixture matrix, per-class
reporting, and all retained crops with their source, split and scripted output.
Filter the gallery by partition or category. Every category has nine crops;
each validation/test category has one crop.

## Output and failure handling

Each invocation creates a fresh timestamp/UUID folder under `demo-runs/`.
It contains `prepared/`, `split/`, `scripted-evaluation.json`, `summary.json`
and `report.html`. For a named destination:

```powershell
# face-mask-detection/demo.py
python demo.py --output my-demo-run
```

Relative output paths are resolved from the caller's current directory.
Existing output directories are refused, including empty ones. Included inputs
are hash-checked and never edited. Failures after output creation preserve a
`failure.json` and any intermediate files instead of printing success.

`--no-install` requires an already usable runtime and forbids package setup.
To bootstrap without network, supply a directory containing a compatible
Pillow 12.0.0 wheel:

```powershell
# face-mask-detection/demo.py
python demo.py --wheelhouse path/to/wheels
```

If setup was interrupted, the launcher preserves the partial environment and
explains how to rename it aside. It does not delete or silently repair an
unknown environment. A setup lock prevents concurrent installation; inspect
an interrupted setup before clearing a stale lock.

## How the demonstration works

1. The launcher selects a usable interpreter and isolates any missing dependency.
2. The fixture manifest verifies file identities before image processing.
3. The existing annotation code applies explicit crop/duplicate policies and
   records each decision. It does not guess annotation coordinates.
4. The existing splitter keeps each source image's crops together and verifies
   class coverage and exact-duplicate separation across partitions.
5. Scripted probabilities are joined by source image and object index, so
   changing the order of prediction rows cannot change which crop they describe.
6. The actual metric functions produce the report. HTML escapes text, embeds
   local illustrations and labels the demonstration's limits.

A source-image group is not a verified person identity. This fixture confirms
the grouping mechanics, not the absence of person-level leakage in unknown data.
The demo establishes no real-image accuracy, calibration, latency or deployment
reliability. Training and camera instructions are separate in the
[research workflow](research-workflow.md).
