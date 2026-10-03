# Project history

## The original experiment

I started Face Mask Detection during the COVID-19 period, using Python, Kaggle
images and a face-detection library. It was an early exploration of how locating
a face and classifying its crop could work together.

The [earlier project README](https://github.com/IGoByLotsOfNames/face-mask-detection/blob/5f2e2ba9796b00255a3d93808f7ecdefdb6c5ecc/README.md)
records a **Commendation at the 2022 Coding Lab International Coding Competition**.
That attribution belongs to the original project, not the later engineering
revision or its benchmark. The award record was not independently rechecked as
part of this revision.

This account of the original motivation and implementation is my own. The new
synthetic samples do not replace or rebrand the historical Kaggle data.

## The maintained binary implementation

The current revision began from the preserved public commit
[`5f2e2ba`](https://github.com/IGoByLotsOfNames/face-mask-detection/tree/5f2e2ba9796b00255a3d93808f7ecdefdb6c5ecc).
It already contained a maintained two-class pipeline with grouped splitting,
model metadata, camera-free inference and automated tests. Those capabilities
were part of the starting point, rather than additions first made in this revision.

The [binary validation record](validation.md) retains the checks performed on
2 October 2026. Its smaller test count and synthetic training exercises describe
that historical state, not the current suite or mask-recognition accuracy.

## Extending the pipeline

The latest work adds the documented third category, explicit annotation
coordinates, traceable crop exclusions and stronger split, model and output
contracts. Models are saved with their metadata and training history as checked
bundles. The older binary contract remains supported.

Changing the input pipeline also created a measurable engineering question:
what happens if it shuffles integer indices before decoding images, instead of
shuffling decoded float32 images? I ran a frozen experiment on 3 October 2026.
Across the five paired runs at 4,096 synthetic images, mean process peak memory
fell from 1,036.840 MiB to 272.972 MiB. The [measurement record](measurement-results.md)
preserves the full 30-run comparison and its scope. It is a pipeline-memory
result, not a historical accuracy figure or a training-memory claim.

The [software demo](demo.md) makes the annotation, split and reporting paths
inspectable using original synthetic illustrations and scripted probabilities.
It demonstrates the software without distributing personal images, external
datasets or trained weights. It does not demonstrate real mask recognition.

Codex assisted the recent implementation, tests and documentation. The original
idea and early implementation remain my work; the current revision continues
that project with more explicit and testable engineering.

## Keeping the evidence in context

| Statement | Basis |
|---|---|
| COVID-era motivation and use of Python, Kaggle and face detection | My account of the original project |
| 2022 competition Commendation | Attribution preserved in the earlier README; not an award for this revision |
| Three-category support and binary compatibility | Current implementation and synthetic contract tests |
| Lower process peak memory on the controlled workload | Frozen user-run experiment, retained observations and independent recalculation |
| Real classification quality, calibration and camera reliability | Unmeasured |
| Hosted CI results | Recorded per commit in [GitHub Actions](https://github.com/IGoByLotsOfNames/face-mask-detection/actions) |

The public repository preserves its commit history, original licence and selected
reproducibility material. Private working records, local environments, external
face data and generated run directories are not part of the source distribution.
