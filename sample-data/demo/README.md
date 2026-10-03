# Included schematic software fixture

These images are original illustrations drawn with Pillow shapes. They depict
no real people and contain no source-dataset photographs. They test software
behaviour; they are not evidence of mask recognition performance.

`raw/images` contains nine distinct three-head scenes, one exact duplicate of
scene-00, and two same-byte sources with conflicting annotations. Scene-08 also
has one deliberately out-of-bounds object. The annotations explicitly use
**zero-based, half-open** boxes: `[xmin, ymin, xmax, ymax)`.

The real preparation code must keep 27 crops (nine per documented class), retain
scene-00 as the exact-duplicate representative, quarantine both conflicting
sources, and reject the out-of-bounds object. Every retained crop has distinct
bytes. A source-image SHA-256 is the grouping unit; it is not a person identity.

`predictions.json` contains 27 hand-authored probability vectors, keyed by
`source_image` and `object_index`. These vectors are **synthetic, not model
output**. For scenes 00, 03 and 06 the predicted class deliberately rotates to
the next class. Other scenes predict the annotated class. Each vector assigns
0.8 to its selected class and 0.1 to the other classes. The errors exercise the
confusion matrix and per-class report; their scores must never be reported as
model accuracy. All vectors use this exact class order:

1. `mask_weared_incorrect`
2. `with_mask`
3. `without_mask`

`manifest.json` lists SHA-256 hashes for every raw input, `predictions.json`,
and `expected.json`. Paths are relative to this directory. The manifest itself
and this README are outside its hash registry. `expected.json` holds independently
specified counts and the all-27 confusion matrix; these are fixture expectations,
not estimates learned from the images. Split-specific results depend on the
real grouped splitter and are calculated by the demo from the retained rows.

To regenerate into a NEW directory (Pillow required):

```powershell
# Repository working directory
python tools/generate_demo_samples.py --output sample-data/demo-new
```

Generation never overwrites an existing directory. Pixel determinism is tested
in the installed environment; a different Pillow version may encode PNG bytes
differently. The included files and their hashes are the executable fixture.
