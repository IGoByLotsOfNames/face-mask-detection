# Software validation

Validated on 2 October 2026, Windows x64 CPU, Python 3.12.14. Runtime: TensorFlow 2.20.0, Keras 3.15.1, OpenCV 4.12.0, NumPy 2.2.6 and Pillow 12.0.0. `requirements-runtime.txt` records installed versions.

| Check | Observed result |
|---|---|
| Core suite | 27 tests passed with TensorFlow/OpenCV imports explicitly blocked; no model or camera required |
| Runtime integration | Passed before the camera-only correction; the earlier full run passed 27 tests, including TensorFlow integration |
| Preprocessing | Saved RGB image and its in-memory BGR frame produce identical resized float32 input arrays |
| Training integration | 12 generated RGB images in two classes; two completed epochs; class order and preprocessing saved alongside the model |
| Serialization/inference | Reloaded predictions agree within rtol 1e-5 / atol 1e-6; headless prediction returns a recorded semantic class |
| Evaluation | Manifest-aligned per-image predictions, individual/mean-ensemble reports and known-overlap rejection |
| Resource cleanup | Fake camera tests verify release/close after open failure, cascade failure, inference exception and normal stream end |
| Overlapping detections | A regression deliberately mutates the display frame while drawing; both overlapping inference crops retain the untouched input pixels |

Other checks cover deterministic grouped splits, connected duplicate handling, conflicting labels, immutable outputs, changed/missing/extra samples, metadata tampering, image shapes, invalid probabilities, undefined metrics and 50 pairwise-AUC oracle comparisons. Core tests deliberately need no TensorFlow or OpenCV import. The optional runtime uses actual TensorFlow, not a mocked neural network.

The final suite contains 28 tests: 27 core and one optional runtime integration test. After the overlapping-crop correction, the core suite was rerun; training/evaluation code was unchanged.

CI repeats core tests on Python 3.11/3.12 and runtime tests on Linux/Windows with Python 3.12. Remote CI status must be checked in the workflow; this record reports local execution only.

Keras emits upstream serialization/reset deprecation messages, and TensorFlow may log end-of-sequence notices. The final run completed both requested epochs and passed all tests.

No real camera was opened. No face data, pretrained weights or historical models were loaded. Mask accuracy, face-detector recall, lighting/pose robustness, subgroup performance, latency and calibration remain unmeasured. Generated colour-image metrics are software fixtures, not mask-detection results. The exact historical dataset's rights and class semantics must be verified before retraining.
