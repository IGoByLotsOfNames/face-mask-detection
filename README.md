# Face Mask Detection

A CNN image-classification project I began during the COVID-19 period, combining Python model training with OpenCV face detection. The maintained version focuses on a reliable path from a labelled face crop to a reproducible prediction.

The original work explored Kaggle images and webcam classification. This repository now records class order with each model, uses the same **RGB preprocessing** for training and OpenCV's BGR camera frames, supports headless image inference, and evaluates a held-out manifest. The older scripts and model files are not presented as validated results.

## How it works

```text
Authorized labelled crops + source-image groups
  → immutable train / validation / test manifest
  → RGB CNN training → model + class metadata
  → headless crop prediction OR optional webcam face detector
  → held-out per-image probabilities and metrics
```

- [Data pipeline](src/mask_detection/data.py): seeded group-aware splitting, exact-duplicate detection and byte-verified manifests.
- [Classifier](src/mask_detection/model.py): a small binary CNN with normalization inside the model.
- [Shared inference](src/mask_detection/inference.py): semantic class labels, hash-checked metadata and matched RGB resizing.
- [Camera adapter](src/mask_detection/webcam.py): clipped face regions and cleanup on stream end or exceptions.
- [Evaluation](src/mask_detection/evaluate.py): sample identities, probabilities, confusion matrix, precision/recall/F1, probability ROC AUC, Brier score and log loss.
- [Tests](tests) and [validation record](docs/validation.md): behaviour checks and an actual synthetic train/save/reload cycle.

## Try the software without a dataset or camera

Use Python 3.12:

```bash
python -m venv .venv
# Activate: Windows .venv\Scripts\activate; macOS/Linux source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

The normal tests require only NumPy/Pillow. They check colour-channel equivalence, label mapping, split leakage, metrics and camera cleanup using fake devices. To run the real CPU TensorFlow smoke test:

```bash
python -m pip install -r requirements-runtime.txt
# PowerShell: $env:RUN_ML_TESTS="1"; $env:TF_ENABLE_ONEDNN_OPTS="0"
# macOS/Linux: export RUN_ML_TESTS=1 TF_ENABLE_ONEDNN_OPTS=0
python -m unittest discover -s tests -v
```

This generates colour fixtures, trains for two epochs, saves and reloads the CNN, runs headless inference and writes temporary evaluation results. It never opens a camera, downloads weights or uses face images. **Passing these checks is not evidence of mask-classification accuracy.**

## Train on authorized data

No dataset or trained model is included. The precise historical dataset version and licence remain unverified. Before training, verify redistribution/use rights, document the two actual classes and prepare source-image/person groups. The code does not assume whether classes mean `mask/no_mask` or `correct/incorrect`: your sorted directory names define index order and the saved metadata preserves it. A binary model cannot classify three categories.

```text
dataset/class_a/image001.png
dataset/class_b/image002.png
```

Supply a CSV covering every image, with `path,group` columns. All crops/augmentations from one source image, and repeated images of a person where applicable to the study, need the same opaque group. At least three independent components per class are required.

```bash
python -m mask_detection.data dataset split-data --groups groups.csv --seed 42
python -m mask_detection.train split-data --epochs 30 --output artifacts/mask.keras
python -m mask_detection.evaluate artifacts/mask.keras split-data --output artifacts/test-report.json
python -m mask_detection.inference artifacts/mask.keras path/to/face-crop.png
```

The headless command classifies an **already cropped face image**; it does not silently detect/crop a full scene. Its output includes the semantic predicted class, probability of class 1, probability of the predicted class and threshold. Probabilities are not calibrated confidence guarantees.

For an interactive camera demonstration after producing an appropriate model:

```bash
python -m mask_detection.webcam artifacts/mask.keras --device 0
```

Press Escape to exit. The model and `.metadata.json` sidecar must stay together. The old `python src/train.py` and `python src/webcam.py` entry points remain wrappers around the package.

## Reproducibility and limits

The splitter never merges into an existing destination. A manifest records the seed, class order, group assignments and hashes. Every training/evaluation run verifies the data; model metadata preserves preprocessing and training/validation hashes, which cannot overlap the test samples. Groups are indivisible, so ratios are approximate. Exact-byte hashing cannot recognize recoded or transformed copies; explicit grouping is essential. `--independent-images` is for known-independent examples such as synthetic fixtures, not unknown identity relationships.

Evaluation exports samples in fixed manifest order, uses probabilities for AUC and marks undefined metrics as null. Choose the threshold using validation data before final testing. Reports refuse overwrite. Seeds and deterministic TensorFlow operations improve replayability but do not guarantee identical training across platforms.

No accuracy, fairness, latency or real-camera reliability result is claimed for the maintained version. Lighting, pose, face-crop quality, camera hardware and mask styles can affect predictions. Haar face detection is a separate stage and may fail even when crop classification works. This is a learning project, **not a safety, identity or compliance system**.

MIT covers the code; [dataset and model rights are separate](LICENSE). Raw images, annotations, personal identifiers, group manifests and model artifacts remain outside Git.
