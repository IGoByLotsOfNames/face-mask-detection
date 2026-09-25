# Face Mask Detection

This project revisits a computer-vision prototype created during the COVID-19 period. The original experiment trained a CNN on cropped face images and combined it with OpenCV face detection for webcam inference.

This public edition separates model training from inference, removes datasets and serialized models, and labels outputs neutrally until the exact class-to-index mapping is recorded during retraining.

## What it demonstrates

- TensorFlow/Keras CNN training
- Binary image classification
- OpenCV face detection and webcam inference
- Clear separation between training and application code

## Data

The historical archive contains thousands of images and annotations, but they are not included here. Before publication:

1. Confirm the original dataset and its licence.
2. Document the two classes and their index order.
3. Recreate the train/validation split with a recorded seed.
4. Record accuracy, precision, recall, F1 and a confusion matrix on a held-out test set.

## Train

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python src/train.py path/to/dataset
```

The dataset directory must contain one child directory per class.

## Run webcam inference

```bash
python src/webcam.py artifacts/mask-classifier.keras
```

Press Escape to close the camera window.

## Limitations

- This is a historical learning project, not a safety or compliance system.
- Face detection and classification can vary with lighting, pose, camera and mask style.
- The public version intentionally does not claim a result until the cleaned pipeline is rerun.
- No demographic or device-level subgroup evaluation is available.

## Licence

MIT for the code. Dataset and model rights are separate and are not granted here.

