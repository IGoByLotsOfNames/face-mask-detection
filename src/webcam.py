from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mask-wearing inference on a webcam stream")
    parser.add_argument("model", type=Path)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Unable to open the default camera")

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for x, y, width, height in cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5):
            face = frame[y : y + height, x : x + width]
            resized = cv2.resize(face, (128, 128)).astype(np.float32)
            probability = float(model.predict(resized[None, ...], verbose=0)[0][0])
            label = "Class 1" if probability >= 0.5 else "Class 0"
            colour = (0, 180, 0) if probability >= 0.5 else (0, 100, 255)
            cv2.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
            cv2.putText(frame, f"{label}: {probability:.2f}", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        cv2.imshow("Mask-wearing classifier", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

