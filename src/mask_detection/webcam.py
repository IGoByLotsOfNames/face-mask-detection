"""Optional interactive adapter. No webcam is opened by importing this module."""

import argparse
from pathlib import Path

from .inference import Predictor


def run_camera(predictor, *, device=0, cv=None, capture=None, cascade=None):
    if cv is None:
        import cv2 as cv
    camera = capture if capture is not None else cv.VideoCapture(device)
    try:
        if not camera.isOpened():
            raise RuntimeError("Unable to open camera")
        detector = (
            cascade
            if cascade is not None
            else cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
        )
        if detector.empty():
            raise RuntimeError("Face cascade could not be loaded")
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            # An earlier face's overlay must not enter another face's crop.
            inference_frame = frame.copy()
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            for x, y, width, height in detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5
            ):
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(frame.shape[1], x + width), min(frame.shape[0], y + height)
                if x2 <= x1 or y2 <= y1:
                    continue
                result = predictor.bgr(inference_frame[y1:y2, x1:x2])
                # Distinct annotation colours identify classes, not safety/compliance.
                palette = ((230, 180, 80), (180, 100, 225), (80, 190, 230))
                index = result.get("class_index", 0)
                colour = palette[index] if type(index) is int and 0 <= index < 3 else palette[0]
                cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                cv.putText(
                    frame,
                    f"{result['class_name']}: {result['probability_predicted_class']:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                )
            cv.imshow("Mask image classifier", frame)
            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        cv.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    run_camera(Predictor.load(args.model), device=args.device)


if __name__ == "__main__":
    main()
