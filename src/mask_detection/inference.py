"""One RGB preprocessing path for cropped images and OpenCV BGR frames."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .artifacts import load_metadata, validate_model
from .data import load_rgb
from .metrics import validate_probabilities


def prepare_bgr(frame, size):
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size or frame.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 BGR frame")
    rgb = Image.fromarray(frame[..., ::-1])
    return np.asarray(rgb.resize((size[1], size[0]), Image.Resampling.BILINEAR), dtype=np.float32)


class Predictor:
    def __init__(self, model, metadata):
        validate_model(model, metadata)
        self.model, self.metadata = model, metadata
        self.size = tuple(metadata["image_size"])

    @classmethod
    def load(cls, path):
        import tensorflow as tf
        metadata = load_metadata(path)
        return cls(tf.keras.models.load_model(path, compile=False), metadata)

    def predict_rgb(self, image):
        if image.shape != (*self.size, 3) or not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
            raise ValueError("RGB image violates model input contract")
        output = np.asarray(self.model(image[None, ...], training=False))
        if output.shape != (1, 1):
            raise ValueError("Expected one binary probability")
        p = validate_probabilities(output.reshape(-1))[0]
        index = int(p >= .5)
        return dict(class_index=index, class_name=self.metadata["class_names"][index],
                    probability_class_1=p, probability_predicted_class=p if index else 1-p,
                    threshold=.5)

    def image(self, path):
        return self.predict_rgb(load_rgb(path, self.size))

    def bgr(self, frame):
        return self.predict_rgb(prepare_bgr(frame, self.size))


def main():
    parser = argparse.ArgumentParser(description="Classify a face crop without opening a camera")
    parser.add_argument("model", type=Path); parser.add_argument("image", type=Path)
    args = parser.parse_args()
    print(json.dumps(Predictor.load(args.model).image(args.image), indent=2))


if __name__ == "__main__":
    main()
