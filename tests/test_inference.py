import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from mask_detection.data import load_rgb
from mask_detection.inference import Predictor, prepare_bgr
from mask_detection.webcam import run_camera


class InferenceTests(unittest.TestCase):
    def test_bgr_matches_rgb_training_bytes(self):
        rng = np.random.default_rng(9)
        rgb = rng.integers(0, 256, size=(17, 21, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.png"
            Image.fromarray(rgb).save(path)
            np.testing.assert_array_equal(
                load_rgb(path, (12, 14)), prepare_bgr(rgb[..., ::-1], (12, 14))
            )

    def test_invalid_frames(self):
        for frame in (np.zeros((0, 2, 3), dtype=np.uint8), np.zeros((2, 2)), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError):
                prepare_bgr(frame, (10, 10))

    def test_semantic_class_and_probability_mapping(self):
        model = Mock(
            return_value=np.array([[0.2]]), input_shape=(None, 2, 2, 3), output_shape=(None, 1)
        )
        predictor = Predictor(model, dict(image_size=[2, 2], class_names=["mask", "no_mask"]))
        result = predictor.bgr(np.zeros((2, 2, 3), dtype=np.uint8))
        self.assertEqual("mask", result["class_name"])
        self.assertAlmostEqual(0.8, result["probability_predicted_class"])

    def test_bad_output_and_shape_rejected(self):
        model = Mock(
            return_value=np.array([[float("nan")]]),
            input_shape=(None, 2, 2, 3),
            output_shape=(None, 1),
        )
        predictor = Predictor(model, dict(image_size=[2, 2], class_names=["a", "b"]))
        with self.assertRaises(ValueError):
            predictor.bgr(np.zeros((2, 2, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            Predictor(model, dict(image_size=[3, 2], class_names=["a", "b"]))

    def test_camera_released_when_open_fails(self):
        camera, cv = Mock(), Mock()
        camera.isOpened.return_value = False
        with self.assertRaises(RuntimeError):
            run_camera(Mock(), cv=cv, capture=camera)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_camera_released_when_cascade_fails(self):
        camera, cv, cascade = Mock(), Mock(), Mock()
        camera.isOpened.return_value = True
        cascade.empty.return_value = True
        with self.assertRaises(RuntimeError):
            run_camera(Mock(), cv=cv, capture=camera, cascade=cascade)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_camera_released_when_inference_raises(self):
        camera, cv, cascade, predictor = Mock(), Mock(), Mock(), Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, np.zeros((10, 10, 3), dtype=np.uint8))
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = [(0, 0, 5, 5)]
        predictor.bgr.side_effect = RuntimeError("synthetic inference failure")
        with self.assertRaises(RuntimeError):
            run_camera(predictor, cv=cv, capture=camera, cascade=cascade)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_end_of_stream_releases_camera(self):
        camera, cv, cascade = Mock(), Mock(), Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        cascade.empty.return_value = False
        run_camera(Mock(), cv=cv, capture=camera, cascade=cascade)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()

    def test_overlapping_faces_use_unannotated_pixels(self):
        camera, cv, cascade, predictor = Mock(), Mock(), Mock(), Mock()
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, frame), (False, None)]
        cascade.empty.return_value = False
        cascade.detectMultiScale.return_value = [(0, 0, 6, 6), (3, 3, 6, 6)]
        captured = []

        def predict(crop):
            captured.append(crop.copy())
            return dict(class_name="fixture", probability_predicted_class=0.8)

        predictor.bgr.side_effect = predict

        # Simulate a real drawing function modifying the displayed pixels.
        def draw(image, start, end, colour, width):
            image[start[1] : end[1], start[0] : end[0]] = colour

        cv.rectangle.side_effect = draw
        cv.waitKey.return_value = -1
        run_camera(predictor, cv=cv, capture=camera, cascade=cascade)
        self.assertEqual(2, len(captured))
        for crop in captured:
            self.assertEqual((6, 6, 3), crop.shape)
            self.assertEqual(0, np.count_nonzero(crop))
        self.assertGreater(np.count_nonzero(frame), 0)
        camera.release.assert_called_once()
        cv.destroyAllWindows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
