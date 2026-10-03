import base64
import copy
import unittest
from html.parser import HTMLParser

from mask_detection.demo_report import render_report

_PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfixture").decode()


def fixture():
    classes = ["mask_weared_incorrect", "with_mask", "without_mask"]
    return {
        "schema_version": 1,
        "kind": "synthetic-software-demo",
        "title": "Face Mask Detection · Pipeline demo",
        "notice": "Synthetic illustrations and scripted predictions; no trained model.",
        "class_names": classes,
        "counts": {
            "source_images": 1200,
            "source_objects": 1400,
            "crops": 1100,
            "quarantined_source_images": 2,
            "excluded_objects": 3,
            "duplicate_groups": 1,
        },
        "split_counts": {"train": 800, "validation": 150, "test": 150},
        "checks": [{"name": "Group separation", "status": "passed", "detail": "No overlap."}],
        "confusion_matrix": [[4, 1, 0], [0, 5, 0], [1, 0, 4]],
        "per_class": [
            {"name": name, "precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5}
            for name in classes
        ],
        "samples": [
            {
                "source_image": "fixture-01.png",
                "object_index": 0,
                "class_name": "with_mask",
                "split": "test",
                "probabilities": [0.1, 0.8, 0.1],
                "predicted_class": "with_mask",
                "image_data_uri": _PNG,
            }
        ],
        "run_summary_relative": "summary.json",
    }


class Elements(HTMLParser):
    def __init__(self, markup):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.text = []
        self.feed(markup)

    def handle_starttag(self, tag, attributes):
        self.tags.append((tag, dict(attributes)))

    def handle_data(self, data):
        self.text.append(data)


class DemoReportTests(unittest.TestCase):
    def test_offline_accessible_report_preserves_meaning(self):
        markup = render_report(fixture())
        parsed = Elements(markup)
        text = " ".join(parsed.text)
        for label in ("Mask worn incorrectly", "Mask worn", "No mask"):
            self.assertIn(label, text)
        self.assertIn("1,200", text)
        self.assertIn("0.800", text)
        self.assertIn("Synthetic illustrations and scripted predictions", text)
        self.assertIn("Rows: fixture label · columns: scripted output", text)
        self.assertEqual(sum(tag == "table" for tag, _ in parsed.tags), 2)
        self.assertTrue(
            any(tag == "th" and attrs.get("scope") == "row" for tag, attrs in parsed.tags)
        )
        self.assertTrue(any(tag == "img" and attrs.get("alt") for tag, attrs in parsed.tags))
        self.assertFalse(any(tag in ("link", "iframe") for tag, _ in parsed.tags))
        for tag, attributes in parsed.tags:
            if "src" in attributes:
                self.assertEqual(tag, "img")
                self.assertTrue(attributes["src"].startswith("data:image/png;base64,"))
        self.assertIn("@media print", markup)

    def test_untrusted_text_is_text_and_not_active_markup(self):
        data = fixture()
        attack = '<img src=x onerror="alert(1)"><script>alert(2)</script>'
        data["title"] = attack
        data["notice"] = attack
        data["checks"][0]["name"] = attack
        data["checks"][0]["detail"] = attack
        data["samples"][0]["source_image"] = attack
        data["samples"][0]["split"] = 'test" onmouseover="alert(3)'
        markup = render_report(data)
        parsed = Elements(markup)
        self.assertNotIn(attack, markup)
        self.assertIn(attack, " ".join(parsed.text))
        self.assertEqual(sum(tag == "script" for tag, _ in parsed.tags), 1)
        self.assertEqual(sum(tag == "img" for tag, _ in parsed.tags), 1)
        self.assertFalse(any(key.startswith("on") for _, attrs in parsed.tags for key in attrs))

    def test_rejects_external_or_malformed_images_and_summary_links(self):
        invalid_images = [
            "https://example.com/picture.png",
            "data:image/svg+xml;base64,PHN2Zz4=",
            "data:image/png;base64,!!!!",
            "data:image/png;base64," + base64.b64encode(b"not a png").decode(),
        ]
        for image in invalid_images:
            with self.subTest(image=image):
                data = fixture()
                data["samples"][0]["image_data_uri"] = image
                with self.assertRaises(ValueError):
                    render_report(data)
        data = fixture()
        data["run_summary_relative"] = "javascript:alert(1)"
        with self.assertRaises(ValueError):
            render_report(data)

    def test_rejects_misleading_numeric_or_schema_values(self):
        cases = []
        for value in (float("nan"), float("inf"), -0.1, 1.1, True, "0.8"):
            data = fixture()
            data["per_class"][0]["f1"] = value
            cases.append(data)
        for value in (-1, 2.5, True, "2"):
            data = fixture()
            data["counts"]["crops"] = value
            cases.append(data)
        data = fixture()
        data["checks"][0]["status"] = "failed"
        cases.append(data)
        data = fixture()
        data["kind"] = "real-inference"
        cases.append(data)
        data = fixture()
        data["confusion_matrix"] = [[1]]
        cases.append(data)
        for data in cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    render_report(data)

    def test_render_is_deterministic_and_does_not_mutate_summary(self):
        data = fixture()
        original = copy.deepcopy(data)
        self.assertEqual(render_report(data), render_report(data))
        self.assertEqual(data, original)


if __name__ == "__main__":
    unittest.main()
