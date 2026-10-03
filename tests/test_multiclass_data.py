"""Regression checks for decoded-image preflight and multiclass grouping."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from mask_detection.data import (
    CoverageSearchExhausted,
    create_split,
    digest,
    inspect_image,
    validate_manifest,
)


class MulticlassDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.groups = {}
        for label, name in enumerate(("with_mask", "without_mask", "mask_weared_incorrect")):
            (self.source / name).mkdir(parents=True)
            for index in range(9):
                path = self.source / name / f"{index}.png"
                Image.new("RGB", (8, 7), (label * 60, index * 20, 30)).save(path)
                self.groups[path.relative_to(self.source).as_posix()] = f"photo-{index}"

    def test_three_classes_keep_mixed_label_photos_together(self):
        before = {p: digest(p) for p in self.source.rglob("*.png")}
        manifest = create_split(self.source, self.root / "split", groups=self.groups)
        self.assertEqual(2, manifest["schema_version"])
        self.assertEqual(
            sorted(("with_mask", "without_mask", "mask_weared_incorrect")), manifest["class_names"]
        )
        self.assertEqual(manifest, validate_manifest(self.root / "split"))
        for group in set(self.groups.values()):
            self.assertEqual(1, len({r["split"] for r in manifest["rows"] if r["group"] == group}))
        self.assertEqual(
            {(s, c) for s in ("train", "validation", "test") for c in range(3)},
            {(r["split"], r["label"]) for r in manifest["rows"]},
        )
        self.assertEqual(before, {p: digest(p) for p in self.source.rglob("*.png")})

    def grouped_fixture(self, vectors):
        source = self.root / "coverage-source"
        groups = {}
        for label in range(len(vectors[0])):
            name = f"class_{label}"
            (source / name).mkdir(parents=True)
            for group, vector in enumerate(vectors):
                if vector[label]:
                    path = source / name / f"g{group}.png"
                    Image.new("RGB", (4, 4), (label * 40, group * 40, 42)).save(path)
                    groups[path.relative_to(source).as_posix()] = f"g{group}"
        return source, groups

    def test_feasible_mixed_groups_repaired_after_greedy_dead_end(self):
        # Greedy seed 42 misses class 1 in test. A feasible partition is
        # {g0,g3}, {g1,g2}, {g4}; no group or duplicate needs to be split.
        vectors = ((1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
        source, groups = self.grouped_fixture(vectors)
        before = {p: digest(p) for p in source.rglob("*.png")}
        first = create_split(source, self.root / "repaired", groups=groups, seed=42)
        repeated = create_split(source, self.root / "repeated", groups=groups, seed=42)
        self.assertEqual(first, repeated)
        self.assertEqual(first, validate_manifest(self.root / "repaired"))
        self.assertEqual(
            "deterministic_class_mask_search_then_greedy", first["coverage_repair"]["method"]
        )
        self.assertEqual(sum(sum(v) for v in vectors), len(first["rows"]))
        self.assertEqual(
            {(s, c) for s in ("train", "validation", "test") for c in range(3)},
            {(r["split"], r["label"]) for r in first["rows"]},
        )
        for group in set(groups.values()):
            self.assertEqual(1, len({r["split"] for r in first["rows"] if r["group"] == group}))
        self.assertEqual(before, {p: digest(p) for p in source.rglob("*.png")})

    def test_four_class_infeasibility_is_proved_without_publishing(self):
        # Each class appears in three components, but every component misses one
        # class. With four components and three splits, at least two splits have
        # only one component and therefore cannot cover all four classes.
        source, groups = self.grouped_fixture(
            ((0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0))
        )
        with self.assertRaisesRegex(ValueError, "No class-complete three-way partition exists"):
            create_split(source, self.root / "impossible", groups=groups)
        self.assertFalse((self.root / "impossible").exists())

    def test_exhausted_coverage_budget_does_not_claim_infeasibility(self):
        source, groups = self.grouped_fixture(
            ((1, 0, 0), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
        )
        with patch("mask_detection.data.COVERAGE_SEARCH_MAX_STATES", 1):
            with self.assertRaisesRegex(
                CoverageSearchExhausted, "feasibility remains undetermined"
            ):
                create_split(source, self.root / "exhausted", groups=groups)
        self.assertFalse((self.root / "exhausted").exists())

    def test_corrupt_supported_extension_rejected_before_publication(self):
        path = self.source / "with_mask/0.png"
        path.write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            create_split(self.source, self.root / "split", groups=self.groups)
        self.assertFalse((self.root / "split").exists())
        self.assertEqual(b"not an image", path.read_bytes())

    def test_truncated_decodable_header_is_not_accepted(self):
        path = self.source / "with_mask/0.png"
        data = path.read_bytes()
        path.write_bytes(data[:40])
        with self.assertRaises(ValueError):
            inspect_image(path)

    def test_forged_dimensions_and_label_rejected(self):
        manifest = create_split(self.source, self.root / "split", groups=self.groups)
        path = self.root / "split/manifest.json"
        manifest["rows"][0]["image_dimensions"]["width"] = 99
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "dimension"):
            validate_manifest(self.root / "split")
        manifest["rows"][0]["label"] = 3
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "class index"):
            validate_manifest(self.root / "split")

    def test_rare_class_requires_independent_groups(self):
        for relative in self.groups:
            if relative.startswith("mask_weared_incorrect/"):
                self.groups[relative] = "one-rare-photo"
        with self.assertRaisesRegex(ValueError, "three independent"):
            create_split(self.source, self.root / "split", groups=self.groups)
        self.assertFalse((self.root / "split").exists())


if __name__ == "__main__":
    unittest.main()
