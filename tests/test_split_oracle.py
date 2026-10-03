"""Compare grouped coverage with direct exhaustive partition enumeration."""

import itertools
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mask_detection.data import (
    CoverageSearchExhausted,
    _coverage_seeds,
    create_split,
    validate_manifest,
)


def exhaustive_assignment(components, class_count):
    """No pruning, masks, search heuristics or representative compression."""
    required = set(range(class_count))
    for assignment in itertools.product(range(3), repeat=len(components)):
        present = [set(), set(), set()]
        for labels, split in zip(components, assignment):
            present[split].update(labels)
        if all(labels == required for labels in present):
            return assignment
    return None


class SplitOracleTests(unittest.TestCase):
    def assert_matches_oracle(self, components, class_count):
        expected = exhaustive_assignment(components, class_count)
        units = [[{"label": label} for label in sorted(labels)] for labels in components]
        try:
            selected, explored = _coverage_seeds(units, class_count)
        except CoverageSearchExhausted as error:
            self.fail(f"Small exhaustive fixture exceeded its search budget: {error}")
        except ValueError:
            self.assertIsNone(expected)
            return
        self.assertIsNotNone(expected)
        self.assertGreater(explored, 0)
        self.assertTrue(set(selected) <= set(range(len(components))))
        present = [set(), set(), set()]
        for index, split in selected.items():
            self.assertIn(split, range(3))
            present[split].update(components[index])
        self.assertEqual([set(range(class_count))] * 3, present)

    def test_all_small_component_multisets_match_exhaustive_assignments(self):
        checked = 0
        for class_count in (2, 3):
            patterns = [
                frozenset(labels)
                for length in range(1, class_count + 1)
                for labels in itertools.combinations(range(class_count), length)
            ]
            for count in range(1, 6):
                for components in itertools.combinations_with_replacement(patterns, count):
                    with self.subTest(classes=class_count, components=components):
                        self.assert_matches_oracle(components, class_count)
                    checked += 1
        self.assertEqual(846, checked)
        # More than three interchangeable patterns and a fourth class exercise
        # representation compression and generalized class coverage independently.
        extra = [
            ([frozenset(range(3))] * 7, 3),
            ([frozenset(set(range(4)) - {missing}) for missing in range(4)], 4),
            (
                [frozenset(set(range(4)) - {missing}) for missing in range(4)]
                + [frozenset(range(4))],
                4,
            ),
        ]
        for components, class_count in extra:
            with self.subTest(classes=class_count, components=components):
                self.assert_matches_oracle(components, class_count)

    def test_actual_split_matches_oracle_for_feasible_and_infeasible_groups(self):
        cases = [
            (({0}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}), 3),
            (({0}, {0}, {0}, {1}, {1}, {1}), 2),
            (tuple(set(range(4)) - {missing} for missing in range(4)), 4),
        ]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent) as temp:
            root = Path(temp)
            for case, (components, class_count) in enumerate(cases):
                source = root / f"source-{case}"
                groups = {}
                for label in range(class_count):
                    directory = source / f"class_{label}"
                    directory.mkdir(parents=True)
                    for index, labels in enumerate(components):
                        if label not in labels:
                            continue
                        path = directory / f"group_{index}.png"
                        Image.new("RGB", (3, 3), (label * 40, index * 35, case * 50)).save(path)
                        groups[path.relative_to(source).as_posix()] = f"group-{index}"
                expected = exhaustive_assignment(components, class_count)
                for seed in (0, 42):
                    destination = root / f"split-{case}-{seed}"
                    with self.subTest(case=case, seed=seed):
                        if expected is None:
                            with self.assertRaisesRegex(ValueError, "No class-complete"):
                                create_split(source, destination, groups=groups, seed=seed)
                            self.assertFalse(destination.exists())
                            continue
                        manifest = create_split(source, destination, groups=groups, seed=seed)
                        self.assertEqual(manifest, validate_manifest(destination))
                        self.assertEqual(sum(map(len, components)), len(manifest["rows"]))
                        self.assertEqual(
                            {
                                (split, label)
                                for split in ("train", "validation", "test")
                                for label in range(class_count)
                            },
                            {(row["split"], row["label"]) for row in manifest["rows"]},
                        )
                        for group in set(groups.values()):
                            self.assertEqual(
                                1,
                                len(
                                    {
                                        row["split"]
                                        for row in manifest["rows"]
                                        if row["group"] == group
                                    }
                                ),
                            )


if __name__ == "__main__":
    unittest.main()
