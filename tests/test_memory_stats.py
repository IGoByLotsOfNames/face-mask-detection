"""Independent arithmetic and design-integrity checks; no benchmark is executed."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import random
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "input_memory_stats.py"
SPEC = importlib.util.spec_from_file_location("input_memory_stats_for_tests", TOOL)
assert SPEC is not None and SPEC.loader is not None
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)
summarize = STATS.summarize
MIB = 1048576


def fixture():
    plan = {"cases": [256, 1024, 4096], "seeds": [17, 29, 43, 71, 101]}
    records = []
    for case_index, rows in enumerate(plan["cases"]):
        factor = 2**case_index
        for seed_index, seed in enumerate(plan["seeds"]):
            for implementation, amount in (("baseline", 100), ("indexed", 90)):
                records.append(
                    dict(
                        schema_version=1,
                        rows=rows,
                        seed=seed,
                        implementation=implementation,
                        metric="process_lifetime_peak_rss_bytes",
                        peak_rss_bytes=(seed_index + 1) * amount * factor * MIB,
                        metric_method="synthetic fixture method, not a measurement",
                        plan_sha256="ab" * 32,
                    )
                )
    return plan, records


class MemorySummaryTests(unittest.TestCase):
    def test_known_five_process_sample_statistics_for_each_case(self):
        plan, records = fixture()
        report = summarize(plan, records)
        self.assertEqual(30, len(records))
        self.assertEqual(3, len(report["cases"]))
        self.assertEqual("ab" * 32, report["plan_sha256"])
        for case_index, case in enumerate(report["cases"]):
            factor = 2**case_index
            self.assertEqual(plan["cases"][case_index], case["rows"])
            self.assertEqual(plan["seeds"], case["seeds"])
            for name, multiplier in (("baseline", 1), ("indexed", 0.9)):
                result = case["by_implementation"][name]
                scale = factor * multiplier
                self.assertEqual(5, result["count"])
                self.assertEqual(1, result["ddof"])
                self.assertEqual("MiB", result["unit"])
                self.assertAlmostEqual(300 * scale, result["mean"])
                self.assertAlmostEqual(300 * scale, result["median"])
                # Deviations [-200,-100,0,100,200]: 100000/(5-1)=25000.
                self.assertAlmostEqual(25000 * scale**2, result["sample_variance"])
                self.assertAlmostEqual(math.sqrt(25000) * scale, result["sample_sd"])
                self.assertAlmostEqual(100 * scale, result["min"])
                self.assertAlmostEqual(500 * scale, result["max"])
            delta = case["paired_indexed_minus_baseline"]
            self.assertAlmostEqual(-30 * factor, delta["mean"])
            self.assertAlmostEqual(-30 * factor, delta["median"])
            self.assertAlmostEqual(250 * factor**2, delta["sample_variance"])
            self.assertAlmostEqual(math.sqrt(250) * factor, delta["sample_sd"])
            self.assertEqual(-50 * factor, delta["min"])
            self.assertEqual(-10 * factor, delta["max"])
            ratio = case["paired_indexed_over_baseline"]
            self.assertAlmostEqual(0.9, ratio["mean"])
            self.assertEqual(0, ratio["sample_variance"])
            self.assertEqual("ratio", ratio["unit"])
        json.dumps(report, allow_nan=False)

    def test_even_median_and_pairing_use_seed_not_record_position(self):
        plan, original = fixture()
        plan = {"cases": [256], "seeds": [17, 29]}
        records = [
            record
            for record in original
            if record["rows"] == 256 and record["seed"] in plan["seeds"]
        ]
        amounts = {
            (17, "baseline"): 10,
            (29, "baseline"): 30,
            (17, "indexed"): 20,
            (29, "indexed"): 15,
        }
        for record in records:
            record["peak_rss_bytes"] = amounts[record["seed"], record["implementation"]] * MIB
        case = summarize(plan, list(reversed(records)))["cases"][0]
        self.assertEqual(20, case["by_implementation"]["baseline"]["median"])
        # Per-pair ratios are 2 and .5: mean 1.25, not mean(20,15)/mean(10,30).
        self.assertEqual(1.25, case["paired_indexed_over_baseline"]["mean"])
        self.assertEqual(1.125, case["paired_indexed_over_baseline"]["sample_variance"])
        self.assertEqual(-2.5, case["paired_indexed_minus_baseline"]["mean"])
        self.assertEqual(312.5, case["paired_indexed_minus_baseline"]["sample_variance"])

    def test_record_permutation_preserves_summary_and_does_not_mutate_inputs(self):
        plan, records = fixture()
        before = copy.deepcopy((plan, records))
        expected = summarize(plan, records)
        shuffled = list(records)
        random.Random(1729).shuffle(shuffled)
        self.assertEqual(expected, summarize(plan, iter(shuffled)))
        self.assertEqual(before, (plan, records))

    def test_declared_case_and_seed_order_preserved_without_pooling(self):
        plan, records = fixture()
        plan["cases"].reverse()
        plan["seeds"].reverse()
        report = summarize(plan, records)
        self.assertEqual(plan["cases"], [case["rows"] for case in report["cases"]])
        self.assertTrue(all(case["seeds"] == plan["seeds"] for case in report["cases"]))
        self.assertNotIn("overall", report)

    def test_missing_duplicate_and_unknown_slots_rejected(self):
        plan, records = fixture()
        bad_record = dict(records[0], rows=2048)
        variants = (records[:-1], records + [records[0]], records[1:] + [bad_record], [])
        for invalid in variants:
            with self.subTest(count=len(invalid)):
                with self.assertRaises(ValueError):
                    summarize(plan, invalid)

    def test_invalid_peak_values_and_identity_types_rejected(self):
        plan, records = fixture()
        variants = [
            ("peak_rss_bytes", value)
            for value in (
                True,
                False,
                0,
                -1,
                1.0,
                "1",
                None,
                float("nan"),
                float("inf"),
                -float("inf"),
                10**1000,
            )
        ] + [
            ("rows", True),
            ("rows", 256.0),
            ("seed", True),
            ("seed", 17.0),
            ("implementation", "other"),
            ("implementation", []),
            ("schema_version", True),
            ("schema_version", 2),
            ("metric", "current_rss_bytes"),
        ]
        for key, value in variants:
            changed = copy.deepcopy(records)
            changed[0][key] = value
            with self.subTest(key=key, value=repr(value)):
                with self.assertRaises(ValueError):
                    summarize(plan, changed)

    def test_missing_required_record_fields_rejected(self):
        plan, records = fixture()
        for key in records[0]:
            changed = copy.deepcopy(records)
            del changed[0][key]
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    summarize(plan, changed)

    def test_inconsistent_or_malformed_method_and_plan_hash_rejected(self):
        plan, records = fixture()
        for key, value in (
            ("metric_method", "another method"),
            ("metric_method", ""),
            ("metric_method", "  "),
            ("metric_method", None),
            ("plan_sha256", "cd" * 32),
            ("plan_sha256", "g" * 64),
            ("plan_sha256", "a" * 63),
            ("plan_sha256", None),
        ):
            changed = copy.deepcopy(records)
            changed[-1][key] = value
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    summarize(plan, changed)

    def test_malformed_plans_and_records_rejected(self):
        plan, records = fixture()
        invalid_plans = (
            None,
            {},
            {"cases": [], "seeds": [17, 29]},
            {"cases": [256, 256], "seeds": [17, 29]},
            {"cases": [True], "seeds": [17, 29]},
            {"cases": [0], "seeds": [17, 29]},
            {"cases": [256], "seeds": [17]},
            {"cases": [256], "seeds": [17, 17]},
            {"cases": [256], "seeds": [True, 29]},
            {"cases": [256], "seeds": [-1, 29]},
        )
        for invalid in invalid_plans:
            with self.subTest(plan=invalid):
                with self.assertRaises(ValueError):
                    summarize(invalid, records)
        for invalid in (None, "records", {}, [None]):
            with self.subTest(records=invalid):
                with self.assertRaises(ValueError):
                    summarize(plan, invalid)


if __name__ == "__main__":
    unittest.main()
