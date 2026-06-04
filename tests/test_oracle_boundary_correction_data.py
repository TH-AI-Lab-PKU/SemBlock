import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from oracle_boundary_correction_data import (
    build_boundary_correction_record,
    choose_best_delta_label,
    classify_keep_vs_adjust,
)


class OracleBoundaryCorrectionDataTests(unittest.TestCase):
    def test_choose_best_delta_label_prefers_smaller_absolute_shift_on_tie(self):
        delta_scores = {2: 0.9, -1: 0.9, 0: 0.1}
        self.assertEqual(choose_best_delta_label(delta_scores), -1)

    def test_choose_best_delta_label_prefers_smaller_signed_delta_on_equal_abs(self):
        delta_scores = {1: 0.75, -1: 0.75}
        self.assertEqual(choose_best_delta_label(delta_scores), -1)

    def test_choose_best_delta_label_accepts_string_keys(self):
        delta_scores = {"2": 0.9, "-1": 0.9}
        self.assertEqual(choose_best_delta_label(delta_scores), -1)

    def test_choose_best_delta_label_rejects_empty_scores(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({})

    def test_choose_best_delta_label_rejects_duplicate_normalized_keys(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({"1": 0.3, 1: 0.4})

    def test_choose_best_delta_label_rejects_invalid_delta_key(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({1.9: 0.3, 0: 0.1})

    def test_choose_best_delta_label_rejects_non_finite_score(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({0: float("nan")})

    def test_choose_best_delta_label_rejects_overflow_score(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({0: 10**1000})

    def test_choose_best_delta_label_rejects_invalid_score_type(self):
        with self.assertRaises(ValueError):
            choose_best_delta_label({0: None})

    def test_classify_keep_vs_adjust(self):
        self.assertEqual(classify_keep_vs_adjust(0), "keep")
        self.assertEqual(classify_keep_vs_adjust(2), "adjust")

    def test_classify_keep_vs_adjust_rejects_invalid_delta(self):
        with self.assertRaises(ValueError):
            classify_keep_vs_adjust("oops")

    def test_build_boundary_correction_record_normalizes_types(self):
        record = build_boundary_correction_record(
            sample_id="gsm8k/test/19",
            boundary_index="3",
            best_delta="0",
            delta_scores={"0": 1, 2: 0.5},
        )
        self.assertEqual(record["sample_id"], "gsm8k/test/19")
        self.assertEqual(record["boundary_index"], 3)
        self.assertEqual(record["best_delta"], 0)
        self.assertEqual(record["gate_label"], "keep")
        self.assertEqual(record["delta_scores"], {0: 1.0, 2: 0.5})

    def test_build_boundary_correction_record_rejects_duplicate_normalized_keys(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=1,
                delta_scores={"1": 0.3, 1: 0.4},
            )

    def test_build_boundary_correction_record_requires_best_delta(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=2,
                delta_scores={"1": 0.3},
            )

    def test_build_boundary_correction_record_rejects_invalid_best_delta(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=True,
                delta_scores={"0": 0.3},
            )

    def test_build_boundary_correction_record_rejects_invalid_boundary_index(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1.9,
                best_delta=0,
                delta_scores={"0": 0.3},
            )

    def test_build_boundary_correction_record_rejects_negative_boundary_index(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=-1,
                best_delta=0,
                delta_scores={"0": 0.3},
            )

    def test_build_boundary_correction_record_rejects_boolean_score(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=0,
                delta_scores={"0": True},
            )

    def test_build_boundary_correction_record_rejects_non_finite_score(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=0,
                delta_scores={"0": float("inf")},
            )

    def test_build_boundary_correction_record_rejects_non_mapping_delta_scores(self):
        with self.assertRaises(ValueError):
            build_boundary_correction_record(
                sample_id="gsm8k/test/19",
                boundary_index=1,
                best_delta=0,
                delta_scores=[("0", 0.2)],
            )


if __name__ == "__main__":
    unittest.main()
