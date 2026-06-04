import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

import eval_math_oracle as emo


class OracleLocalCorrectionEvalTests(unittest.TestCase):
    def test_apply_local_boundary_correction_applies_deltas_and_clamps(self):
        doc = {"oracle_block_sizes": [5, 1, 3]}
        predicted_deltas = {0: -2, 1: -2, 2: 2}

        corrected = emo.apply_local_boundary_correction(doc, predicted_deltas)

        self.assertEqual(corrected, [3, 1, 5])

    def test_apply_local_boundary_correction_handles_string_keys(self):
        doc = {"oracle_block_sizes": [2, 5]}
        predicted_deltas = {"1": -2}

        corrected = emo.apply_local_boundary_correction(doc, predicted_deltas)

        self.assertEqual(corrected, [2, 3])

    def test_apply_local_boundary_correction_rejects_falsy_non_mapping(self):
        doc = {"oracle_block_sizes": [2, 5]}

        with self.assertRaises(ValueError):
            emo.apply_local_boundary_correction(doc, [])

    def test_apply_local_boundary_correction_rejects_non_mapping(self):
        doc = {"oracle_block_sizes": [2, 5]}

        with self.assertRaises(ValueError):
            emo.apply_local_boundary_correction(doc, ["bad"])

    def test_apply_local_boundary_correction_rejects_out_of_range_indices(self):
        doc = {"oracle_block_sizes": [4, 2]}
        predicted_deltas = {5: 1}

        corrected = emo.apply_local_boundary_correction(doc, predicted_deltas)

        self.assertEqual(corrected, [4, 2])

    def test_apply_local_boundary_correction_rejects_out_of_window_delta(self):
        doc = {"oracle_block_sizes": [2, 5]}

        with self.assertRaises(ValueError):
            emo.apply_local_boundary_correction(doc, {1: 3})


if __name__ == "__main__":
    unittest.main()
