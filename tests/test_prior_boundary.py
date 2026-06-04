import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from prior_boundary import choose_boundary_length_from_scores


class PriorBoundaryTests(unittest.TestCase):
    def test_choose_boundary_length_uses_best_index_when_above_threshold(self):
        self.assertEqual(
            choose_boundary_length_from_scores([0.1, 0.92, 0.4], default_block_length=5, threshold=0.5),
            2,
        )

    def test_choose_boundary_length_falls_back_when_below_threshold(self):
        self.assertEqual(
            choose_boundary_length_from_scores([0.1, 0.2, 0.4], default_block_length=5, threshold=0.8),
            5,
        )


if __name__ == "__main__":
    unittest.main()
