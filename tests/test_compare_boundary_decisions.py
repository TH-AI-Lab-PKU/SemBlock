import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from compare_boundary_decisions import align_boundary_positions, compute_overlap_metrics


class CompareBoundaryDecisionTests(unittest.TestCase):
    def test_align_boundary_positions_groups_trace_rows_by_sample(self):
        adablock_rows = [
            {"sample_id": "sample-a", "selected_boundary_position": 5},
            {"sample_id": "sample-a", "selected_boundary_position": 10},
            {"sample_id": "sample-b", "selected_boundary_position": 3},
        ]
        semantic_rows = [
            {"sample_id": "sample-a", "selected_boundary_position": 5},
            {"sample_id": "sample-b", "selected_boundary_position": 4},
        ]

        aligned = align_boundary_positions(adablock_rows, semantic_rows)

        self.assertEqual(sorted(aligned.keys()), ["sample-a", "sample-b"])
        self.assertEqual(aligned["sample-a"]["adablock_positions"], [5, 10])
        self.assertEqual(aligned["sample-a"]["semantic_positions"], [5])
        self.assertEqual(aligned["sample-b"]["semantic_positions"], [4])

    def test_compute_overlap_metrics_reports_precision_recall_and_f1(self):
        metrics = compute_overlap_metrics(adablock_positions=[5, 10, 20], semantic_positions=[5, 12, 20])

        self.assertEqual(metrics["exact_overlap"], 2)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertAlmostEqual(metrics["mean_semantic_to_adablock_distance"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
