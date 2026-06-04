import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from oracle_boundary_candidates import (
    build_candidate_boundary_offsets,
    build_candidate_boundary_points,
)


class OracleBoundaryCandidateTests(unittest.TestCase):
    def test_build_candidate_boundary_offsets_is_symmetric_and_includes_zero(self):
        self.assertEqual(build_candidate_boundary_offsets(radius=2), [-2, -1, 0, 1, 2])

    def test_build_candidate_boundary_offsets_handles_negative_radius(self):
        self.assertEqual(build_candidate_boundary_offsets(radius=-3), [0])

    def test_build_candidate_boundary_points_clamps_to_local_window(self):
        points = build_candidate_boundary_points(
            segment_token_lengths=[12, 18, 9, 7],
            boundary_index=2,
            radius=2,
        )
        self.assertEqual(points["prior_index"], 2)
        self.assertEqual(points["candidate_indices"], [0, 1, 2, 3])
        self.assertEqual(points["candidate_deltas"], [-2, -1, 0, 1])

    def test_build_candidate_boundary_points_clamps_out_of_range_index(self):
        points = build_candidate_boundary_points(
            segment_token_lengths=[4, 5, 6],
            boundary_index=10,
            radius=1,
        )
        self.assertEqual(points["prior_index"], 2)
        self.assertEqual(points["candidate_indices"], [1, 2])
        self.assertEqual(points["candidate_deltas"], [-1, 0])

    def test_build_candidate_boundary_points_handles_empty_segments(self):
        points = build_candidate_boundary_points(
            segment_token_lengths=[],
            boundary_index=0,
            radius=2,
        )
        self.assertIsNone(points["prior_index"])
        self.assertEqual(points["candidate_indices"], [])
        self.assertEqual(points["candidate_deltas"], [])


if __name__ == "__main__":
    unittest.main()
