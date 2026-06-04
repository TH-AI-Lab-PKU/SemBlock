import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

import oracle_boundary_runtime_features as obr


class OracleBoundaryRuntimeFeaturesTests(unittest.TestCase):
    def test_mean_window_clamps_and_means(self):
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
        pooled = obr._mean_window(hidden_states, start=-2, end=2)
        expected = torch.tensor([0.5, 0.5])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_mean_window_falls_back_before_range(self):
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
        pooled = obr._mean_window(hidden_states, start=-5, end=-1)
        expected = torch.tensor([0.0, 0.0])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_mean_window_falls_back_after_range(self):
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
        pooled = obr._mean_window(hidden_states, start=5, end=7)
        expected = torch.tensor([2.0, 2.0])
        self.assertTrue(torch.allclose(pooled, expected))

    def test_pool_boundary_hidden_features_edges(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        pooled = obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=0, window=1)
        self.assertTrue(torch.allclose(pooled["left"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(pooled["center"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(pooled["right"], torch.tensor([1.0])))

    def test_pool_boundary_hidden_features_middle(self):
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]])
        pooled = obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=1, window=1)
        self.assertTrue(torch.allclose(pooled["left"], torch.tensor([0.0, 0.0])))
        self.assertTrue(torch.allclose(pooled["center"], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(pooled["right"], torch.tensor([2.0, 2.0])))

    def test_pool_boundary_hidden_features_normalizes_window(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        pooled = obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=2.0, window=0)
        self.assertTrue(torch.allclose(pooled["left"], torch.tensor([1.0])))
        self.assertTrue(torch.allclose(pooled["center"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(pooled["right"], torch.tensor([2.0])))

    def test_pool_boundary_hidden_features_rejects_fractional_index(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        with self.assertRaises(ValueError):
            obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=1.25, window=1)

    def test_pool_boundary_hidden_features_clamps_negative_index(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        pooled = obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=-5, window=1)
        self.assertTrue(torch.allclose(pooled["left"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(pooled["center"], torch.tensor([0.0])))
        self.assertTrue(torch.allclose(pooled["right"], torch.tensor([0.0])))

    def test_pool_boundary_hidden_features_clamps_large_index(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        pooled = obr.pool_boundary_hidden_features(hidden_states, boundary_token_index=99, window=1)
        self.assertTrue(torch.allclose(pooled["left"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(pooled["center"], torch.tensor([2.0])))
        self.assertTrue(torch.allclose(pooled["right"], torch.tensor([2.0])))

    def test_boundary_index_to_token_index_uses_cumulative_block_sizes(self):
        token_index = obr.boundary_index_to_token_index([2, 3, 1], boundary_index=1)
        self.assertEqual(token_index, 4)

    def test_build_boundary_feature_vector_concatenates_hidden_and_structure(self):
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]])
        feature = obr.build_boundary_feature_vector(
            hidden_states=hidden_states,
            oracle_block_sizes=[2, 3, 1],
            boundary_index=1,
            prior_boundary_point={"candidate_indices": [0, 1, 2]},
            has_final_answer_anchor=True,
        )
        self.assertEqual(tuple(feature.shape), (16,))
        self.assertTrue(torch.allclose(feature[:2], torch.tensor([3.0, 3.0])))
        self.assertTrue(torch.allclose(feature[2:4], torch.tensor([4.0, 4.0])))
        self.assertTrue(torch.allclose(feature[4:6], torch.tensor([5.0, 5.0])))
        self.assertTrue(torch.allclose(feature[6:8], torch.tensor([2.0, 2.0])))
        expected_tail = torch.tensor([0.5, 2.0 / 6.0, 1.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0, 0.5, 3.0 / 5.0, 1.0])
        self.assertTrue(torch.allclose(feature[-8:], expected_tail))

    def test_build_transition_feature_matrix_uses_real_transitions_only(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]]])
        matrix = obr.build_transition_feature_matrix(
            hidden_states=hidden_states,
            oracle_block_sizes=[2, 3, 1],
            prior_boundary_points=[
                {"candidate_indices": [0, 1]},
                {"candidate_indices": [0, 1, 2]},
                {"candidate_indices": [1, 2]},
            ],
            has_final_answer_anchor=True,
        )
        self.assertEqual(tuple(matrix.shape), (2, 12))

    def test_build_transition_feature_matrix_handles_single_block(self):
        hidden_states = torch.tensor([[[0.0], [1.0], [2.0]]])
        matrix = obr.build_transition_feature_matrix(
            hidden_states=hidden_states,
            oracle_block_sizes=[3],
            prior_boundary_points=[],
            has_final_answer_anchor=False,
        )
        self.assertEqual(tuple(matrix.shape), (0, 0))

    def test_mean_window_rejects_non_3d(self):
        hidden_states = torch.zeros(3, 4)
        with self.assertRaises(ValueError):
            obr._mean_window(hidden_states, start=0, end=1)

    def test_mean_window_rejects_empty_sequence(self):
        hidden_states = torch.zeros(1, 0, 3)
        with self.assertRaises(ValueError):
            obr._mean_window(hidden_states, start=0, end=1)

    def test_mean_window_rejects_empty_batch(self):
        hidden_states = torch.zeros(0, 2, 3)
        with self.assertRaises(ValueError):
            obr._mean_window(hidden_states, start=0, end=1)


if __name__ == "__main__":
    unittest.main()
