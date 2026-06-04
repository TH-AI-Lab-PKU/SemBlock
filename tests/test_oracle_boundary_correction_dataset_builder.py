import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

import build_local_boundary_correction_dataset as builder
from oracle_boundary_correction_data import build_boundary_correction_record


class OracleBoundaryCorrectionDatasetBuilderTests(unittest.TestCase):
    def test_apply_single_boundary_delta_clamps_only_target_boundary(self):
        corrected = builder.apply_single_boundary_delta([5, 1, 4], boundary_index=1, delta=-2)
        self.assertEqual(corrected, [5, 1, 4])

        corrected = builder.apply_single_boundary_delta([5, 1, 4], boundary_index=0, delta=2)
        self.assertEqual(corrected, [7, 1, 4])

    def test_delta_to_target_class_uses_fixed_five_class_mapping(self):
        self.assertEqual(builder.delta_to_target_class(-2), 0)
        self.assertEqual(builder.delta_to_target_class(-1), 1)
        self.assertEqual(builder.delta_to_target_class(0), 2)
        self.assertEqual(builder.delta_to_target_class(1), 3)
        self.assertEqual(builder.delta_to_target_class(2), 4)
        with self.assertRaises(ValueError):
            builder.delta_to_target_class(3)

    def test_build_training_row_includes_targets_and_metadata(self):
        correction_record = build_boundary_correction_record(
            sample_id="gsm8k/train/0",
            boundary_index=1,
            best_delta=1,
            delta_scores={-1: 0.0, 0: 0.0, 1: 1.0},
        )
        doc = {
            "sample_id": "gsm8k/train/0",
            "source_dataset": "gsm8k",
            "oracle_block_sizes": [11, 7, 3],
            "has_final_answer_anchor": True,
        }
        row = builder.build_training_row(
            doc=doc,
            correction_record=correction_record,
            feature_vector=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        )
        self.assertEqual(row["sample_id"], "gsm8k/train/0")
        self.assertEqual(row["boundary_index"], 1)
        self.assertEqual(row["best_delta"], 1)
        self.assertEqual(row["gate_target"], 1)
        self.assertEqual(row["delta_target"], 3)
        self.assertEqual(row["oracle_block_sizes"], [11, 7, 3])
        self.assertTrue(row["has_final_answer_anchor"])
        self.assertEqual(row["features"], [0.1, 0.2, 0.3])
        self.assertEqual(row["delta_scores"], {-1: 0.0, 0: 0.0, 1: 1.0})

    def test_split_rows_by_document_keeps_documents_together(self):
        rows = [
            {"sample_id": "gsm8k/train/0", "boundary_index": 0},
            {"sample_id": "gsm8k/train/0", "boundary_index": 1},
            {"sample_id": "gsm8k/train/1", "boundary_index": 0},
            {"sample_id": "gsm8k/train/2", "boundary_index": 0},
        ]
        train_rows, valid_rows = builder.split_rows_by_document(rows, valid_doc_count=1)
        self.assertEqual([row["sample_id"] for row in valid_rows], ["gsm8k/train/2"])
        self.assertEqual([row["sample_id"] for row in train_rows], ["gsm8k/train/0", "gsm8k/train/0", "gsm8k/train/1"])

    def test_score_gsm8k_prediction_is_binary_exact_match(self):
        self.assertEqual(builder.score_gsm8k_prediction("#### 42", "The answer is 42"), 1.0)
        self.assertEqual(builder.score_gsm8k_prediction("#### 42", "The answer is 41"), 0.0)

    def test_score_gsm8k_prediction_numeric_margin_gives_partial_credit(self):
        exact = builder.score_gsm8k_prediction("#### 42", "The answer is 42", objective="numeric_margin")
        close = builder.score_gsm8k_prediction("#### 42", "The answer is 41", objective="numeric_margin")
        far = builder.score_gsm8k_prediction("#### 42", "The answer is 420", objective="numeric_margin")
        missing = builder.score_gsm8k_prediction("#### 42", "I am not sure.", objective="numeric_margin")

        self.assertEqual(exact, 1.0)
        self.assertGreater(close, far)
        self.assertGreater(far, missing)
        self.assertGreater(close, 0.0)
        self.assertLess(close, 1.0)

    def test_score_gsm8k_prediction_rejects_unknown_objective(self):
        with self.assertRaises(ValueError):
            builder.score_gsm8k_prediction("#### 42", "The answer is 42", objective="does_not_exist")

    def test_build_rows_for_document_uses_best_scored_delta_per_boundary(self):
        doc = {
            "sample_id": "gsm8k/train/0",
            "source_dataset": "gsm8k",
            "oracle_block_sizes": [11, 7],
            "has_final_answer_anchor": True,
        }
        feature_matrix = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
        rows = builder.build_rows_for_document(
            doc=doc,
            feature_matrix=feature_matrix,
            delta_scores_by_boundary={
                0: {-1: 0.0, 0: 0.0, 1: 1.0},
                1: {-1: 1.0, 0: 0.0, 1: 0.0},
            },
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["best_delta"], 1)
        self.assertEqual(rows[0]["delta_target"], 3)
        self.assertEqual(rows[1]["best_delta"], -1)
        self.assertEqual(rows[1]["delta_target"], 1)

    def test_collect_boundary_delta_scores_uses_candidate_scorer(self):
        doc = {"oracle_block_sizes": [11, 7, 3]}
        scores = builder.collect_boundary_delta_scores(
            doc=doc,
            boundary_index=0,
            candidate_deltas=[-1, 0, 1],
            candidate_scorer=lambda block_sizes, delta: 1.0 if block_sizes[0] == 12 and delta == 1 else 0.0,
        )
        self.assertEqual(scores, {-1: 0.0, 0: 0.0, 1: 1.0})

    def test_write_dataset_artifacts_writes_expected_files(self):
        train_rows = [{"sample_id": "gsm8k/train/0", "gate_target": 0, "delta_target": 2}]
        valid_rows = [{"sample_id": "gsm8k/train/1", "gate_target": 1, "delta_target": 3}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = builder.write_dataset_artifacts(
                output_dir=output_dir,
                train_rows=train_rows,
                valid_rows=valid_rows,
            )
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "valid.jsonl").exists())
            self.assertTrue((output_dir / "summary.json").exists())
            self.assertEqual(Path(paths["train_jsonl"]), output_dir / "train.jsonl")
            self.assertEqual(Path(paths["valid_jsonl"]), output_dir / "valid.jsonl")
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["row_count"], 1)
            self.assertEqual(summary["valid"]["row_count"], 1)

    def test_summarize_rows_reports_gate_and_delta_distribution(self):
        rows = [
            {"sample_id": "gsm8k/train/0", "gate_target": 0, "delta_target": 2},
            {"sample_id": "gsm8k/train/0", "gate_target": 1, "delta_target": 3},
            {"sample_id": "gsm8k/train/1", "gate_target": 1, "delta_target": 1},
        ]
        summary = builder.summarize_rows(rows, split_name="train")
        self.assertEqual(summary["split_name"], "train")
        self.assertEqual(summary["row_count"], 3)
        self.assertEqual(summary["document_count"], 2)
        self.assertEqual(summary["gate_target_distribution"], {"0": 1, "1": 2})
        self.assertEqual(summary["delta_target_distribution"], {"1": 1, "2": 1, "3": 1})


if __name__ == "__main__":
    unittest.main()
