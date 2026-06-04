import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

import build_boundary_gate_dataset as builder


class BoundaryGateDatasetBuilderTests(unittest.TestCase):
    def test_build_reference_boundary_carry_mask_defaults_to_all_on(self):
        reference_mask = builder.build_reference_boundary_carry_mask(transition_count=3)
        self.assertEqual(reference_mask, [1, 1, 1])

    def test_build_boundary_carry_masks_toggle_target_against_reference(self):
        carry_on_mask, carry_off_mask = builder.build_boundary_carry_masks(
            reference_mask=[1, 1, 1],
            transition_index=1,
        )
        self.assertEqual(carry_on_mask, [1, 1, 1])
        self.assertEqual(carry_off_mask, [1, 0, 1])

    def test_build_gate_training_row_records_gate_scores(self):
        row = builder.build_gate_training_row(
            doc={
                "sample_id": "gsm8k/train/0",
                "source_dataset": "gsm8k",
                "oracle_block_sizes": [11, 7, 3],
                "has_final_answer_anchor": True,
            },
            transition_index=0,
            feature_vector=torch.tensor([0.1, 0.2], dtype=torch.float32),
            carry_on_score=1.0,
            carry_off_score=0.0,
        )
        self.assertEqual(row["sample_id"], "gsm8k/train/0")
        self.assertEqual(row["transition_index"], 0)
        self.assertEqual(row["features"], [0.1, 0.2])
        self.assertEqual(row["gate_target"], 1)
        self.assertEqual(row["gate_scores"], {"carry_on": 1.0, "carry_off": 0.0})
        self.assertEqual(row["oracle_block_sizes"], [11, 7, 3])
        self.assertTrue(row["has_final_answer_anchor"])

    def test_build_gate_training_row_prefers_no_carry_when_scores_tie(self):
        row = builder.build_gate_training_row(
            doc={
                "sample_id": "gsm8k/train/0",
                "source_dataset": "gsm8k",
                "oracle_block_sizes": [11, 7, 3],
                "has_final_answer_anchor": False,
            },
            transition_index=1,
            feature_vector=torch.tensor([0.3, 0.4], dtype=torch.float32),
            carry_on_score=0.0,
            carry_off_score=0.0,
        )
        self.assertEqual(row["gate_target"], 0)

    def test_build_rows_for_document_uses_carry_score_pairs(self):
        rows = builder.build_rows_for_document(
            doc={
                "sample_id": "gsm8k/train/0",
                "source_dataset": "gsm8k",
                "oracle_block_sizes": [11, 7, 3],
                "has_final_answer_anchor": True,
            },
            feature_matrix=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
            carry_score_pairs={
                0: {"carry_on": 1.0, "carry_off": 0.0},
                1: {"carry_on": 0.0, "carry_off": 0.0},
            },
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["transition_index"], 0)
        self.assertEqual(rows[0]["gate_target"], 1)
        self.assertEqual(rows[1]["transition_index"], 1)
        self.assertEqual(rows[1]["gate_target"], 0)

    def test_split_rows_by_document_keeps_documents_together(self):
        rows = [
            {"sample_id": "gsm8k/train/0", "transition_index": 0, "gate_target": 1},
            {"sample_id": "gsm8k/train/0", "transition_index": 1, "gate_target": 0},
            {"sample_id": "gsm8k/train/1", "transition_index": 0, "gate_target": 1},
        ]
        train_rows, valid_rows = builder.split_rows_by_document(rows, valid_doc_count=1)
        self.assertEqual([row["sample_id"] for row in train_rows], ["gsm8k/train/0", "gsm8k/train/0"])
        self.assertEqual([row["sample_id"] for row in valid_rows], ["gsm8k/train/1"])

    def test_write_dataset_artifacts_writes_summary(self):
        train_rows = [{"sample_id": "gsm8k/train/0", "gate_target": 1}]
        valid_rows = [{"sample_id": "gsm8k/train/1", "gate_target": 0}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = builder.write_dataset_artifacts(
                output_dir=output_dir,
                train_rows=train_rows,
                valid_rows=valid_rows,
                metadata={"task": "gsm8k", "label_objective": "carry_gate_binary"},
            )
            self.assertTrue((output_dir / "train.jsonl").exists())
            self.assertTrue((output_dir / "valid.jsonl").exists())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["train"]["row_count"], 1)
            self.assertEqual(summary["valid"]["row_count"], 1)
            self.assertEqual(Path(paths["summary_json"]), output_dir / "summary.json")


if __name__ == "__main__":
    unittest.main()
