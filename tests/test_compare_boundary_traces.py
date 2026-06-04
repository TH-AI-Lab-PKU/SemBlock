from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llada.compare_boundary_traces import (
    _load_jsonl_trace_records,
    _records_to_boundary_sets,
    _summarize_one_pair,
)


class CompareBoundaryTracesTest(unittest.TestCase):
    def test_excludes_terminal_and_compares_generated_offsets(self):
        semantic_records = [
            {"sample_id": "a", "step_index": 0, "generated_length": 0, "selected_block_length": 5, "selected_boundary_index": 4},
            {"sample_id": "a", "step_index": 1, "generated_length": 5, "selected_block_length": 7, "selected_boundary_index": None},
        ]
        adablock_records = [
            {"sample_id": "a", "step_index": 0, "generated_length": 0, "selected_block_length": 6, "selected_boundary_index": 5},
            {"sample_id": "a", "step_index": 1, "generated_length": 6, "selected_block_length": 6, "selected_boundary_index": 5},
        ]

        semantic_sets = _records_to_boundary_sets(
            semantic_records,
            match_key="sample_id",
            include_terminal=False,
            source="all",
        )
        adablock_sets = _records_to_boundary_sets(
            adablock_records,
            match_key="sample_id",
            include_terminal=False,
            source="all",
        )
        summary = _summarize_one_pair(
            task_name="toy",
            semantic_sets=semantic_sets,
            adablock_sets=adablock_sets,
            tolerances=[0, 1],
        )

        self.assertEqual(summary["semantic_boundary_count"], 1)
        self.assertEqual(summary["adablock_boundary_count"], 1)
        self.assertEqual(summary["exact"]["intersection_count"], 0)
        self.assertEqual(summary["tolerance"]["1"]["semantic_to_adablock_precision"], 1.0)

    def test_expands_prediction_records_with_block_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rank_0.jsonl"
            path.write_text(
                json.dumps({"sample_id": "gsm8k/test/0", "task_name": "gsm8k", "block_history": [3, 4]}) + "\n",
                encoding="utf-8",
            )

            records = _load_jsonl_trace_records(path)

        self.assertEqual([record["generated_length"] for record in records], [0, 3])
        self.assertEqual([record["selected_block_length"] for record in records], [3, 4])
        self.assertEqual(records[0]["sample_id"], "gsm8k/test/0")


if __name__ == "__main__":
    unittest.main()
