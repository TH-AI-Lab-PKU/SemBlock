import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from semantic_scheduler import (
    SchedulerTraceEvent,
    choose_semantic_block_length_from_scores,
    trace_events_from_block_history,
    write_scheduler_trace,
)


class SemanticSchedulerTests(unittest.TestCase):
    def test_choose_semantic_block_length_uses_best_score_when_above_threshold(self):
        block_length, best_index, best_score = choose_semantic_block_length_from_scores(
            [0.2, 0.91, 0.5],
            default_block_length=3,
            threshold=0.8,
        )

        self.assertEqual(block_length, 2)
        self.assertEqual(best_index, 1)
        self.assertAlmostEqual(best_score, 0.91)

    def test_choose_semantic_block_length_falls_back_to_default_when_below_threshold(self):
        block_length, best_index, best_score = choose_semantic_block_length_from_scores(
            [0.2, 0.4, 0.3],
            default_block_length=3,
            threshold=0.8,
        )

        self.assertEqual(block_length, 3)
        self.assertIsNone(best_index)
        self.assertAlmostEqual(best_score, 0.4)

    def test_write_scheduler_trace_serializes_jsonl(self):
        event = SchedulerTraceEvent(
            scheduler_name="semantic",
            sample_id="demo-1",
            step_index=0,
            generated_length=4,
            block_start=12,
            window_size=3,
            default_block_length=5,
            selected_block_length=2,
            selected_boundary_index=1,
            selected_boundary_position=13,
            selected_score=0.91,
            threshold=0.8,
            candidate_scores=[0.2, 0.91, 0.5],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            write_scheduler_trace(path, [event])
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_id"], "demo-1")
        self.assertEqual(rows[0]["selected_boundary_position"], 13)
        self.assertEqual(rows[0]["candidate_scores"], [0.2, 0.91, 0.5])

    def test_trace_events_from_block_history_uses_cumulative_positions(self):
        events = trace_events_from_block_history(
            scheduler_name="adablock",
            sample_id="demo-1",
            prompt_length=10,
            block_history=[3, 2],
        )

        self.assertEqual([event.selected_boundary_position for event in events], [12, 14])
        self.assertEqual([event.generated_length for event in events], [0, 3])


if __name__ == "__main__":
    unittest.main()
