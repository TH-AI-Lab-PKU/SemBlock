import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from checkpoint_proxy_selection import extract_pass_at_1, materialize_best_checkpoint, rank_checkpoint_records


class CheckpointProxySelectionTests(unittest.TestCase):
    def test_extract_pass_at_1_reads_typical_lm_eval_results_payload(self):
        payload = {
            "results": {
                "humaneval": {
                    "pass@1,none": 0.61,
                    "pass@10,none": 0.8,
                },
                "mbpp": {
                    "pass_at_1,none": 0.42,
                },
            }
        }

        self.assertEqual(extract_pass_at_1(payload, "humaneval"), 0.61)
        self.assertEqual(extract_pass_at_1(payload, "mbpp"), 0.42)

    def test_extract_pass_at_1_accepts_direct_task_mapping_and_task_aliases(self):
        payload = {
            "HumanEval-50": {"pass@1": 0.7},
            "mbpp_20": {"pass_at_1": 0.35},
        }

        self.assertEqual(extract_pass_at_1(payload, "HumanEval"), 0.7)
        self.assertEqual(extract_pass_at_1(payload, "mbpp"), 0.35)

    def test_extract_pass_at_1_accepts_custom_humaneval_subset_task_name(self):
        payload = {
            "results": {
                "llada_humaneval_subset": {"pass@1,none": 0.64},
            }
        }

        self.assertEqual(extract_pass_at_1(payload, "humaneval"), 0.64)

    def test_rank_checkpoint_records_prefers_humaneval_then_mbpp(self):
        records = [
            {
                "checkpoint_path": "step-100",
                "optimizer_step": 100,
                "proxy_eval_results": {
                    "results": {
                        "humaneval": {"pass@1": 0.55},
                        "mbpp": {"pass_at_1": 0.95},
                    }
                },
            },
            {
                "checkpoint_path": "step-200",
                "optimizer_step": 200,
                "proxy_eval_results": {
                    "results": {
                        "humaneval": {"pass@1": 0.6},
                        "mbpp": {"pass_at_1": 0.2},
                    }
                },
            },
            {
                "checkpoint_path": "step-300",
                "optimizer_step": 300,
                "proxy_eval_results": {
                    "results": {
                        "humaneval": {"pass@1": 0.6},
                        "mbpp": {"pass_at_1": 0.4},
                    }
                },
            },
        ]

        ranked = rank_checkpoint_records(records)

        self.assertEqual(
            [record["checkpoint_path"] for record in ranked],
            ["step-300", "step-200", "step-100"],
        )

    def test_rank_checkpoint_records_accepts_direct_results_payloads(self):
        records = [
            {
                "checkpoint_path": "step-100",
                "optimizer_step": 100,
                "results": {
                    "humaneval": {"pass@1": 0.4},
                    "mbpp": {"pass_at_1": 0.95},
                },
            },
            {
                "checkpoint_path": "step-200",
                "optimizer_step": 200,
                "results": {
                    "humaneval": {"pass@1": 0.6},
                    "mbpp": {"pass_at_1": 0.2},
                },
            },
        ]

        ranked = rank_checkpoint_records(records, preserve_earlier_optimizer_step=False)

        self.assertEqual(
            [record["checkpoint_path"] for record in ranked],
            ["step-200", "step-100"],
        )

    def test_rank_checkpoint_records_can_prefer_earlier_optimizer_step_on_exact_ties(self):
        records = [
            {
                "checkpoint_path": "step-300",
                "optimizer_step": 300,
                "proxy_eval_results": {
                    "results": {
                        "humaneval": {"pass@1": 0.6},
                        "mbpp": {"pass_at_1": 0.4},
                    }
                },
            },
            {
                "checkpoint_path": "step-100",
                "optimizer_step": 100,
                "proxy_eval_results": {
                    "results": {
                        "humaneval": {"pass@1": 0.6},
                        "mbpp": {"pass_at_1": 0.4},
                    }
                },
            },
        ]

        ranked = rank_checkpoint_records(records, preserve_earlier_optimizer_step=True)

        self.assertEqual(
            [record["checkpoint_path"] for record in ranked],
            ["step-100", "step-300"],
        )

    def test_rank_checkpoint_records_prefers_screening_humaneval_score_over_legacy_proxy_score(self):
        records = [
            {
                "checkpoint_path": "step-100",
                "optimizer_step": 100,
                "humaneval_score": 0.80,
                "screening_humaneval_score": 0.48,
                "mbpp_score": 0.40,
            },
            {
                "checkpoint_path": "step-200",
                "optimizer_step": 200,
                "humaneval_score": 0.55,
                "screening_humaneval_score": 0.56,
                "mbpp_score": 0.20,
            },
        ]

        ranked = rank_checkpoint_records(records)

        self.assertEqual(
            [record["checkpoint_path"] for record in ranked],
            ["step-200", "step-100"],
        )

    def test_rank_checkpoint_records_uses_completion_proxy_tiebreakers_before_boundary_f1(self):
        records = [
            {
                "checkpoint_path": "step-boundary-f1",
                "optimizer_step": 100,
                "screening_humaneval_score": 0.60,
                "mbpp_score": 0.40,
                "valid_boundary_f1": 0.95,
                "parse_rate": 0.70,
                "block_length_distribution_error": 0.40,
                "avg_nfe": 120.0,
            },
            {
                "checkpoint_path": "step-parse-runtime",
                "optimizer_step": 200,
                "screening_humaneval_score": 0.60,
                "mbpp_score": 0.40,
                "valid_boundary_f1": 0.60,
                "parse_rate": 0.92,
                "block_length_distribution_error": 0.10,
                "avg_nfe": 90.0,
            },
        ]

        ranked = rank_checkpoint_records(records)

        self.assertEqual("step-parse-runtime", ranked[0]["checkpoint_path"])

    def test_materialize_best_checkpoint_copies_proxy_best_step_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            best_source = tmpdir_path / "boundary_head_step_000200.pt"
            other_source = tmpdir_path / "boundary_head_step_000100.pt"
            best_source.write_text("best-step", encoding="utf-8")
            other_source.write_text("older-step", encoding="utf-8")
            alias_path = tmpdir_path / "boundary_head_best.pt"

            summary = {
                "best_checkpoint_path": str(best_source),
                "best_record": {
                    "checkpoint_path": str(best_source),
                    "optimizer_step": 200,
                    "humaneval_score": 0.62,
                    "mbpp_score": 0.37,
                },
            }

            materialized = materialize_best_checkpoint(summary, alias_path)

            self.assertEqual(materialized, alias_path)
            self.assertTrue(alias_path.exists())
            self.assertEqual(alias_path.read_text(encoding="utf-8"), "best-step")


if __name__ == "__main__":
    unittest.main()
