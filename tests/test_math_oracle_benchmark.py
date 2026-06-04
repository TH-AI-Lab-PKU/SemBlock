import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from math_oracle_benchmark import (
    build_gsm8k_cot_prompt,
    build_math_cot_prompt,
    compute_oracle_generation_budget,
    select_math_fewshots,
    summarize_generation_records,
    summarize_oracle_documents,
)


class MathOracleBenchmarkTests(unittest.TestCase):
    def test_build_gsm8k_cot_prompt_appends_reasoning_stub(self):
        prompt = build_gsm8k_cot_prompt(
            question="If Tom has 2 apples and buys 2 more, how many apples does he have?",
            fewshot_examples=[
                {
                    "question": "What is 1 + 1?",
                    "answer": "We add the numbers to get 2. The answer is 2.",
                }
            ],
        )

        self.assertIn("Q: What is 1 + 1?", prompt)
        self.assertIn("A: Let's think step by step.\nWe add the numbers to get 2. The answer is 2.", prompt)
        self.assertTrue(
            prompt.endswith(
                "Q: If Tom has 2 apples and buys 2 more, how many apples does he have?\nA: Let's think step by step."
            )
        )

    def test_build_math_cot_prompt_formats_problem_answer_pairs(self):
        prompt = build_math_cot_prompt(
            problem="Compute 2+2.",
            fewshot_examples=[
                {
                    "problem": "Compute 1+1.",
                    "solution": "We add to get \\boxed{2}.",
                }
            ],
        )

        self.assertIn("Problem: Compute 1+1.\nAnswer: We add to get \\boxed{2}.", prompt)
        self.assertTrue(prompt.endswith("Problem: Compute 2+2.\nAnswer:"))

    def test_summarize_oracle_documents_reports_block_and_anchor_rates(self):
        summary = summarize_oracle_documents(
            [
                {"oracle_block_sizes": [5, 3], "has_final_answer_anchor": True},
                {"oracle_block_sizes": [4], "has_final_answer_anchor": False},
            ]
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["avg_block_length"], 4.0)
        self.assertAlmostEqual(summary["avg_block_count"], 1.5)
        self.assertEqual(summary["block_count_distribution"], {1: 1, 2: 1})
        self.assertAlmostEqual(summary["boundary_coverage_rate"], 0.5)
        self.assertAlmostEqual(summary["final_answer_anchor_hit_rate"], 0.5)

    def test_summarize_generation_records_reports_exact_match_and_nfe(self):
        summary = summarize_generation_records(
            [
                {"is_correct": True, "nfe_history": [2, 1], "block_history": [5, 3]},
                {"is_correct": False, "nfe_history": [4], "block_history": [4]},
            ]
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertAlmostEqual(summary["exact_match"], 0.5)
        self.assertAlmostEqual(summary["strict_match"], 0.5)
        self.assertAlmostEqual(summary["avg_nfe"], 3.5)
        self.assertAlmostEqual(summary["avg_generated_block_count"], 1.5)
        self.assertAlmostEqual(summary["avg_generated_block_length"], 4.0)

    def test_compute_oracle_generation_budget_prefers_gold_budget_over_global_default(self):
        self.assertEqual(compute_oracle_generation_budget([29, 27, 5], default_gen_length=512), 61)
        self.assertEqual(compute_oracle_generation_budget([], default_gen_length=512), 512)
        self.assertEqual(compute_oracle_generation_budget([300, 300], default_gen_length=512), 512)

    def test_select_math_fewshots_excludes_test_problem_overlaps(self):
        rows = [
            {"problem": "Compute 2+2.", "solution": "\\boxed{4}", "subject": "algebra"},
            {"problem": "Compute 3+3.", "solution": "\\boxed{6}", "subject": "geometry"},
            {"problem": "Compute 4+4.", "solution": "\\boxed{8}", "subject": "number_theory"},
        ]

        fewshots = select_math_fewshots(
            rows,
            num_fewshot=2,
            excluded_problem_texts=["Compute 2+2."],
        )

        self.assertEqual(len(fewshots), 2)
        self.assertEqual([row["problem"] for row in fewshots], ["Compute 3+3.", "Compute 4+4."])


if __name__ == "__main__":
    unittest.main()
