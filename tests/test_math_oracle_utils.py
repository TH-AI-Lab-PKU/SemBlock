import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from math_oracle_utils import (
    build_math_oracle_document,
    is_gsm8k_correct,
    is_hendrycks_math_correct,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text) + 1))}


class MathOracleUtilsTests(unittest.TestCase):
    def test_build_math_oracle_document_marks_final_anchor_and_block_sizes(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-1",
            source_dataset="gsm8k",
            prompt_text="Question: 3+4?",
            solution_text="We add 3 and 4 to get 7.\n#### 7",
            tokenizer=FakeTokenizer(),
            max_length=64,
        )

        self.assertEqual(doc["sample_id"], "gsm8k-1")
        self.assertEqual(doc["source_dataset"], "gsm8k")
        self.assertTrue(doc["has_final_answer_anchor"])
        self.assertEqual(len(doc["segments"]), len(doc["segment_boundary_types"]))
        self.assertEqual(len(doc["segments"]), len(doc["oracle_block_sizes"]))
        self.assertEqual(len(doc["segments"]), len(doc["oracle_prior_boundary_points"]))
        self.assertEqual(doc["oracle_prior_boundary_points"][0]["prior_index"], 0)
        self.assertIn("final_answer_anchor", doc["segment_boundary_types"][-1])

    def test_build_math_oracle_document_truncates_parallel_arrays_on_max_length(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-truncate",
            source_dataset="gsm8k",
            prompt_text="Question",
            solution_text=(
                "First find the initial amount: 10 + 5 = 15\n"
                "Then add the next amount: 15 + 5 = 20\n"
                "#### 20"
            ),
            tokenizer=FakeTokenizer(),
            max_length=1,
        )

        self.assertEqual(len(doc["oracle_block_sizes"]), 1)
        self.assertEqual(len(doc["segments"]), 1)
        self.assertEqual(len(doc["segment_boundary_types"]), 1)
        self.assertEqual(len(doc["oracle_prior_boundary_points"]), 1)


    def test_build_math_oracle_document_coalesces_instruction_equation_pairs_into_action_blocks(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-steps",
            source_dataset="gsm8k",
            prompt_text="Question",
            solution_text=(
                "First find how many gigabytes are in 40% of the file:\n"
                "200 GB * 40% = <<200*40*.01=80>>80 GB\n"
                "Then divide that number by the download rate to find the time until Windows restarts:\n"
                "80 GB / 2 GB/minute = <<80/2=40>>40 minutes\n"
                "#### 160"
            ),
            tokenizer=FakeTokenizer(),
            max_length=256,
        )

        self.assertEqual(
            doc["segments"],
            [
                "gigabytes in 40% of the file: 200 GB * 40% = <<200*40*.01=80>>80 GB",
                "time until Windows restarts: 80 GB / 2 GB/minute = <<80/2=40>>40 minutes",
                "#### 160",
            ],
        )

    def test_build_math_oracle_document_merges_label_only_and_parenthetical_segments(self):
        doc = build_math_oracle_document(
            sample_id="math-labels",
            source_dataset="math",
            prompt_text="Problem",
            solution_text=(
                "We can check each number one by one.\n\n"
                "3: 3 is not a factor of 34 since there is no number that can be multiplied by 3 to get 34."
                " ($34\div3$ gives a quotient of 11 and a remainder of 1.)\n\n"
                "14: 14 is a multiple of 7 since $7\cdot2=14$ .\n\n"
                "So, $\boxed{2}$ of the 2 numbers satisfy the condition."
            ),
            tokenizer=FakeTokenizer(),
            max_length=256,
        )

        self.assertFalse(any(segment.strip() == "3:" for segment in doc["segments"]))
        self.assertFalse(any(segment.strip() == "14:" for segment in doc["segments"]))
        self.assertTrue(any(segment.startswith("3: 3 is not a factor of 34") for segment in doc["segments"]))
        self.assertTrue(any("($34\div3$ gives a quotient of 11 and a remainder of 1.)" in segment for segment in doc["segments"]))

    def test_build_math_oracle_document_rewrites_pronoun_heavy_leads_into_explicit_state_labels(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-state",
            source_dataset="gsm8k",
            prompt_text="Question",
            solution_text=(
                "Then add those two amounts to find the total amount Jill makes per week: $700/week + $450/week = $<<700+450=1150>>1150/week\n"
                "Then multiply that number by the number of weeks Jill works in a year to find her annual salary: $1150/week * 50 weeks/year = $<<1150*50=57500>>57,500\n"
                "#### 57500"
            ),
            tokenizer=FakeTokenizer(),
            max_length=256,
        )

        self.assertEqual(
            doc["segments"],
            [
                "total amount Jill makes per week: $700/week + $450/week = $<<700+450=1150>>1150/week",
                "her annual salary: $1150/week * 50 weeks/year = $<<1150*50=57500>>57,500",
                "#### 57500",
            ],
        )
        self.assertFalse(any("that number" in segment.lower() for segment in doc["segments"]))
        self.assertFalse(any("those two amounts" in segment.lower() for segment in doc["segments"]))

    def test_build_math_oracle_document_rewrites_cost_base_and_profit_state_labels(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-profit",
            source_dataset="gsm8k",
            prompt_text="Question",
            solution_text=(
                "The cost of the house and repairs came out to 80,000+50,000=$<<80000+50000=130000>>130,000\n"
                "He increased the value of the house by 80,000*1.5=<<80000*1.5=120000>>120,000\n"
                "So the new value of the house is 120,000+80,000=$<<120000+80000=200000>>200,000\n"
                "So he made a profit of 200,000-130,000=$<<200000-130000=70000>>70,000\n"
                "#### 70000"
            ),
            tokenizer=FakeTokenizer(),
            max_length=2048,
        )

        self.assertEqual(
            doc["segments"],
            [
                "cost of the house and repairs: 80,000+50,000=$<<80000+50000=130000>>130,000",
                "increase in the value of the house: 80,000*1.5=<<80000*1.5=120000>>120,000",
                "new value of the house: 120,000+80,000=$<<120000+80000=200000>>200,000",
                "profit: 200,000-130,000=$<<200000-130000=70000>>70,000",
                "#### 70000",
            ],
        )

    def test_build_math_oracle_document_rewrites_fraction_and_amount_state_labels(self):
        doc = build_math_oracle_document(
            sample_id="gsm8k-pension",
            source_dataset="gsm8k",
            prompt_text="Question",
            solution_text=(
                "First find how many years Marcy works after 20 years: 30 years - 20 years = <<30-20=10>>10 years\n"
                "Then multiply that number by the amount of her pension she gets per year: 10 years * 5% = 50%\n"
                "Then multiply that percentage by the total amount of the full pension to find how much she gets: $50,000 * 50% = $<<50000*50*.01=25000>>25,000\n"
                "#### 25000"
            ),
            tokenizer=FakeTokenizer(),
            max_length=256,
        )

        self.assertEqual(
            doc["segments"],
            [
                "years Marcy works after 20 years: 30 years - 20 years = <<30-20=10>>10 years",
                "fraction of the full pension she gets: 10 years * 5% = 50%",
                "amount she gets: $50,000 * 50% = $<<50000*50*.01=25000>>25,000",
                "#### 25000",
            ],
        )

    def test_is_gsm8k_correct_matches_strict_numeric_answers(self):
        self.assertTrue(is_gsm8k_correct("We add them.\n#### 7", "Reasoning...\n#### 7"))
        self.assertFalse(is_gsm8k_correct("We add them.\n#### 7", "Reasoning...\n#### 8"))

    def test_is_hendrycks_math_correct_accepts_equivalent_boxed_fraction(self):
        self.assertTrue(
            is_hendrycks_math_correct(
                "Therefore the answer is \boxed{\frac{1}{2}}.",
                "Thus we get \boxed{1/2}.",
            )
        )
        self.assertFalse(
            is_hendrycks_math_correct(
                "Therefore the answer is \boxed{\frac{1}{2}}.",
                "Thus we get \boxed{2}.",
            )
        )


if __name__ == "__main__":
    unittest.main()
