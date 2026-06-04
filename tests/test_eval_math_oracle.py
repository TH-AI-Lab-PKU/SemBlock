import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

import eval_math_oracle as emo
from models.local_boundary_corrector import LocalBoundaryCorrector


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2, 3, 4]}

    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded"


class EvalMathOracleTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            {
                "sample_id": "gsm8k/test/0",
                "source_dataset": "gsm8k",
                "prompt_text": "Q: demo\nA: Let's think step by step.",
                "solution_text": "step 1\nstep 2\n#### 6",
                "gold_solution": "#### 6",
                "oracle_block_sizes": [11, 7, 3],
                "oracle_prior_boundary_points": [
                    {"prior_index": 0, "candidate_indices": [0, 1], "candidate_deltas": [0, 1]},
                    {"prior_index": 1, "candidate_indices": [0, 1, 2], "candidate_deltas": [-1, 0, 1]},
                    {"prior_index": 2, "candidate_indices": [1, 2], "candidate_deltas": [-1, 0]},
                ],
                "has_final_answer_anchor": True,
                "segments": ["a", "b", "c"],
                "question": "demo question",
            }
        ]
        self.model = mock.Mock(device="cpu")
        self.tokenizer = FakeTokenizer()

    def _run_eval(self, docs=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdir:
            return emo.evaluate_documents(
                task_name="gsm8k",
                documents=self.docs if docs is None else docs,
                model=self.model,
                tokenizer=self.tokenizer,
                is_instruct=False,
                device="cpu",
                steps=16,
                gen_length=64,
                block_length=32,
                threshold=0.9,
                mask_id=126336,
                remasking="low_confidence",
                gsm8k_landing_control=False,
                gsm8k_landing_tail_lines=4,
                output_dir=Path(tmpdir),
                **kwargs,
            )

    def test_resolve_oracle_generator_uses_independent_module(self):
        self.assertEqual(emo.resolve_oracle_generator(use_cache=False, dual_cache=False).__module__, "generate_oracle_blocks")
        self.assertEqual(emo.resolve_oracle_generator(use_cache=True, dual_cache=False).__module__, "generate_oracle_blocks")
        self.assertEqual(emo.resolve_oracle_generator(use_cache=True, dual_cache=True).__module__, "generate_oracle_blocks")
        self.assertIs(emo.resolve_oracle_generator(use_cache=False, dual_cache=False), emo.generate_oracle_blocks)
        self.assertIs(emo.resolve_oracle_generator(use_cache=True, dual_cache=False), emo.generate_oracle_blocks_prefix_cache)
        self.assertIs(emo.resolve_oracle_generator(use_cache=True, dual_cache=True), emo.generate_oracle_blocks_dual_cache)

    def test_evaluate_documents_skips_cache_for_short_docs_when_min_block_gate_is_enabled(self):
        with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
             mock.patch.object(emo, "decode_generation", return_value="prediction"), \
             mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
             mock.patch.object(emo, "generate_oracle_blocks", return_value=(torch.tensor([[1, 2]]), [1], [11, 7, 3])) as no_cache_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])) as prefix_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_dual_cache", return_value=(torch.tensor([[1, 2]]), [3], [11, 7, 3])) as dual_mock:
            self._run_eval(use_cache=True, dual_cache=False, cache_min_block_count=4)

        no_cache_mock.assert_called_once()
        prefix_mock.assert_not_called()
        dual_mock.assert_not_called()

    def test_evaluate_documents_uses_prefix_cache_when_enabled(self):
        with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
             mock.patch.object(emo, "decode_generation", return_value="prediction"), \
             mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
             mock.patch.object(emo, "generate_oracle_blocks", return_value=(torch.tensor([[1, 2]]), [1], [11, 7, 3])) as no_cache_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])) as prefix_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_dual_cache", return_value=(torch.tensor([[1, 2]]), [3], [11, 7, 3])) as dual_mock:
            self._run_eval(use_cache=True, dual_cache=False)

        prefix_mock.assert_called_once()
        self.assertEqual(prefix_mock.call_args.kwargs["oracle_block_sizes"], [11, 7, 3])
        no_cache_mock.assert_not_called()
        dual_mock.assert_not_called()

    def test_evaluate_documents_requires_predicted_deltas_for_prior_correction(self):
        with self.assertRaises(ValueError):
            with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
                 mock.patch.object(emo, "decode_generation", return_value="prediction"), \
                 mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
                 mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])):
                self._run_eval(docs=self.docs, cache_policy="prior_correction", use_cache=False, dual_cache=False)

    def test_evaluate_documents_applies_prior_correction_with_prefix_cache(self):
        docs = [dict(self.docs[0], predicted_deltas={1: -2})]
        with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
             mock.patch.object(emo, "decode_generation", return_value="prediction"), \
             mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
             mock.patch.object(emo, "generate_oracle_blocks", return_value=(torch.tensor([[1, 2]]), [1], [11, 7, 3])) as no_cache_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 5, 3])) as prefix_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_dual_cache", return_value=(torch.tensor([[1, 2]]), [3], [11, 7, 3])) as dual_mock:
            result = self._run_eval(docs=docs, cache_policy="prior_correction", use_cache=False, dual_cache=False)

        prefix_mock.assert_called_once()
        self.assertEqual(prefix_mock.call_args.kwargs["oracle_block_sizes"], [11, 5, 3])
        self.assertEqual(result["records"][0]["oracle_block_sizes"], [11, 5, 3])
        self.assertEqual(result["records"][0]["cache_mode"], "prefix_cache")
        self.assertEqual(result["records"][0]["cache_policy"], "prior_correction")
        self.assertTrue(result["records"][0]["prior_correction_applied"])
        self.assertEqual(result["records"][0]["predicted_delta_count"], 1)
        self.assertEqual(result["summary"]["cache_policy"], "prior_correction")
        no_cache_mock.assert_not_called()
        dual_mock.assert_not_called()

    def test_evaluate_documents_rejects_local_corrector_without_prior_correction(self):
        with self.assertRaises(ValueError):
            self._run_eval(use_cache=False, dual_cache=False, local_corrector_path=Path('/tmp/local_corrector.pt'))

    def test_evaluate_documents_uses_boundary_gate_policy_with_runtime_mask(self):
        docs = [dict(self.docs[0])]
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]])
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "local_corrector.pt"
            model = LocalBoundaryCorrector(input_dim=16, hidden_dim=4, delta_classes=5)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.gate_head.bias.copy_(torch.tensor([0.0, 1.0]))
            torch.save(model.state_dict(), checkpoint_path)

            with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
                 mock.patch.object(emo, "decode_generation", return_value="prediction"), \
                 mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
                 mock.patch.object(emo, "extract_early_hidden_states", return_value=hidden_states), \
                 mock.patch.object(emo, "generate_oracle_blocks_boundary_gate", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])) as gate_mock:
                result = self._run_eval(
                    docs=docs,
                    cache_policy="boundary_gate",
                    use_cache=False,
                    dual_cache=False,
                    local_corrector_path=checkpoint_path,
                )

        gate_mock.assert_called_once()
        self.assertEqual(gate_mock.call_args.kwargs["boundary_carry_mask"], [1, 1])
        self.assertEqual(result["records"][0]["cache_policy"], "boundary_gate")
        self.assertEqual(result["records"][0]["cache_mode"], "boundary_gate")
        self.assertTrue(result["records"][0]["boundary_gate_applied"])
        self.assertEqual(result["records"][0]["carry_on_boundary_count"], 2)
        self.assertEqual(result["summary"]["cache_policy"], "boundary_gate")
        self.assertEqual(result["summary"]["boundary_gate_applied_doc_count"], 1)

    def test_evaluate_documents_uses_local_corrector_checkpoint_when_provided(self):
        docs = [dict(self.docs[0])]
        hidden_states = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]])
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "local_corrector.pt"
            model = LocalBoundaryCorrector(input_dim=16, hidden_dim=4, delta_classes=5)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.gate_head.bias.copy_(torch.tensor([0.0, 1.0]))
                model.delta_head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 2.0, 0.0]))
            torch.save(model.state_dict(), checkpoint_path)

            with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
                 mock.patch.object(emo, "decode_generation", return_value="prediction"), \
                 mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
                 mock.patch.object(emo, "extract_early_hidden_states", return_value=hidden_states), \
                 mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [12, 8, 4])) as prefix_mock:
                result = self._run_eval(docs=docs, cache_policy="prior_correction", use_cache=False, dual_cache=False, local_corrector_path=checkpoint_path)

        prefix_mock.assert_called_once()
        self.assertEqual(prefix_mock.call_args.kwargs["oracle_block_sizes"], [12, 8, 4])
        self.assertEqual(result["records"][0]["oracle_block_sizes"], [12, 8, 4])
        self.assertEqual(result["records"][0]["predicted_delta_count"], 3)
        self.assertTrue(result["records"][0]["prior_correction_applied"])
        self.assertEqual(result["summary"]["prior_correction_applied_doc_count"], 1)
        self.assertEqual(result["summary"]["prior_correction_requested_doc_count"], 1)

    def test_evaluate_documents_marks_prior_correction_noop_when_deltas_are_empty(self):
        docs = [dict(self.docs[0], predicted_deltas={})]
        with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
             mock.patch.object(emo, "decode_generation", return_value="prediction"), \
             mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
             mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])) as prefix_mock:
            result = self._run_eval(docs=docs, cache_policy="prior_correction", use_cache=False, dual_cache=False)

        prefix_mock.assert_called_once()
        self.assertFalse(result["records"][0]["prior_correction_applied"])
        self.assertEqual(result["records"][0]["predicted_delta_count"], 0)

    def test_evaluate_documents_rejects_conflicting_cache_configuration(self):
        with self.assertRaises(ValueError):
            with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
                 mock.patch.object(emo, "decode_generation", return_value="prediction"), \
                 mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
                 mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])):
                self._run_eval(use_cache=True, dual_cache=False, cache_policy="prefix_cache")

    def test_evaluate_documents_uses_dual_cache_when_requested(self):
        with mock.patch.object(emo, "build_generation_prompt", return_value="prompt"), \
             mock.patch.object(emo, "decode_generation", return_value="prediction"), \
             mock.patch.object(emo, "is_gsm8k_correct", return_value=True), \
             mock.patch.object(emo, "generate_oracle_blocks", return_value=(torch.tensor([[1, 2]]), [1], [11, 7, 3])) as no_cache_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_prefix_cache", return_value=(torch.tensor([[1, 2]]), [2], [11, 7, 3])) as prefix_mock, \
             mock.patch.object(emo, "generate_oracle_blocks_dual_cache", return_value=(torch.tensor([[1, 2]]), [3], [11, 7, 3])) as dual_mock:
            self._run_eval(use_cache=True, dual_cache=True)

        dual_mock.assert_called_once()
        self.assertEqual(dual_mock.call_args.kwargs["oracle_block_sizes"], [11, 7, 3])
        no_cache_mock.assert_not_called()
        prefix_mock.assert_not_called()

    def test_extract_early_hidden_states_returns_last_layer(self):
        class FakeOutput:
            def __init__(self, hidden_states):
                self.hidden_states = hidden_states

        class FakeModel:
            def __init__(self):
                self.calls = []

            def __call__(self, input_tensor, **kwargs):
                self.calls.append(kwargs)
                return FakeOutput([torch.zeros(1, 2, 3), torch.ones(1, 2, 3)])

        model = FakeModel()
        input_tensor = torch.tensor([[1, 2]])
        output = emo.extract_early_hidden_states(model, input_tensor)
        self.assertTrue(torch.equal(output, torch.ones(1, 2, 3)))
        self.assertTrue(model.calls)
        self.assertTrue(model.calls[0].get("output_hidden_states"))
        self.assertTrue(model.calls[0].get("return_dict"))

    def test_sample_math_rows_stratified_round_robin_balances_subject_and_level(self):
        rows = [
            {"unique_id": "a1", "subject": "algebra", "level": "1"},
            {"unique_id": "a2", "subject": "algebra", "level": "1"},
            {"unique_id": "a3", "subject": "algebra", "level": "1"},
            {"unique_id": "g1", "subject": "geometry", "level": "2"},
            {"unique_id": "g2", "subject": "geometry", "level": "2"},
            {"unique_id": "n1", "subject": "number_theory", "level": "3"},
            {"unique_id": "n2", "subject": "number_theory", "level": "3"},
            {"unique_id": "c1", "subject": "combinatorics", "level": "1"},
        ]

        sampled = emo.sample_math_rows_stratified(rows, limit=6, seed=0)

        self.assertEqual(len(sampled), 6)
        sampled_ids = [str(row["unique_id"]) for row in sampled]
        self.assertCountEqual(sampled_ids[:4], ["a1", "g1", "n1", "c1"])
        self.assertEqual(len(set(sampled_ids)), 6)
        self.assertNotEqual(sampled_ids, ["a1", "a2", "a3", "g1", "g2", "n1"])

    def test_sample_math_rows_stratified_returns_all_rows_when_limit_covers_dataset(self):
        rows = [
            {"unique_id": "a1", "subject": "algebra", "level": "1"},
            {"unique_id": "g1", "subject": "geometry", "level": "2"},
        ]

        sampled = emo.sample_math_rows_stratified(rows, limit=4, seed=123)

        self.assertEqual(sampled, rows)

    def test_build_math_documents_uses_stratified_sampling_for_limited_eval(self):
        train_rows = [
            {"problem": "train-a", "solution": "sol-a"},
            {"problem": "train-b", "solution": "sol-b"},
        ]
        test_rows = [
            {"unique_id": "a1", "problem": "p-a1", "solution": "s-a1", "subject": "algebra", "level": "1"},
            {"unique_id": "a2", "problem": "p-a2", "solution": "s-a2", "subject": "algebra", "level": "1"},
            {"unique_id": "g1", "problem": "p-g1", "solution": "s-g1", "subject": "geometry", "level": "2"},
            {"unique_id": "n1", "problem": "p-n1", "solution": "s-n1", "subject": "number_theory", "level": "3"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            train_path = tmpdir_path / "train.jsonl"
            test_path = tmpdir_path / "test.jsonl"
            train_path.write_text("\n".join(__import__("json").dumps(row) for row in train_rows) + "\n", encoding="utf-8")
            test_path.write_text("\n".join(__import__("json").dumps(row) for row in test_rows) + "\n", encoding="utf-8")

            with mock.patch.object(emo, "select_math_fewshots", return_value=train_rows[:1]), \
                 mock.patch.object(emo, "build_math_cot_prompt", side_effect=lambda problem, fewshots: f"prompt::{problem}"), \
                 mock.patch.object(
                     emo,
                     "build_math_oracle_document",
                     side_effect=lambda sample_id, source_dataset, prompt_text, solution_text, tokenizer, max_length: {
                         "sample_id": sample_id,
                         "source_dataset": source_dataset,
                         "prompt_text": prompt_text,
                         "solution_text": solution_text,
                         "oracle_block_sizes": [1],
                         "segments": ["seg"],
                         "has_final_answer_anchor": False,
                     },
                 ):
                documents = emo.build_math_documents(
                    tokenizer=self.tokenizer,
                    train_path=train_path,
                    test_path=test_path,
                    num_fewshot=1,
                    limit=3,
                    gen_length=64,
                )

        self.assertEqual([doc["sample_id"] for doc in documents], ["a1", "g1", "n1"])
        self.assertEqual([doc["subject"] for doc in documents], ["algebra", "geometry", "number_theory"])


if __name__ == "__main__":
    unittest.main()
