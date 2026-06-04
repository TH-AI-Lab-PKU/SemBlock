import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from eval_prompting import (
    build_generation_prompt,
    is_code_completion_doc,
    should_use_chat_template,
    should_use_raw_completion_decode,
    truncate_generated_text,
)


class StubTokenizer:
    def apply_chat_template(self, chat_history, add_generation_prompt=True, tokenize=False):
        assert add_generation_prompt is True
        assert tokenize is False
        return f"<CHAT>{chat_history[0]['content']}"


class EvalPromptingTests(unittest.TestCase):
    def test_humaneval_docs_are_treated_as_code_completion(self):
        doc = {
            "task_id": "HumanEval/0",
            "entry_point": "has_close_elements",
            "canonical_solution": "    return False\n",
        }

        self.assertTrue(is_code_completion_doc(doc))
        self.assertFalse(should_use_chat_template(is_instruct=True, doc=doc))
        self.assertTrue(should_use_raw_completion_decode(is_instruct=True, doc=doc))

    def test_mbpp_docs_are_treated_as_code_completion(self):
        doc = {
            "task_id": 11,
            "text": "Write a function to count words.",
            "test_list": ["assert count_words('a a') == 2"],
        }

        self.assertTrue(is_code_completion_doc(doc))
        self.assertFalse(should_use_chat_template(is_instruct=True, doc=doc))
        self.assertFalse(should_use_raw_completion_decode(is_instruct=True, doc=doc))

    def test_gsm8k_docs_keep_chat_template_for_instruct_models(self):
        doc = {
            "question": "If Alice has 3 apples...",
            "answer": "#### 5",
        }

        self.assertFalse(is_code_completion_doc(doc))
        self.assertTrue(should_use_chat_template(is_instruct=True, doc=doc))

    def test_build_generation_prompt_uses_raw_prompt_for_humaneval(self):
        tokenizer = StubTokenizer()
        question = "def has_close_elements(numbers, threshold):\n"
        doc = {
            "task_id": "HumanEval/0",
            "entry_point": "has_close_elements",
            "canonical_solution": "    return False\n",
        }

        prompt = build_generation_prompt(
            tokenizer,
            question=question,
            is_instruct=True,
            doc=doc,
        )

        self.assertEqual(prompt, question)

    def test_build_generation_prompt_uses_chat_template_for_gsm8k(self):
        tokenizer = StubTokenizer()
        question = "Question: 1 + 1 = ?"
        doc = {
            "question": "1 + 1 = ?",
            "answer": "#### 2",
        }

        prompt = build_generation_prompt(
            tokenizer,
            question=question,
            is_instruct=True,
            doc=doc,
        )

        self.assertEqual(prompt, "<CHAT>Question: 1 + 1 = ?")

    def test_truncate_generated_text_keeps_humaneval_completion_only(self):
        doc = {
            "task_id": "HumanEval/0",
            "entry_point": "has_close_elements",
            "canonical_solution": "    return False\n",
        }
        text = (
            "    return False\n\n"
            "if __name__ == \"__main__\":\n"
            "    import doctest\n"
            "```\n"
            "### Explanation:\n"
            "This is extra commentary.\n"
        )

        truncated = truncate_generated_text(
            text,
            stop_tokens=["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
            is_instruct=True,
            doc=doc,
        )

        self.assertEqual(truncated, "    return False")

    def test_truncate_generated_text_uses_default_stop_tokens_for_gsm8k(self):
        doc = {
            "question": "If Alice has 3 apples...",
            "answer": "#### 5",
        }

        truncated = truncate_generated_text(
            "#### 5\nQuestion: next one",
            stop_tokens=["Question:"],
            is_instruct=True,
            doc=doc,
        )

        self.assertEqual(truncated, "#### 5")


if __name__ == "__main__":
    unittest.main()
