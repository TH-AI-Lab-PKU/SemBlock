import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from analyze_humaneval_failures import analyze_samples_file


class HumanEvalFailureAnalysisTests(unittest.TestCase):
    def test_analyze_samples_file_counts_assertion_and_syntax_failures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples_path = Path(tmpdir) / "samples.jsonl"

            rows = [
                {
                    "doc": {
                        "task_id": "HumanEval/0",
                        "entry_point": "foo",
                    },
                    "target": "assert foo() == 1",
                    "filtered_resps": [["def foo():\n    return 0"]],
                    "pass@1": 0.0,
                },
                {
                    "doc": {
                        "task_id": "HumanEval/1",
                        "entry_point": "bar",
                    },
                    "target": "assert bar() == 1",
                    "filtered_resps": [["def bar(:\n    return 1"]],
                    "pass@1": 0.0,
                },
                {
                    "doc": {
                        "task_id": "HumanEval/2",
                        "entry_point": "baz",
                    },
                    "target": "assert baz() == 1",
                    "filtered_resps": [["def baz():\n    return 1"]],
                    "pass@1": 1.0,
                },
            ]
            with open(samples_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            summary = analyze_samples_file(samples_path)

            self.assertEqual(summary["total_samples"], 3)
            self.assertEqual(summary["failed_samples"], 2)
            self.assertEqual(summary["failure_type_counts"]["AssertionError"], 1)
            self.assertEqual(summary["failure_type_counts"]["SyntaxError"], 1)


if __name__ == "__main__":
    unittest.main()
