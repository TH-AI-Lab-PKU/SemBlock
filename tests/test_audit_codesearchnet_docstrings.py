import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from audit_codesearchnet_docstrings import classify_segment, summarize_jsonl


class CodeSearchNetAuditTests(unittest.TestCase):
    def test_classify_segment_marks_triple_quoted_blocks_as_leaks(self):
        self.assertEqual(classify_segment('"""temporary note"""'), "docstring_like")
        self.assertEqual(classify_segment("'''api description'''"), "docstring_like")
        self.assertIsNone(classify_segment("return total"))

    def test_summarize_jsonl_counts_rows_and_hits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "language": "python",
                                "segmentation_method": "ast",
                                "has_docstring": True,
                                "segments": ["def solve():", '"""oops"""'],
                            }
                        ),
                        json.dumps(
                            {
                                "language": "python",
                                "segmentation_method": "ast",
                                "has_docstring": True,
                                "segments": ["def keep():", "return 1"],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = summarize_jsonl(path)
            self.assertEqual(summary["total_rows"], 2)
            self.assertEqual(summary["rows_with_docstring_like_segments"], 1)
            self.assertEqual(summary["segment_hits"]["docstring_like"], 1)

    def test_classify_segment_ignores_inline_triple_quoted_literals(self):
        self.assertIsNone(classify_segment('print("""hello""")'))
        self.assertIsNone(classify_segment("return builder('''sql''')"))

    def test_classify_segment_ignores_multiline_inline_triple_quoted_literals(self):
        segment = (
            "if args.help:\n"
            "        print(\"\"\"Ant Globs\n"
            "=========\n"
            "\"\"\")"
        )
        self.assertIsNone(classify_segment(segment))


if __name__ == "__main__":
    unittest.main()
