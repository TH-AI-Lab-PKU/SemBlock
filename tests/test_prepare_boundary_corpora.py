import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from prepare_boundary_corpora import (
    build_balanced_combined_outputs,
    compute_split_counts,
    enrich_task_usable_record,
    prepare_math_v2_corpora,
    preprocess_codesearchnet,
    should_keep_lean_workbook_math_v2_row,
    split_codesearchnet_code,
    split_math_text,
)


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class PrepareBoundaryBalancedTests(unittest.TestCase):
    def test_compute_split_counts_for_three_one_one(self):
        self.assertEqual(compute_split_counts(275, (3, 1, 1)), {"train": 165, "valid": 55, "test": 55})

    def test_build_balanced_combined_outputs_resamples_across_all_dataset_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            processed = root / "processed"

            write_jsonl(
                processed / "gum" / "train.jsonl",
                [
                    {"source": "gum", "split": "train", "doc_id": "g1", "segments": ["a", "b"]},
                    {"source": "gum", "split": "train", "doc_id": "g2", "segments": ["a", "b"]},
                ],
            )
            write_jsonl(
                processed / "gum" / "valid.jsonl",
                [
                    {"source": "gum", "split": "valid", "doc_id": "g3", "segments": ["a", "b"]},
                    {"source": "gum", "split": "valid", "doc_id": "g4", "segments": ["a", "b"]},
                ],
            )
            write_jsonl(
                processed / "juice" / "train.jsonl",
                [
                    {"source": "juice", "split": "train", "notebook_path": "j1.ipynb", "segments": ["x", "y"]},
                    {"source": "juice", "split": "train", "notebook_path": "j2.ipynb", "segments": ["x", "y"]},
                ],
            )
            write_jsonl(
                processed / "juice" / "valid.jsonl",
                [
                    {"source": "juice", "split": "valid", "notebook_path": "j3.ipynb", "segments": ["x", "y"]},
                ],
            )

            outputs = build_balanced_combined_outputs(
                root=root,
                datasets=["gum", "juice"],
                output_subdir="combined_balanced",
                split_ratios=(1, 1, 1),
            )

            self.assertEqual(sorted(outputs.keys()), ["test", "train", "valid"])

            for split_name in ("train", "valid", "test"):
                rows = read_jsonl(outputs[split_name])
                self.assertEqual(len(rows), 2)
                self.assertEqual(Counter(row["source"] for row in rows), Counter({"gum": 1, "juice": 1}))
                self.assertTrue(all(row["split"] == split_name for row in rows))
                self.assertTrue(all("original_split" in row for row in rows))


class TaskUsableBoundaryRecordTests(unittest.TestCase):
    def test_split_math_text_prioritizes_equations_and_final_answer_markers(self):
        segments = split_math_text(
            "We simplify the expression.\n2x + 2 = 10\nThus the final answer is\n\\boxed{4}"
        )
        self.assertIn("2x + 2 = 10", segments)
        self.assertTrue(any("\\boxed{4}" in segment for segment in segments))
        self.assertFalse(any(segment.strip().lower() in {"thus", "therefore", "so"} for segment in segments))

    def test_enrich_task_usable_record_marks_code_safe_commit_boundaries(self):
        enriched = enrich_task_usable_record(
            {
                "source": "codesearchnet",
                "split": "train",
                "has_docstring": True,
                "segments": [
                    "Return the sum.",
                    "def add(a, b):",
                    "return a + b",
                ],
            },
            training_mode="separate",
        )
        self.assertEqual(enriched["task_family"], "code")
        self.assertEqual(enriched["training_mode"], "separate")
        self.assertEqual(enriched["label_schema_version"], "task_usable_v1")
        self.assertIn("internal_step_boundary", enriched["segment_boundary_types"][0])
        self.assertIn("safe_commit_boundary", enriched["segment_boundary_types"][1])
        self.assertIn("safe_commit_boundary", enriched["segment_boundary_types"][2])

    def test_enrich_task_usable_record_marks_final_answer_anchor_for_formal_math(self):
        enriched = enrich_task_usable_record(
            {
                "source": "proofnet",
                "split": "valid",
                "segments": [
                    "Assume x is an integer.",
                    "x^2 = 4.",
                    "\\boxed{2}",
                ],
            },
            training_mode="separate",
        )
        self.assertEqual(enriched["task_family"], "formal_math")
        self.assertIn("internal_step_boundary", enriched["segment_boundary_types"][0])
        self.assertIn("internal_step_boundary", enriched["segment_boundary_types"][1])
        self.assertIn("final_answer_anchor", enriched["segment_boundary_types"][2])


class CodeSearchNetSegmentationTests(unittest.TestCase):
    def test_preprocess_codesearchnet_drops_docstring_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "raw" / "codesearchnet" / "extracted" / "python" / "train" / "sample.jsonl"
            write_jsonl(
                source_path,
                [
                    {
                        "repo": "demo/repo",
                        "path": "demo.py",
                        "func_name": "solve",
                        "docstring": "Return the running total.",
                        "code": (
                            "def solve(items):\n"
                            '    """Return the running total."""\n'
                            "    total = 0\n"
                            "    for item in items:\n"
                            "        total += item\n"
                            "    return total\n"
                        ),
                    }
                ],
            )

            outputs = preprocess_codesearchnet(root, ["python"])
            rows = read_jsonl(outputs["train"])
            self.assertEqual(len(rows), 1)
            row = rows[0]

            self.assertTrue(row["has_docstring"])
            self.assertEqual(row["segments"][0], "def solve(items):")
            self.assertNotIn("Return the running total.", row["segments"])
            self.assertEqual(row["code_segment_count"], len(row["segments"]))

    def test_split_codesearchnet_code_keeps_only_large_python_blocks(self):
        code = """def solve(items):
    total = 0
    count = len(items)
    if count > 0:
        total += count
    for item in items:
        total += item
    return total
"""
        method, quality, segments = split_codesearchnet_code("python", code)

        self.assertEqual(method, "ast")
        self.assertEqual(quality, "high")
        self.assertEqual(segments[0], "def solve(items):")
        self.assertTrue(any(segment.lstrip().startswith("if count > 0") for segment in segments[1:]))
        self.assertTrue(any(segment.lstrip().startswith("for item in items") for segment in segments[1:]))
        self.assertTrue(any("total = 0" in segment and "count = len(items)" in segment for segment in segments[1:]))
        self.assertTrue(any("return total" in segment for segment in segments[1:]))
        self.assertFalse(any(segment.strip() == "total = 0" for segment in segments[1:]))
        self.assertFalse(any(segment.strip() == "count = len(items)" for segment in segments[1:]))
        self.assertFalse(any(segment.lstrip().startswith("return total") for segment in segments[1:]))

    def test_split_codesearchnet_code_keeps_only_large_javascript_blocks(self):
        code = """function solve(items) {
  let total = 0;
  const count = items.length;
  if (count > 0) {
    total += count;
  }
  for (const item of items) {
    total += item;
  }
  return total;
}
"""
        method, quality, segments = split_codesearchnet_code("javascript", code)

        self.assertIn(method, {"heuristic", "heuristic_coarsened"})
        self.assertIn(quality, {"fallback", "coarsened"})
        self.assertTrue(segments[0].startswith("function solve(items)"))
        self.assertTrue(any(segment.lstrip().startswith("if (count > 0)") for segment in segments[1:]))
        self.assertTrue(any(segment.lstrip().startswith("for (const item of items)") for segment in segments[1:]))
        self.assertTrue(any("let total = 0;" in segment and "const count = items.length;" in segment for segment in segments[1:]))
        self.assertTrue(any("return total;" in segment for segment in segments[1:]))
        self.assertFalse(any(segment.strip() == "let total = 0;" for segment in segments[1:]))
        self.assertFalse(any(segment.strip() == "const count = items.length;" for segment in segments[1:]))
        self.assertFalse(any(segment.lstrip().startswith("return total;") for segment in segments[1:]))


    def test_split_codesearchnet_code_filters_internal_docstring_blocks(self):
        code = '''def solve(items):
    """Return the running total."""
    total = 0
    """temporary implementation note that should not become a segment"""
    for item in items:
        total += item
    return total
'''
        method, quality, segments = split_codesearchnet_code("python", code)

        self.assertEqual(method, "ast")
        self.assertEqual(quality, "high")
        self.assertEqual(segments[0], "def solve(items):")
        self.assertFalse(any("temporary implementation note" in segment for segment in segments))
        self.assertFalse(any(segment.strip().startswith('"""') for segment in segments[1:]))
        self.assertTrue(any("total = 0" in segment for segment in segments[1:]))
        self.assertTrue(any(segment.lstrip().startswith("for item in items") for segment in segments[1:]))

    def test_preprocess_codesearchnet_writes_to_versioned_output_subdir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "raw" / "codesearchnet" / "extracted" / "python" / "train" / "sample.jsonl"
            write_jsonl(
                source_path,
                [
                    {
                        "repo": "demo/repo",
                        "path": "demo.py",
                        "func_name": "solve",
                        "docstring": "Return the running total.",
                        "code": (
                            "def solve(items):\n"
                            "    total = 0\n"
                            "    return total\n"
                        ),
                    }
                ],
            )

            outputs = preprocess_codesearchnet(
                root,
                ["python"],
                output_subdir="codesearchnet_20260409_coarse_clean",
                dataset_version="2026-04-09-coarse-clean",
            )

            self.assertIn("codesearchnet_20260409_coarse_clean", str(outputs["train"]))
            rows = read_jsonl(outputs["train"])
            self.assertEqual(rows[0]["dataset_version"], "2026-04-09-coarse-clean")



    def test_split_codesearchnet_code_heuristic_strips_leading_docstring_after_commented_signature(self):
        code = (
            "def raise_(type_, value=None, traceback=None):  # pylint: disable=W0613\n"
            '    """\n'
            "    Does the same as ordinary ``raise`` with arguments do in Python 2.\n"
            '    """\n'
            "    prev_exc = sys.exc_info()[1]\n"
            "    if traceback:\n"
            "        raise value, None, traceback\n"
            "    return prev_exc\n"
        )
        method, quality, segments = split_codesearchnet_code("python", code)

        self.assertEqual(method, "heuristic")
        self.assertEqual(quality, "fallback")
        self.assertEqual(segments[0], "def raise_(type_, value=None, traceback=None):  # pylint: disable=W0613")
        self.assertFalse(any('"""' in segment for segment in segments))
        self.assertFalse(any("Does the same as ordinary" in segment for segment in segments))
        self.assertTrue(any("prev_exc = sys.exc_info()[1]" in segment for segment in segments[1:]))


class MathV2BoundaryCorpusTests(unittest.TestCase):
    def test_split_math_text_separates_display_math_followed_by_sentence_without_space(self):
        segments = split_math_text(
            "We get $$10\cdot 6\cdot 7=\boxed{420}.$$If we multiply directly, the product still rounds to 420."
        )

        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(any("\boxed{420}" in segment for segment in segments))
        self.assertTrue(any(segment.startswith("If we multiply directly") for segment in segments))

    def test_should_keep_lean_workbook_math_v2_row_requires_proof_and_reasoning_like_statement(self):
        keep_row = {
            "natural_language_statement": "First compute 2+3=5. Then note this is the total. Therefore \boxed{5}.",
            "proof": ["simp"],
        }
        drop_row = {
            "natural_language_statement": "Volume of sphere: 4/3 pi r^3",
            "proof": ["simp"],
        }

        keep, keep_stats = should_keep_lean_workbook_math_v2_row(keep_row)
        drop, drop_stats = should_keep_lean_workbook_math_v2_row(drop_row)

        self.assertTrue(keep)
        self.assertGreaterEqual(keep_stats["statement_segment_count"], 3)
        self.assertFalse(drop)
        self.assertEqual(drop_stats["kept_reason"], "statement_too_shallow")

    def test_split_math_text_breaks_single_sentence_multi_step_math_reasoning(self):
        segments = split_math_text(
            "There are 4 different ways to roll a 9 (3+6, 4+5, 5+4, 6+3), which makes the probability of rolling a 9 equal to \\dfrac{4}{36} = \oxed{\\dfrac{1}{9}}."
        )

        self.assertGreaterEqual(len(segments), 2)
        self.assertTrue(segments[0].startswith("There are 4 different ways to roll a 9"))
        self.assertTrue(any("\boxed{\dfrac{1}{9}}" in segment for segment in segments))

    def test_prepare_math_v2_corpora_filters_lean_and_adds_math_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lean_path = root / "raw" / "lean_workbook" / "lean_workbook.json"
            lean_path.parent.mkdir(parents=True, exist_ok=True)
            lean_path.write_text(
                json.dumps(
                    [
                        {
                            "split": "lean_workbook",
                            "tags": ["algebra"],
                            "natural_language_statement": "First compute 2+3=5. Then note this is the total. Therefore \boxed{5}.",
                            "formal_statement": "theorem keep : True := by trivial",
                            "proof": ["simp"],
                        },
                        {
                            "split": "lean_workbook",
                            "tags": ["geometry"],
                            "natural_language_statement": "Volume of sphere: 4/3 pi r^3",
                            "formal_statement": "theorem drop : True := by trivial",
                            "proof": ["simp"],
                        },
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            proofnet_dir = root / "raw" / "proofnet"
            proofnet_dir.mkdir(parents=True, exist_ok=True)
            for split_name in ("valid", "test"):
                write_jsonl(
                    proofnet_dir / f"{split_name}.jsonl",
                    [
                        {
                            "id": f"proof-{split_name}",
                            "nl_statement": "Prove that 1+1=2.",
                            "nl_proof": "\begin{proof} We add 1 and 1. Therefore \boxed{2}. \end{proof}",
                            "formal_statement": "theorem t : 1 + 1 = 2 := by norm_num",
                        }
                    ],
                )

            summary = prepare_math_v2_corpora(
                root,
                output_subdir="math_v2_full",
                pilot_output_subdir="math_v2_pilot",
                pilot_limits={"proofnet": 1, "lean_workbook": 1},
                seed=7,
            )

            lean_rows = []
            for path_str in summary["full_outputs"]["lean_workbook"].values():
                lean_rows.extend(read_jsonl(Path(path_str)))
            proof_rows = []
            for path_str in summary["full_outputs"]["proofnet"].values():
                proof_rows.extend(read_jsonl(Path(path_str)))

            self.assertEqual(len(lean_rows), 1)
            self.assertEqual(lean_rows[0]["source_dataset"], "lean_workbook")
            self.assertTrue(lean_rows[0]["is_filtered_from_lean"])
            self.assertEqual(lean_rows[0]["proof_style"], "filtered_lean_statement")
            self.assertTrue(lean_rows[0]["has_final_answer_anchor"])
            self.assertEqual(len(lean_rows[0]["segments"]), len(lean_rows[0]["segment_boundary_types"]))

            self.assertEqual(len(proof_rows), 2)
            self.assertTrue(all(row["source_dataset"] == "proofnet" for row in proof_rows))
            self.assertTrue(all(row["proof_style"] == "natural_language_proof" for row in proof_rows))
            self.assertTrue(all(not row["is_filtered_from_lean"] for row in proof_rows))
            self.assertTrue(Path(summary["audit_report_path"]).exists())
            self.assertTrue(summary["pilot_outputs"])

if __name__ == "__main__":
    unittest.main()
