import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from filter_boundary_jsonl import filter_boundary_jsonl


class PythonOnlyFilterTests(unittest.TestCase):
    def test_filter_boundary_jsonl_keeps_only_python_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            input_path = tmpdir_path / "input.jsonl"
            output_path = tmpdir_path / "python_only.jsonl"
            metadata_path = tmpdir_path / "python_only.metadata.json"

            rows = [
                {"language": "python", "segments": ["def a():", " return 1"]},
                {"language": "java", "segments": ["class A {}", " // noop"]},
                {"language": "python", "segments": ["def b():", " return 2"]},
            ]
            with open(input_path, "w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            summary = filter_boundary_jsonl(
                input_path=input_path,
                output_path=output_path,
                language="python",
                metadata_path=metadata_path,
            )

            written_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(len(written_rows), 2)
            self.assertTrue(all(row["language"] == "python" for row in written_rows))
            self.assertEqual(summary["written_examples"], 2)
            self.assertTrue(metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
