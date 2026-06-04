import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from humaneval_subset import (
    build_humaneval_subset_manifest,
    select_humaneval_indices,
)


class HumanEvalSubsetTests(unittest.TestCase):
    def test_stratified_subset_is_reproducible_for_fixed_seed(self):
        first = select_humaneval_indices(strategy="stratified", sample_size=50, seed=20260416)
        second = select_humaneval_indices(strategy="stratified", sample_size=50, seed=20260416)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)

    def test_stratified_subset_is_not_first_fifty_prefix(self):
        indices = select_humaneval_indices(strategy="stratified", sample_size=50, seed=20260416)

        self.assertNotEqual(indices, list(range(50)))
        self.assertTrue(any(index >= 50 for index in indices))
        self.assertTrue(any(index >= 100 for index in indices))

    def test_uniform_subset_is_reproducible_for_fixed_seed(self):
        first = select_humaneval_indices(strategy="uniform", sample_size=50, seed=7)
        second = select_humaneval_indices(strategy="uniform", sample_size=50, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)

    def test_manifest_records_subset_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "subset_manifest.json"
            manifest = build_humaneval_subset_manifest(
                output_path=manifest_path,
                subset_label="screening",
                strategy="stratified",
                sample_size=50,
                seed=1234,
            )

            self.assertTrue(manifest_path.exists())
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["subset_label"], "screening")
            self.assertEqual(loaded["strategy"], "stratified")
            self.assertEqual(loaded["sample_size"], 50)
            self.assertEqual(loaded["seed"], 1234)
            self.assertEqual(loaded["indices"], manifest["indices"])


if __name__ == "__main__":
    unittest.main()
