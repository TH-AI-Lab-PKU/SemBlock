import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from models.local_boundary_corrector import LocalBoundaryCorrector
from train_local_boundary_corrector import _train


def _run_train(rows, extra_lines=None, extra_args=None):
    temp_dir = tempfile.TemporaryDirectory()
    train_path = Path(temp_dir.name) / "train.jsonl"
    output_path = Path(temp_dir.name) / "model.pt"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
        if extra_lines:
            for line in extra_lines:
                handle.write(line)
                if not line.endswith("\n"):
                    handle.write("\n")

    script_path = REPO_ROOT / "llada" / "train_local_boundary_corrector.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--train-jsonl",
            str(train_path),
            "--output",
            str(output_path),
            *(extra_args or []),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output_path, temp_dir


class LocalBoundaryCorrectorTests(unittest.TestCase):
    def test_output_shapes(self):
        model = LocalBoundaryCorrector(input_dim=4, hidden_dim=8, delta_classes=5)
        features = torch.randn(2, 4)
        gate_logits, delta_logits = model(features)
        self.assertEqual(gate_logits.shape, (2, 2))
        self.assertEqual(delta_logits.shape, (2, 5))

    def test_train_script_saves_state_dict(self):
        rows = [
            {"features": [0.1, 0.2, 0.3, 0.4], "gate_target": 0, "delta_target": 1},
            {"features": [0.2, 0.1, 0.0, 0.3], "gate_target": 1, "delta_target": 3},
        ]
        expected_input_dim = len(rows[0]["features"])
        result, output_path, temp_dir = _run_train(rows, extra_lines=["\n"])
        try:
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())

            try:
                state_dict = torch.load(output_path, map_location="cpu", weights_only=True)
            except (TypeError, ValueError, RuntimeError):
                state_dict = torch.load(output_path, map_location="cpu")
            self.assertIsInstance(state_dict, dict)

            model = LocalBoundaryCorrector(
                input_dim=expected_input_dim,
                hidden_dim=128,
                delta_classes=5,
            )
            model.load_state_dict(state_dict)
            gate_logits, delta_logits = model(torch.zeros(2, expected_input_dim))
            self.assertEqual(gate_logits.shape, (2, 2))
            self.assertEqual(delta_logits.shape, (2, 5))
        finally:
            temp_dir.cleanup()

    def test_train_script_supports_gate_only_rows(self):
        rows = [
            {"features": [0.1, 0.2, 0.3], "gate_target": 1},
            {"features": [0.2, 0.1, 0.4], "gate_target": 0},
        ]
        result, output_path, temp_dir = _run_train(rows, extra_args=["--gate-only"])
        try:
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue(output_path.exists())
        finally:
            temp_dir.cleanup()

    def test_rejects_ragged_features(self):
        rows = [
            {"features": [0.1, 0.2], "gate_target": 0, "delta_target": 1},
            {"features": [0.2], "gate_target": 1, "delta_target": 2},
        ]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("length", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_empty_features(self):
        rows = [{"features": [], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("non-empty", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_invalid_gate_target(self):
        rows = [{"features": [0.1, 0.2], "gate_target": 2, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("gate_target", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_non_integer_gate_target(self):
        rows = [
            {"features": [0.1, 0.2], "gate_target": 0.0, "delta_target": 1},
            {"features": [0.1, 0.2], "gate_target": "0", "delta_target": 1},
        ]
        for row in rows:
            result, _, temp_dir = _run_train([row])
            try:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("gate_target", result.stderr)
            finally:
                temp_dir.cleanup()

    def test_rejects_invalid_delta_target(self):
        rows = [
            {"features": [0.1, 0.2], "gate_target": 1, "delta_target": -1},
            {"features": [0.1, 0.2], "gate_target": 1, "delta_target": 5},
        ]
        for row in rows:
            result, _, temp_dir = _run_train([row])
            try:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("delta_target", result.stderr)
            finally:
                temp_dir.cleanup()

    def test_rejects_non_integer_delta_target(self):
        rows = [
            {"features": [0.1, 0.2], "gate_target": 1, "delta_target": 1.0},
            {"features": [0.1, 0.2], "gate_target": 1, "delta_target": "1"},
        ]
        for row in rows:
            result, _, temp_dir = _run_train([row])
            try:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("delta_target", result.stderr)
            finally:
                temp_dir.cleanup()

    def test_rejects_boolean_gate_target(self):
        rows = [{"features": [0.1, 0.2], "gate_target": True, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("gate_target", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_boolean_delta_target(self):
        rows = [{"features": [0.1, 0.2], "gate_target": 0, "delta_target": False}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("delta_target", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_non_numeric_feature(self):
        rows = [{"features": [0.1, "bad"], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("numeric", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_non_vector_features(self):
        rows = [
            {"features": "abc", "gate_target": 0, "delta_target": 1},
            {"features": None, "gate_target": 0, "delta_target": 1},
            {"features": {"a": 1}, "gate_target": 0, "delta_target": 1},
        ]
        for row in rows:
            result, _, temp_dir = _run_train([row])
            try:
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("features", result.stderr)
                self.assertIn("vector", result.stderr)
            finally:
                temp_dir.cleanup()

    def test_rejects_boolean_feature(self):
        rows = [{"features": [True, 0.1], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("numeric", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_non_finite_feature(self):
        rows = [{"features": [float("inf"), 0.1], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("finite", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_nan_feature(self):
        rows = [{"features": [float("nan"), 0.1], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("features", result.stderr)
            self.assertIn("finite", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_malformed_json_line(self):
        rows = [{"features": [0.1, 0.2], "gate_target": 0, "delta_target": 1}]
        result, _, temp_dir = _run_train(rows, extra_lines=["{bad json"] )
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("JSON", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_non_dict_row(self):
        rows = [[1, 2, 3]]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Row 0", result.stderr)
            self.assertIn("object", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_missing_keys(self):
        rows = [{"features": [0.1, 0.2]}]
        result, _, temp_dir = _run_train(rows)
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing keys", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_rejects_blank_only_training_file(self):
        result, _, temp_dir = _run_train([], extra_lines=["\n", "   " ])
        try:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No training rows", result.stderr)
        finally:
            temp_dir.cleanup()

    def test_train_rejects_empty_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.pt"
            with self.assertRaisesRegex(ValueError, "training rows"):
                _train([], output_path)


if __name__ == "__main__":
    unittest.main()
