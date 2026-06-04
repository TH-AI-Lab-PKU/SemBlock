import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "llada"))

from step_checkpointing import (
    build_step_checkpoint_path,
    parse_step_list,
    resolve_save_every_steps,
    should_save_step_checkpoint,
)


class StepCheckpointScheduleTests(unittest.TestCase):
    def test_parse_step_list_handles_pipe_and_comma_values(self):
        self.assertEqual(parse_step_list("100|200|400"), [100, 200, 400])
        self.assertEqual(parse_step_list("50,150,300"), [50, 150, 300])

    def test_parse_step_list_normalizes_duplicates(self):
        self.assertEqual(parse_step_list("300|100|300|200"), [100, 200, 300])

    def test_should_save_step_checkpoint_uses_optimizer_steps_only(self):
        self.assertTrue(
            should_save_step_checkpoint(
                optimizer_step=100,
                save_every_steps=100,
                checkpoint_steps=None,
            )
        )
        self.assertFalse(
            should_save_step_checkpoint(
                optimizer_step=99,
                save_every_steps=100,
                checkpoint_steps=None,
            )
        )

    def test_should_save_step_checkpoint_honors_explicit_steps(self):
        self.assertTrue(
            should_save_step_checkpoint(
                optimizer_step=150,
                save_every_steps=None,
                checkpoint_steps=[50, 150, 300],
            )
        )
        self.assertFalse(
            should_save_step_checkpoint(
                optimizer_step=151,
                save_every_steps=None,
                checkpoint_steps=[50, 150, 300],
            )
        )

    def test_build_step_checkpoint_path_uses_zero_padded_name(self):
        checkpoint_path = build_step_checkpoint_path("/tmp/run", optimizer_step=42)

        self.assertEqual(checkpoint_path.name, "boundary_head_step_000042.pt")

    def test_resolve_save_every_steps_uses_explicit_interval_when_provided(self):
        self.assertEqual(
            resolve_save_every_steps(
                save_every_steps=120,
                checkpoint_steps=[],
                max_train_steps=1000,
                optimizer_steps_per_epoch=80,
            ),
            120,
        )

    def test_resolve_save_every_steps_uses_step_budget_by_default(self):
        self.assertEqual(
            resolve_save_every_steps(
                save_every_steps=None,
                checkpoint_steps=[],
                max_train_steps=1000,
                optimizer_steps_per_epoch=80,
            ),
            200,
        )

    def test_resolve_save_every_steps_uses_epoch_budget_when_max_train_steps_missing(self):
        self.assertEqual(
            resolve_save_every_steps(
                save_every_steps=None,
                checkpoint_steps=[],
                max_train_steps=None,
                optimizer_steps_per_epoch=80,
            ),
            16,
        )

    def test_resolve_save_every_steps_returns_none_when_explicit_checkpoint_steps_exist(self):
        self.assertIsNone(
            resolve_save_every_steps(
                save_every_steps=None,
                checkpoint_steps=[100, 200, 300],
                max_train_steps=1000,
                optimizer_steps_per_epoch=80,
            )
        )


if __name__ == "__main__":
    unittest.main()
