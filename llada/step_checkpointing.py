from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def parse_step_list(value: str | Sequence[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        if not text:
            return []
        items = [item.strip() for item in text.replace("|", ",").split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in value if str(item).strip()]

    parsed: set[int] = set()
    for item in items:
        step = int(item)
        if step <= 0:
            raise ValueError("checkpoint steps must be positive integers")
        parsed.add(step)
    return sorted(parsed)


def should_save_step_checkpoint(
    *,
    optimizer_step: int,
    save_every_steps: int | None,
    checkpoint_steps: Iterable[int] | None,
) -> bool:
    step = int(optimizer_step)
    if step <= 0:
        return False

    if save_every_steps is not None:
        interval = int(save_every_steps)
        if interval <= 0:
            raise ValueError("save_every_steps must be a positive integer")
        if step % interval == 0:
            return True

    if checkpoint_steps is None:
        return False

    return step in {int(value) for value in checkpoint_steps}


def resolve_save_every_steps(
    *,
    save_every_steps: int | None,
    checkpoint_steps: Sequence[int] | None,
    max_train_steps: int | None,
    optimizer_steps_per_epoch: int,
) -> int | None:
    if save_every_steps is not None:
        interval = int(save_every_steps)
        if interval <= 0:
            raise ValueError("save_every_steps must be a positive integer")
        return interval

    if checkpoint_steps:
        return None

    total_steps_hint = int(max_train_steps) if max_train_steps is not None else int(optimizer_steps_per_epoch)
    total_steps_hint = max(1, total_steps_hint)
    return max(1, total_steps_hint // 5)


def resolve_training_epochs(
    *,
    epochs: int,
    max_train_steps: int | None,
    optimizer_steps_per_epoch: int,
) -> int:
    epoch_budget = int(epochs)
    if epoch_budget <= 0:
        raise ValueError("epochs must be a positive integer")

    if max_train_steps is None:
        return epoch_budget

    step_budget = int(max_train_steps)
    if step_budget <= 0:
        raise ValueError("max_train_steps must be a positive integer when provided")

    steps_per_epoch = max(1, int(optimizer_steps_per_epoch))
    return max(1, (step_budget + steps_per_epoch - 1) // steps_per_epoch)


def build_step_checkpoint_path(output_dir: str | Path, optimizer_step: int) -> Path:
    step = int(optimizer_step)
    if step <= 0:
        raise ValueError("optimizer_step must be positive")
    return Path(output_dir) / f"boundary_head_step_{step:06d}.pt"
