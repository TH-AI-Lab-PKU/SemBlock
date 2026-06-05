from __future__ import annotations

from typing import Optional

import torch


def _load_torch_npu() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu")


def is_npu_available() -> bool:
    if not _load_torch_npu():
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def default_device(index: Optional[int] = None) -> str:
    if is_npu_available():
        return "npu" if index is None else f"npu:{index}"
    return "cpu"


def is_npu_device(device: str | torch.device) -> bool:
    return str(device).startswith("npu")


def accelerator_dtype(device: str | torch.device) -> torch.dtype:
    return torch.bfloat16 if is_npu_device(device) else torch.float32


def manual_seed_all(seed: int) -> None:
    if is_npu_available():
        torch.npu.manual_seed_all(seed)


def empty_cache() -> None:
    if is_npu_available():
        torch.npu.empty_cache()


def autocast_dtype() -> torch.dtype:
    if is_npu_available() and hasattr(torch.npu, "get_autocast_dtype"):
        return torch.npu.get_autocast_dtype()
    return torch.bfloat16
