from __future__ import annotations

import json
import os
from pathlib import Path

import evaluate as hf_evaluate


_SUBSET_ENV_VAR = "LLADA_HUMANEVAL_SUBSET_MANIFEST"

compute_ = hf_evaluate.load("code_eval")


def _load_subset_indices() -> list[int]:
    manifest_path = os.environ.get(_SUBSET_ENV_VAR)
    if not manifest_path:
        raise ValueError(
            f"{_SUBSET_ENV_VAR} must be set when running the llada_humaneval_subset task."
        )
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Subset manifest {manifest_path} does not contain a non-empty indices list.")
    return [int(index) for index in indices]


def process_docs(dataset):
    indices = _load_subset_indices()
    if max(indices) >= len(dataset):
        raise ValueError("Subset manifest references indices outside the HumanEval dataset size.")
    return dataset.select(indices)


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    results = compute_.compute(
        references=references,
        predictions=predictions,
        k=k,
    )
    return results[0]


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + resp for resp in response_list] for response_list, doc in zip(resps, docs)]
