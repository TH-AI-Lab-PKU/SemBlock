from __future__ import annotations

import json
import os
from pathlib import Path

import evaluate as hf_evaluate


_SUBSET_ENV_VAR = "LLADA_MBPP_SUBSET_MANIFEST"

pass_at_k = hf_evaluate.load("code_eval")


def _load_subset_indices() -> list[int]:
    manifest_path = os.environ.get(_SUBSET_ENV_VAR)
    if not manifest_path:
        raise ValueError(f"{_SUBSET_ENV_VAR} must be set when running llada_mbpp_subset.")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    indices = payload.get("indices")
    if not isinstance(indices, list) or not indices:
        raise ValueError(f"Subset manifest {manifest_path} does not contain a non-empty indices list.")
    return [int(index) for index in indices]


def process_docs(dataset):
    indices = _load_subset_indices()
    if min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError("Subset manifest references indices outside the MBPP dataset size.")
    return dataset.select(indices)


def pass_at_1(references, predictions):
    return pass_at_k.compute(
        references=references,
        predictions=[predictions],
        k=[1],
    )[0]["pass@1"]


def list_fewshot_samples():
    return [
        {
            "task_id": 2,
            "text": "Write a function to find the similar elements from the given two tuple lists.",
            "code": "def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res) ",
            "test_list": [
                "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
                "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
                "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 3,
            "text": "Write a python function to identify non-prime numbers.",
            "code": "import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result",
            "test_list": [
                "assert is_not_prime(2) == False",
                "assert is_not_prime(10) == True",
                "assert is_not_prime(35) == True",
            ],
            "is_fewshot": True,
        },
        {
            "task_id": 4,
            "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
            "code": "import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums",
            "test_list": [
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
                "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
            ],
            "is_fewshot": True,
        },
    ]
