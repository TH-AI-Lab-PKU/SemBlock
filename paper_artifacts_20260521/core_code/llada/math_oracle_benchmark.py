from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_gsm8k_cot_prompt(question: str, fewshot_examples: Sequence[Mapping[str, str]]) -> str:
    parts: List[str] = []
    for example in fewshot_examples:
        answer = str(example.get("answer") or example.get("target") or "").strip()
        parts.append(
            "\n".join(
                [
                    f"Q: {str(example.get('question', '')).strip()}",
                    "A: Let's think step by step.",
                    answer,
                ]
            ).rstrip()
        )
    parts.append(f"Q: {question.strip()}\nA: Let's think step by step.")
    return "\n\n".join(part for part in parts if part)


def build_math_cot_prompt(problem: str, fewshot_examples: Sequence[Mapping[str, str]]) -> str:
    parts: List[str] = []
    for example in fewshot_examples:
        solution = str(example.get("solution") or example.get("answer") or "").strip()
        parts.append(f"Problem: {str(example.get('problem', '')).strip()}\nAnswer: {solution}".rstrip())
    parts.append(f"Problem: {problem.strip()}\nAnswer:")
    return "\n\n".join(part for part in parts if part)


def compute_oracle_generation_budget(oracle_block_sizes: Sequence[int], default_gen_length: int) -> int:
    cleaned_sizes = [int(size) for size in oracle_block_sizes if int(size) > 0]
    if not cleaned_sizes:
        return int(default_gen_length)
    return min(int(default_gen_length), sum(cleaned_sizes))


def summarize_oracle_documents(documents: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    sample_count = len(documents)
    all_block_lengths: List[int] = []
    block_counts: List[int] = []
    block_count_distribution: Counter[int] = Counter()
    docs_with_boundary = 0
    docs_with_final_anchor = 0

    for document in documents:
        block_sizes = [int(size) for size in (document.get("oracle_block_sizes") or []) if int(size) > 0]
        block_count = len(block_sizes)
        block_counts.append(block_count)
        block_count_distribution[block_count] += 1
        all_block_lengths.extend(block_sizes)
        if block_count > 1:
            docs_with_boundary += 1
        if bool(document.get("has_final_answer_anchor")):
            docs_with_final_anchor += 1

    avg_block_length = (sum(all_block_lengths) / len(all_block_lengths)) if all_block_lengths else 0.0
    avg_block_count = (sum(block_counts) / sample_count) if sample_count else 0.0
    boundary_coverage_rate = (docs_with_boundary / sample_count) if sample_count else 0.0
    final_answer_anchor_hit_rate = (docs_with_final_anchor / sample_count) if sample_count else 0.0

    return {
        "sample_count": sample_count,
        "avg_block_length": avg_block_length,
        "avg_block_count": avg_block_count,
        "block_count_distribution": dict(sorted(block_count_distribution.items())),
        "boundary_coverage_rate": boundary_coverage_rate,
        "final_answer_anchor_hit_rate": final_answer_anchor_hit_rate,
    }


def summarize_generation_records(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    sample_count = len(records)
    correct_count = 0
    total_nfe = 0.0
    total_block_count = 0.0
    all_block_lengths: List[int] = []

    for record in records:
        if bool(record.get("is_correct")):
            correct_count += 1
        nfe_history = [int(value) for value in (record.get("nfe_history") or [])]
        block_history = [int(value) for value in (record.get("block_history") or []) if int(value) > 0]
        total_nfe += float(sum(nfe_history))
        total_block_count += float(len(block_history))
        all_block_lengths.extend(block_history)

    exact_match = (correct_count / sample_count) if sample_count else 0.0
    avg_nfe = (total_nfe / sample_count) if sample_count else 0.0
    avg_generated_block_count = (total_block_count / sample_count) if sample_count else 0.0
    avg_generated_block_length = (sum(all_block_lengths) / len(all_block_lengths)) if all_block_lengths else 0.0

    return {
        "sample_count": sample_count,
        "correct_count": correct_count,
        "exact_match": exact_match,
        "strict_match": exact_match,
        "avg_nfe": avg_nfe,
        "avg_generated_block_count": avg_generated_block_count,
        "avg_generated_block_length": avg_generated_block_length,
    }


def load_gsm8k_cot_fewshots(num_fewshot: int) -> List[Dict[str, str]]:
    import lm_eval
    import yaml

    path = Path(lm_eval.__file__).resolve().parent / "tasks" / "gsm8k" / "gsm8k-cot.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    samples = config.get("fewshot_config", {}).get("samples", [])
    fewshots: List[Dict[str, str]] = []
    for sample in samples[: max(0, num_fewshot)]:
        fewshots.append(
            {
                "question": str(sample.get("question", "")).strip(),
                "answer": str(sample.get("target", "")).strip(),
            }
        )
    return fewshots


def normalize_problem_text_for_overlap(text: object) -> str:
    normalized = str(text or "").lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def select_math_fewshots(
    rows: Sequence[Mapping[str, object]],
    num_fewshot: int,
    excluded_problem_texts: Optional[Sequence[str]] = None,
) -> List[Dict[str, str]]:
    if num_fewshot <= 0:
        return []

    excluded_norms = {
        normalize_problem_text_for_overlap(text)
        for text in (excluded_problem_texts or [])
        if normalize_problem_text_for_overlap(text)
    }

    selected: List[Dict[str, str]] = []
    seen_subjects = set()

    for row in rows:
        problem = str(row.get("problem", "")).strip()
        if not problem:
            continue
        if normalize_problem_text_for_overlap(problem) in excluded_norms:
            continue
        subject = str(row.get("subject") or "").strip().lower()
        if subject in seen_subjects:
            continue
        selected.append(
            {
                "problem": problem,
                "solution": str(row.get("solution", "")).strip(),
            }
        )
        seen_subjects.add(subject)
        if len(selected) >= num_fewshot:
            return selected

    for row in rows:
        problem = str(row.get("problem", "")).strip()
        if not problem:
            continue
        if normalize_problem_text_for_overlap(problem) in excluded_norms:
            continue
        candidate = {
            "problem": problem,
            "solution": str(row.get("solution", "")).strip(),
        }
        if candidate in selected:
            continue
        selected.append(candidate)
        if len(selected) >= num_fewshot:
            break

    return selected
