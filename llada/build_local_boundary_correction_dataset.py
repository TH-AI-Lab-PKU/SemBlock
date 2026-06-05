from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

try:
    from .accelerator_utils import default_device, manual_seed_all
    from .math_oracle_benchmark import build_gsm8k_cot_prompt, compute_oracle_generation_budget, load_gsm8k_cot_fewshots
    from .math_oracle_utils import build_math_oracle_document, extract_gsm8k_answer, is_gsm8k_correct
    from .oracle_boundary_correction_data import (
        build_boundary_correction_record,
        choose_best_delta_label,
        classify_keep_vs_adjust,
    )
except ImportError:  # pragma: no cover
    from accelerator_utils import default_device, manual_seed_all
    from math_oracle_benchmark import build_gsm8k_cot_prompt, compute_oracle_generation_budget, load_gsm8k_cot_fewshots
    from math_oracle_utils import build_math_oracle_document, extract_gsm8k_answer, is_gsm8k_correct
    from oracle_boundary_correction_data import (
        build_boundary_correction_record,
        choose_best_delta_label,
        classify_keep_vs_adjust,
    )


DELTA_CLASS_VALUES = (-2, -1, 0, 1, 2)
DEFAULT_MODEL_PATH = "/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "experiments" / "semantic_boundary_indep"
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_STEPS = 16
DEFAULT_GEN_LENGTH = 512
DEFAULT_THRESHOLD = 0.9
DEFAULT_MASK_ID = 126336
DEFAULT_REMASKING = "low_confidence"
LABEL_OBJECTIVE_EXACT_MATCH_BINARY = "exact_match_binary"
LABEL_OBJECTIVE_NUMERIC_MARGIN = "numeric_margin"
SUPPORTED_LABEL_OBJECTIVES = (
    LABEL_OBJECTIVE_EXACT_MATCH_BINARY,
    LABEL_OBJECTIVE_NUMERIC_MARGIN,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    manual_seed_all(seed)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def apply_single_boundary_delta(
    oracle_block_sizes: Sequence[int],
    *,
    boundary_index: int,
    delta: int,
) -> List[int]:
    sizes = [max(1, int(size)) for size in list(oracle_block_sizes or [])]
    index = int(boundary_index)
    if index < 0 or index >= len(sizes):
        raise IndexError(f"boundary_index out of range: {boundary_index}")
    sizes[index] = max(1, sizes[index] + int(delta))
    return sizes


def delta_to_target_class(delta: int) -> int:
    normalized_delta = int(delta)
    if normalized_delta not in DELTA_CLASS_VALUES:
        raise ValueError(f"Unsupported delta value: {delta}")
    return DELTA_CLASS_VALUES.index(normalized_delta)


def gate_to_target_class(best_delta: int) -> int:
    gate_label = classify_keep_vs_adjust(best_delta)
    return 0 if gate_label == "keep" else 1


def feature_tensor_to_list(feature_vector: torch.Tensor) -> List[float]:
    if isinstance(feature_vector, torch.Tensor):
        values = feature_vector.detach().cpu().view(-1).tolist()
    else:
        values = list(feature_vector)
    return [round(float(value), 6) for value in values]


def build_training_row(
    *,
    doc: Mapping[str, object],
    correction_record: Mapping[str, object],
    feature_vector: torch.Tensor,
) -> Dict[str, object]:
    best_delta = int(correction_record["best_delta"])
    return {
        "sample_id": str(correction_record["sample_id"]),
        "source_dataset": str(doc.get("source_dataset") or "gsm8k"),
        "boundary_index": int(correction_record["boundary_index"]),
        "features": feature_tensor_to_list(feature_vector),
        "gate_target": gate_to_target_class(best_delta),
        "delta_target": delta_to_target_class(best_delta),
        "best_delta": best_delta,
        "delta_scores": dict(correction_record.get("delta_scores") or {}),
        "oracle_block_sizes": [int(size) for size in list(doc.get("oracle_block_sizes") or [])],
        "has_final_answer_anchor": bool(doc.get("has_final_answer_anchor")),
    }


def split_rows_by_document(
    rows: Sequence[Mapping[str, object]],
    *,
    valid_doc_count: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    ordered_doc_ids: List[str] = []
    seen = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in seen:
            seen.add(sample_id)
            ordered_doc_ids.append(sample_id)
    valid_doc_count = max(0, min(int(valid_doc_count), len(ordered_doc_ids)))
    valid_doc_ids = set(ordered_doc_ids[-valid_doc_count:]) if valid_doc_count else set()

    train_rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    for row in rows:
        target = valid_rows if str(row.get("sample_id") or "") in valid_doc_ids else train_rows
        target.append(dict(row))
    return train_rows, valid_rows


def parse_gsm8k_numeric_value(text: str) -> float | None:
    answer = extract_gsm8k_answer(str(text))
    if answer is None:
        return None
    try:
        return float(answer)
    except (TypeError, ValueError):
        return None


def score_gsm8k_numeric_margin(gold_solution: str, prediction: str) -> float:
    if is_gsm8k_correct(str(gold_solution), str(prediction)):
        return 1.0
    gold_value = parse_gsm8k_numeric_value(gold_solution)
    predicted_value = parse_gsm8k_numeric_value(prediction)
    if gold_value is None or predicted_value is None:
        return 0.0
    relative_error = abs(predicted_value - gold_value) / max(abs(gold_value), 1.0)
    closeness = 1.0 / (1.0 + relative_error)
    return 0.2 + 0.6 * closeness


def score_gsm8k_prediction(
    gold_solution: str,
    prediction: str,
    *,
    objective: str = LABEL_OBJECTIVE_EXACT_MATCH_BINARY,
) -> float:
    normalized_objective = str(objective or LABEL_OBJECTIVE_EXACT_MATCH_BINARY).strip().lower()
    if normalized_objective == LABEL_OBJECTIVE_EXACT_MATCH_BINARY:
        return 1.0 if is_gsm8k_correct(str(gold_solution), str(prediction)) else 0.0
    if normalized_objective == LABEL_OBJECTIVE_NUMERIC_MARGIN:
        return score_gsm8k_numeric_margin(str(gold_solution), str(prediction))
    raise ValueError(f"Unsupported GSM8K scoring objective: {objective}")


def collect_boundary_delta_scores(
    *,
    doc: Mapping[str, object],
    boundary_index: int,
    candidate_deltas: Sequence[int],
    candidate_scorer,
) -> Dict[int, float]:
    oracle_block_sizes = [int(size) for size in list(doc.get("oracle_block_sizes") or [])]
    delta_scores: Dict[int, float] = {}
    for raw_delta in list(candidate_deltas or []):
        delta = int(raw_delta)
        candidate_block_sizes = apply_single_boundary_delta(
            oracle_block_sizes,
            boundary_index=boundary_index,
            delta=delta,
        )
        delta_scores[delta] = float(candidate_scorer(candidate_block_sizes, delta))
    return delta_scores


def build_rows_for_document(
    *,
    doc: Mapping[str, object],
    feature_matrix: torch.Tensor,
    delta_scores_by_boundary: Mapping[int, Mapping[int, float]],
) -> List[Dict[str, object]]:
    if feature_matrix.dim() != 2:
        raise ValueError("feature_matrix must be 2D")
    rows: List[Dict[str, object]] = []
    for boundary_index in range(feature_matrix.shape[0]):
        delta_scores = dict((delta_scores_by_boundary or {}).get(boundary_index) or {0: 0.0})
        best_delta = choose_best_delta_label(delta_scores)
        correction_record = build_boundary_correction_record(
            sample_id=str(doc.get("sample_id") or ""),
            boundary_index=boundary_index,
            best_delta=best_delta,
            delta_scores=delta_scores,
        )
        rows.append(
            build_training_row(
                doc=doc,
                correction_record=correction_record,
                feature_vector=feature_matrix[boundary_index],
            )
        )
    return rows


def summarize_rows(rows: Sequence[Mapping[str, object]], *, split_name: str) -> Dict[str, object]:
    row_list = [dict(row) for row in rows]
    gate_counter = Counter(str(int(row["gate_target"])) for row in row_list)
    delta_counter = Counter(str(int(row["delta_target"])) for row in row_list)
    document_ids = []
    seen = set()
    for row in row_list:
        sample_id = str(row.get("sample_id") or "")
        if sample_id not in seen:
            seen.add(sample_id)
            document_ids.append(sample_id)
    return {
        "split_name": split_name,
        "row_count": len(row_list),
        "document_count": len(document_ids),
        "gate_target_distribution": dict(sorted(gate_counter.items())),
        "delta_target_distribution": dict(sorted(delta_counter.items())),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def write_dataset_artifacts(
    *,
    output_dir: Path,
    train_rows: Sequence[Mapping[str, object]],
    valid_rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"
    summary_path = output_dir / "summary.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(valid_path, valid_rows)
    summary = {
        "metadata": dict(metadata or {}),
        "train": summarize_rows(train_rows, split_name="train"),
        "valid": summarize_rows(valid_rows, split_name="valid"),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "train_jsonl": str(train_path),
        "valid_jsonl": str(valid_path),
        "summary_json": str(summary_path),
    }


def build_gsm8k_train_documents(*, tokenizer, num_fewshot: int, limit: int | None, gen_length: int) -> List[Dict[str, object]]:
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    if limit is not None:
        dataset = dataset.select(range(min(int(limit), len(dataset))))
    fewshots = load_gsm8k_cot_fewshots(num_fewshot)
    documents: List[Dict[str, object]] = []
    for index, row in enumerate(dataset):
        prompt_text = build_gsm8k_cot_prompt(str(row["question"]), fewshots)
        doc = build_math_oracle_document(
            sample_id=f"gsm8k/train/{index}",
            source_dataset="gsm8k",
            prompt_text=prompt_text,
            solution_text=str(row["answer"]),
            tokenizer=tokenizer,
            max_length=gen_length,
        )
        doc["question"] = str(row["question"])
        doc["gold_solution"] = str(row["answer"])
        documents.append(doc)
    return documents


def build_gsm8k_candidate_scorer_for_doc(
    *,
    doc: Mapping[str, object],
    model,
    tokenizer,
    is_instruct: bool,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    threshold: float,
    mask_id: int,
    remasking: str,
    gsm8k_landing_control: bool,
    gsm8k_landing_tail_lines: int,
    label_objective: str,
):
    from eval_math_oracle import decode_generation
    from eval_prompting import build_generation_prompt
    from generate_oracle_blocks import generate_oracle_blocks_prefix_cache

    prompt_text = str(doc["prompt_text"])
    user_input = build_generation_prompt(
        tokenizer,
        question=prompt_text,
        is_instruct=is_instruct,
        doc=doc,
    )
    input_ids = tokenizer(user_input)["input_ids"]
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    def candidate_scorer(candidate_block_sizes: Sequence[int], delta: int) -> float:
        sample_gen_length = compute_oracle_generation_budget(candidate_block_sizes, default_gen_length=gen_length)
        generated_tokens, _, _ = generate_oracle_blocks_prefix_cache(
            model,
            input_tensor,
            steps=steps,
            gen_length=sample_gen_length,
            init_block_length=block_length,
            temperature=0.0,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            delimiter_ids=[198],
            delimiter_threshold=float("inf"),
            block_strategy="oracle",
            task_type="math",
            tokenizer=tokenizer,
            oracle_block_sizes=list(candidate_block_sizes),
        )
        prediction = decode_generation(
            tokenizer=tokenizer,
            generated_tokens=generated_tokens,
            input_length=input_tensor.shape[1],
            stop_tokens=["Q:"],
            is_instruct=is_instruct,
            doc=doc,
            gsm8k_landing_control=gsm8k_landing_control,
            gsm8k_landing_tail_lines=gsm8k_landing_tail_lines,
        )
        return score_gsm8k_prediction(
            str(doc["gold_solution"]),
            prediction,
            objective=label_objective,
        )

    return candidate_scorer


def generate_rows_for_documents(
    *,
    documents: Sequence[Mapping[str, object]],
    model,
    tokenizer,
    is_instruct: bool,
    device: str,
    steps: int,
    gen_length: int,
    block_length: int,
    threshold: float,
    mask_id: int,
    remasking: str,
    gsm8k_landing_control: bool,
    gsm8k_landing_tail_lines: int,
    label_objective: str = LABEL_OBJECTIVE_EXACT_MATCH_BINARY,
    candidate_deltas: Sequence[int] = DELTA_CLASS_VALUES,
) -> List[Dict[str, object]]:
    from eval_math_oracle import build_local_corrector_feature_matrix_for_doc

    all_rows: List[Dict[str, object]] = []
    for doc in tqdm(documents, desc="gsm8k-local-correction-data"):
        oracle_block_sizes = list(doc.get("oracle_block_sizes") or [])
        if not oracle_block_sizes:
            continue
        feature_matrix = build_local_corrector_feature_matrix_for_doc(
            doc=doc,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        candidate_scorer = build_gsm8k_candidate_scorer_for_doc(
            doc=doc,
            model=model,
            tokenizer=tokenizer,
            is_instruct=is_instruct,
            device=device,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            threshold=threshold,
            mask_id=mask_id,
            remasking=remasking,
            gsm8k_landing_control=gsm8k_landing_control,
            gsm8k_landing_tail_lines=gsm8k_landing_tail_lines,
            label_objective=label_objective,
        )
        delta_scores_by_boundary: Dict[int, Dict[int, float]] = {}
        for boundary_index in range(len(oracle_block_sizes)):
            delta_scores_by_boundary[boundary_index] = collect_boundary_delta_scores(
                doc=doc,
                boundary_index=boundary_index,
                candidate_deltas=candidate_deltas,
                candidate_scorer=candidate_scorer,
            )
        all_rows.extend(
            build_rows_for_document(
                doc=doc,
                feature_matrix=feature_matrix,
                delta_scores_by_boundary=delta_scores_by_boundary,
            )
        )
    return all_rows


def write_dataset_audit_markdown(
    *,
    output_dir: Path,
    timestamp: str,
    args: argparse.Namespace,
    train_rows: Sequence[Mapping[str, object]],
    valid_rows: Sequence[Mapping[str, object]],
) -> Path:
    path = output_dir / f"dataset_audit_{timestamp}.md"
    train_summary = summarize_rows(train_rows, split_name="train")
    valid_summary = summarize_rows(valid_rows, split_name="valid")
    lines = [
        f"# GSM8K Local Corrector 数据审计 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- model_path: `{args.model_path}`",
        f"- output_dir: `{output_dir}`",
        f"- device: `{args.device}`",
        f"- gsm8k_limit: `{args.limit}`",
        f"- valid_doc_count: `{args.valid_doc_count}`",
        f"- candidate_deltas: `{list(DELTA_CLASS_VALUES)}`",
        f"- label_objective: `{args.label_objective}`",
        "- 本数据构建链路独立于 Ada，不使用 Ada entropy、不使用 Ada boundary head，也不复用 Ada 的生成决策。",
        "",
        "## 训练集统计",
        "",
        f"- row_count: `{train_summary['row_count']}`",
        f"- document_count: `{train_summary['document_count']}`",
        f"- gate_target_distribution: `{json.dumps(train_summary['gate_target_distribution'], ensure_ascii=False)}`",
        f"- delta_target_distribution: `{json.dumps(train_summary['delta_target_distribution'], ensure_ascii=False)}`",
        "",
        "## 验证集统计",
        "",
        f"- row_count: `{valid_summary['row_count']}`",
        f"- document_count: `{valid_summary['document_count']}`",
        f"- gate_target_distribution: `{json.dumps(valid_summary['gate_target_distribution'], ensure_ascii=False)}`",
        f"- delta_target_distribution: `{json.dumps(valid_summary['delta_target_distribution'], ensure_ascii=False)}`",
        "",
        "## 备注",
        "",
        "- 当前只构建 GSM8K correction data，暂不扩到 MATH。",
        "- 建议先在 smoke oracle 上验证标签定义是否改善，再决定是否扩大数据规模。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GSM8K local boundary correction data.")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--valid-doc-count", type=int, default=8)
    parser.add_argument("--gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--device", type=str, default=default_device(0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--remasking", type=str, default=DEFAULT_REMASKING)
    parser.add_argument("--label-objective", type=str, default=LABEL_OBJECTIVE_EXACT_MATCH_BINARY, choices=SUPPORTED_LABEL_OBJECTIVES)
    parser.add_argument("--gsm8k-landing-control", action="store_true")
    parser.add_argument("--gsm8k-landing-tail-lines", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    from eval_math_oracle import load_model_and_tokenizer

    args = parse_args()
    set_seed(args.seed)
    timestamp = timestamp_slug()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    model, tokenizer, is_instruct = load_model_and_tokenizer(args.model_path, args.device)
    documents = build_gsm8k_train_documents(
        tokenizer=tokenizer,
        num_fewshot=args.gsm8k_fewshot,
        limit=args.limit,
        gen_length=args.gen_length,
    )
    all_rows = generate_rows_for_documents(
        documents=documents,
        model=model,
        tokenizer=tokenizer,
        is_instruct=is_instruct,
        device=args.device,
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        threshold=args.threshold,
        mask_id=args.mask_id,
        remasking=args.remasking,
        gsm8k_landing_control=args.gsm8k_landing_control,
        gsm8k_landing_tail_lines=args.gsm8k_landing_tail_lines,
        label_objective=args.label_objective,
    )
    train_rows, valid_rows = split_rows_by_document(all_rows, valid_doc_count=args.valid_doc_count)
    paths = write_dataset_artifacts(
        output_dir=output_dir,
        train_rows=train_rows,
        valid_rows=valid_rows,
        metadata={
            "task": "gsm8k",
            "timestamp": timestamp,
            "limit": args.limit,
            "valid_doc_count": args.valid_doc_count,
            "candidate_deltas": list(DELTA_CLASS_VALUES),
            "label_objective": args.label_objective,
            "elapsed_seconds": round(time.time() - start_time, 3),
        },
    )
    audit_path = write_dataset_audit_markdown(
        output_dir=output_dir,
        timestamp=timestamp,
        args=args,
        train_rows=train_rows,
        valid_rows=valid_rows,
    )
    manifest_path = output_dir / f"run_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    final_summary = {
        "timestamp": timestamp,
        "output_dir": str(output_dir),
        "train_row_count": len(train_rows),
        "valid_row_count": len(valid_rows),
        "audit_path": str(audit_path),
        **paths,
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
