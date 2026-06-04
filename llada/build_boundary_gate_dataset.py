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
    from .math_oracle_benchmark import build_gsm8k_cot_prompt, compute_oracle_generation_budget, load_gsm8k_cot_fewshots
    from .math_oracle_utils import build_math_oracle_document, extract_gsm8k_answer, is_gsm8k_correct
except ImportError:  # pragma: no cover
    from math_oracle_benchmark import build_gsm8k_cot_prompt, compute_oracle_generation_budget, load_gsm8k_cot_fewshots
    from math_oracle_utils import build_math_oracle_document, extract_gsm8k_answer, is_gsm8k_correct


DEFAULT_MODEL_PATH = "/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_STEPS = 16
DEFAULT_GEN_LENGTH = 512
DEFAULT_BLOCK_LENGTH = 32
DEFAULT_THRESHOLD = 0.9
DEFAULT_MASK_ID = 126336
DEFAULT_REMASKING = "low_confidence"
LABEL_OBJECTIVE_CARRY_GATE_BINARY = "carry_gate_binary"
LABEL_OBJECTIVE_NUMERIC_MARGIN = "numeric_margin"
SUPPORTED_LABEL_OBJECTIVES = (
    LABEL_OBJECTIVE_CARRY_GATE_BINARY,
    LABEL_OBJECTIVE_NUMERIC_MARGIN,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def feature_tensor_to_list(feature_vector: torch.Tensor | Sequence[float]) -> List[float]:
    if isinstance(feature_vector, torch.Tensor):
        values = feature_vector.detach().cpu().view(-1).tolist()
    else:
        values = list(feature_vector)
    return [round(float(value), 6) for value in values]


def build_reference_boundary_carry_mask(*, transition_count: int) -> List[int]:
    transition_count = int(transition_count)
    if transition_count < 0:
        raise ValueError("transition_count must be >= 0")
    return [1 for _ in range(transition_count)]


def build_boundary_carry_masks(*, reference_mask: Sequence[int], transition_index: int) -> Tuple[List[int], List[int]]:
    normalized_reference = [1 if bool(value) else 0 for value in list(reference_mask or [])]
    transition_count = len(normalized_reference)
    transition_index = int(transition_index)
    if transition_index < 0 or transition_index >= transition_count:
        raise IndexError(f"transition_index out of range: {transition_index}")
    carry_on_mask = list(normalized_reference)
    carry_off_mask = list(normalized_reference)
    carry_on_mask[transition_index] = 1
    carry_off_mask[transition_index] = 0
    return carry_on_mask, carry_off_mask


def build_gate_training_row(
    *,
    doc: Mapping[str, object],
    transition_index: int,
    feature_vector: torch.Tensor | Sequence[float],
    carry_on_score: float,
    carry_off_score: float,
) -> Dict[str, object]:
    carry_on_score = float(carry_on_score)
    carry_off_score = float(carry_off_score)
    gate_target = 1 if carry_on_score > carry_off_score else 0
    return {
        "sample_id": str(doc.get("sample_id") or ""),
        "source_dataset": str(doc.get("source_dataset") or ""),
        "transition_index": int(transition_index),
        "features": feature_tensor_to_list(feature_vector),
        "gate_target": gate_target,
        "gate_scores": {
            "carry_on": carry_on_score,
            "carry_off": carry_off_score,
        },
        "oracle_block_sizes": [int(size) for size in list(doc.get("oracle_block_sizes") or [])],
        "has_final_answer_anchor": bool(doc.get("has_final_answer_anchor")),
    }


def build_rows_for_document(
    *,
    doc: Mapping[str, object],
    feature_matrix: torch.Tensor,
    carry_score_pairs: Mapping[int, Mapping[str, float]],
) -> List[Dict[str, object]]:
    if feature_matrix.dim() != 2:
        raise ValueError("feature_matrix must be 2D")
    rows: List[Dict[str, object]] = []
    for transition_index in range(feature_matrix.shape[0]):
        scores = dict((carry_score_pairs or {}).get(int(transition_index)) or {})
        rows.append(
            build_gate_training_row(
                doc=doc,
                transition_index=transition_index,
                feature_vector=feature_matrix[transition_index],
                carry_on_score=float(scores.get("carry_on", 0.0)),
                carry_off_score=float(scores.get("carry_off", 0.0)),
            )
        )
    return rows


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


def summarize_rows(rows: Sequence[Mapping[str, object]], *, split_name: str) -> Dict[str, object]:
    row_list = [dict(row) for row in rows]
    gate_counter = Counter(str(int(row["gate_target"])) for row in row_list)
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


def load_rows_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_number} must be an object")
            rows.append(row)
    return rows


def parse_gsm8k_numeric_value(text: str) -> float | None:
    answer = extract_gsm8k_answer(str(text))
    if answer is None:
        return None
    try:
        return float(answer)
    except (TypeError, ValueError):
        return None


def score_gsm8k_prediction(gold_solution: str, prediction: str, *, objective: str) -> float:
    normalized_objective = str(objective or LABEL_OBJECTIVE_CARRY_GATE_BINARY).strip().lower()
    if normalized_objective == LABEL_OBJECTIVE_CARRY_GATE_BINARY:
        return 1.0 if is_gsm8k_correct(str(gold_solution), str(prediction)) else 0.0
    if normalized_objective == LABEL_OBJECTIVE_NUMERIC_MARGIN:
        if is_gsm8k_correct(str(gold_solution), str(prediction)):
            return 1.0
        gold_value = parse_gsm8k_numeric_value(gold_solution)
        predicted_value = parse_gsm8k_numeric_value(prediction)
        if gold_value is None or predicted_value is None:
            return 0.0
        relative_error = abs(predicted_value - gold_value) / max(abs(gold_value), 1.0)
        closeness = 1.0 / (1.0 + relative_error)
        return 0.2 + 0.6 * closeness
    raise ValueError(f"Unsupported label objective: {objective}")


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


def build_gsm8k_boundary_gate_scorer_for_doc(
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
    try:
        from .eval_math_oracle import decode_generation
        from .eval_prompting import build_generation_prompt
        from .generate_oracle_blocks import generate_oracle_blocks_boundary_gate
    except ImportError:  # pragma: no cover
        from eval_math_oracle import decode_generation
        from eval_prompting import build_generation_prompt
        from generate_oracle_blocks import generate_oracle_blocks_boundary_gate

    oracle_block_sizes = [int(size) for size in list(doc.get("oracle_block_sizes") or [])]
    transition_count = max(0, len(oracle_block_sizes) - 1)
    prompt_text = str(doc["prompt_text"])
    user_input = build_generation_prompt(
        tokenizer,
        question=prompt_text,
        is_instruct=is_instruct,
        doc=doc,
    )
    input_ids = tokenizer(user_input)["input_ids"]
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    score_cache: Dict[Tuple[int, ...], float] = {}

    def score_mask(boundary_carry_mask: Sequence[int]) -> float:
        cache_key = tuple(int(value) for value in list(boundary_carry_mask))
        if cache_key in score_cache:
            return score_cache[cache_key]
        sample_gen_length = compute_oracle_generation_budget(oracle_block_sizes, default_gen_length=gen_length)
        generated_tokens, _, _ = generate_oracle_blocks_boundary_gate(
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
            oracle_block_sizes=oracle_block_sizes,
            boundary_carry_mask=list(cache_key),
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
        score = score_gsm8k_prediction(
            str(doc["gold_solution"]),
            prediction,
            objective=label_objective,
        )
        score_cache[cache_key] = float(score)
        return float(score)

    reference_mask = build_reference_boundary_carry_mask(transition_count=transition_count)
    reference_score = score_mask(reference_mask)

    def score_boundary_pair(transition_index: int) -> Dict[str, float]:
        carry_on_mask, carry_off_mask = build_boundary_carry_masks(
            reference_mask=reference_mask,
            transition_index=transition_index,
        )
        return {
            "carry_on": reference_score if carry_on_mask == reference_mask else score_mask(carry_on_mask),
            "carry_off": score_mask(carry_off_mask),
        }

    return score_boundary_pair


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
    label_objective: str,
) -> List[Dict[str, object]]:
    try:
        from .eval_math_oracle import build_local_corrector_transition_feature_matrix_for_doc
    except ImportError:  # pragma: no cover
        from eval_math_oracle import build_local_corrector_transition_feature_matrix_for_doc

    all_rows: List[Dict[str, object]] = []
    for doc in tqdm(documents, desc="gsm8k-boundary-gate-data"):
        oracle_block_sizes = [int(size) for size in list(doc.get("oracle_block_sizes") or [])]
        transition_count = max(0, len(oracle_block_sizes) - 1)
        if transition_count <= 0:
            continue
        feature_matrix = build_local_corrector_transition_feature_matrix_for_doc(
            doc=doc,
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        if feature_matrix.shape[0] != transition_count:
            raise ValueError(
                f"feature_matrix row count {feature_matrix.shape[0]} must equal transition_count {transition_count}"
            )
        score_boundary_pair = build_gsm8k_boundary_gate_scorer_for_doc(
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
        carry_score_pairs: Dict[int, Dict[str, float]] = {}
        for transition_index in range(transition_count):
            carry_score_pairs[transition_index] = score_boundary_pair(transition_index)
        all_rows.extend(
            build_rows_for_document(
                doc=doc,
                feature_matrix=feature_matrix,
                carry_score_pairs=carry_score_pairs,
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
    train_summary = summarize_rows(train_rows, split_name="train")
    valid_summary = summarize_rows(valid_rows, split_name="valid")
    audit_path = output_dir / f"dataset_audit_{timestamp}.md"
    lines = [
        f"# GSM8K Boundary Gate 数据审计 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- model_path: `{args.model_path}`",
        f"- output_dir: `{output_dir}`",
        f"- device: `{args.device}`",
        f"- gsm8k_limit: `{args.limit}`",
        f"- valid_doc_count: `{args.valid_doc_count}`",
        f"- label_objective: `{args.label_objective}`",
        "- 本数据构建链路独立于 Ada，不使用 Ada entropy、不使用 Ada boundary head，也不复用 Ada 的生成决策。",
        "- 当前标签定义采用条件式 toggle：参考 mask 默认全 `carry`，每次只比较当前 transition 的 keep-on 与 turn-off。",
        "",
        "## 训练集统计",
        "",
        f"- row_count: `{train_summary['row_count']}`",
        f"- document_count: `{train_summary['document_count']}`",
        f"- gate_target_distribution: `{json.dumps(train_summary['gate_target_distribution'], ensure_ascii=False)}`",
        "",
        "## 验证集统计",
        "",
        f"- row_count: `{valid_summary['row_count']}`",
        f"- document_count: `{valid_summary['document_count']}`",
        f"- gate_target_distribution: `{json.dumps(valid_summary['gate_target_distribution'], ensure_ascii=False)}`",
        "",
        "## 备注",
        "",
        "- 当前只构建 GSM8K gate-only 数据，不扩到 MATH。",
        "- 目标是先验证独立 `boundary_gate` 链路是否能把 GSM8K smoke32 从当前上界继续往 80.6 靠近。",
    ]
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audit_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GSM8K boundary carry gate training data.")
    parser.add_argument("--rows-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", type=str, default="gsm8k")
    parser.add_argument("--label-objective", type=str, default=LABEL_OBJECTIVE_CARRY_GATE_BINARY, choices=SUPPORTED_LABEL_OBJECTIVES)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--valid-doc-count", type=int, default=2)
    parser.add_argument("--gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--gen-length", type=int, default=DEFAULT_GEN_LENGTH)
    parser.add_argument("--block-length", type=int, default=DEFAULT_BLOCK_LENGTH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--mask-id", type=int, default=DEFAULT_MASK_ID)
    parser.add_argument("--remasking", type=str, default=DEFAULT_REMASKING)
    parser.add_argument("--gsm8k-landing-control", action="store_true")
    parser.add_argument("--gsm8k-landing-tail-lines", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows_jsonl is not None:
        rows = load_rows_jsonl(args.rows_jsonl)
        train_rows, valid_rows = split_rows_by_document(rows, valid_doc_count=args.valid_doc_count)
        paths = write_dataset_artifacts(
            output_dir=args.output_dir,
            train_rows=train_rows,
            valid_rows=valid_rows,
            metadata={
                "task": args.task,
                "label_objective": args.label_objective,
                "valid_doc_count": int(args.valid_doc_count),
                "rows_jsonl": str(args.rows_jsonl),
            },
        )
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    try:
        from .eval_math_oracle import load_model_and_tokenizer
    except ImportError:  # pragma: no cover
        from eval_math_oracle import load_model_and_tokenizer

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
            "task": args.task,
                "timestamp": timestamp,
                "limit": int(args.limit),
                "valid_doc_count": int(args.valid_doc_count),
                "label_objective": args.label_objective,
                "reference_mask_policy": "all_on_toggle",
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
        "run_manifest": str(manifest_path),
        **paths,
    }
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
