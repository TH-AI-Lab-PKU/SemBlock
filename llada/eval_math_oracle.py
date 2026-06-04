from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from eval_prompting import build_generation_prompt, should_use_raw_completion_decode, truncate_generated_text
from generate_oracle_blocks import (
    generate_oracle_blocks,
    generate_oracle_blocks_boundary_gate,
    generate_oracle_blocks_dual_cache,
    generate_oracle_blocks_prefix_cache,
)
from gsm8k_landing import maybe_apply_gsm8k_landing
from math_oracle_benchmark import (
    build_gsm8k_cot_prompt,
    build_math_cot_prompt,
    compute_oracle_generation_budget,
    load_gsm8k_cot_fewshots,
    load_jsonl,
    select_math_fewshots,
    summarize_generation_records,
    summarize_oracle_documents,
)
from math_oracle_utils import build_math_oracle_document, is_gsm8k_correct, is_hendrycks_math_correct
from model.modeling_llada import LLaDAModelLM
from models.local_boundary_corrector import LocalBoundaryCorrector
from oracle_boundary_runtime_features import build_boundary_feature_vector, build_transition_feature_matrix


DEFAULT_MODEL_PATH = "/home/nvme03/workspace/models/GSAI-ML/LLaDA-8B-Instruct"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "experiments" / "semantic_boundary_indep"
DEFAULT_MATH_TRAIN_PATH = Path("/home/ubuntu/.cache/opencompass/data/math/train.jsonl")
DEFAULT_MATH_TEST_PATH = Path("/home/ubuntu/.cache/opencompass/data/math/test.jsonl")
BASELINE_SCORES = {
    "gsm8k": {"Vanilla": 76.7, "Dynamic": 77.6, "+Ada": 80.6},
    "math": {"Vanilla": 36.9, "Dynamic": 36.9, "+Ada": 37.3},
}
LOCAL_CORRECTION_DELTA_LIMIT = 2
LOCAL_CORRECTION_DELTA_CLASS_VALUES = (-2, -1, 0, 1, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_model_and_tokenizer(model_path: str, device: str):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    config = AutoConfig.from_pretrained(model_path)
    if hasattr(config, "flash_attention"):
        config.flash_attention = bool(device.startswith("cuda"))
    model = LLaDAModelLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        config=config,
    )
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    is_instruct = "instruct" in model_path.lower()
    return model, tokenizer, is_instruct



def extract_early_hidden_states(model, input_tensor: torch.Tensor) -> torch.Tensor:
    output = model(input_tensor, output_hidden_states=True, return_dict=True)
    return output.hidden_states[-1]


def tokenize_without_special_tokens(tokenizer, text: str) -> List[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer(text)
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if input_ids is None:
        raise ValueError("Tokenizer output must contain input_ids")
    if isinstance(input_ids, torch.Tensor):
        return [int(token) for token in input_ids.view(-1).tolist()]
    return [int(token) for token in list(input_ids)]


def build_local_corrector_feature_matrix_for_doc(
    *,
    doc: Mapping[str, object],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    oracle_block_sizes = _normalize_oracle_block_sizes(doc.get("oracle_block_sizes") or [])
    solution_text = str(doc.get("solution_text") or "")
    solution_token_ids = tokenize_without_special_tokens(tokenizer, solution_text)
    if not solution_token_ids:
        raise ValueError("local_corrector_path requires non-empty solution_text token ids")
    solution_token_ids = solution_token_ids[: max(1, sum(oracle_block_sizes))]
    input_tensor = torch.tensor(solution_token_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        hidden_states = extract_early_hidden_states(model, input_tensor)

    prior_boundary_points = list(doc.get("oracle_prior_boundary_points") or [])
    feature_vectors: List[torch.Tensor] = []
    for boundary_index in range(len(oracle_block_sizes)):
        prior_boundary_point = None
        if boundary_index < len(prior_boundary_points):
            candidate_point = prior_boundary_points[boundary_index]
            if isinstance(candidate_point, Mapping):
                prior_boundary_point = candidate_point
        feature_vector = build_boundary_feature_vector(
            hidden_states=hidden_states,
            oracle_block_sizes=oracle_block_sizes,
            boundary_index=boundary_index,
            prior_boundary_point=prior_boundary_point,
            has_final_answer_anchor=bool(doc.get("has_final_answer_anchor")),
        )
        feature_vectors.append(feature_vector.to(dtype=torch.float32))
    return torch.stack(feature_vectors, dim=0)


def build_local_corrector_transition_feature_matrix_for_doc(
    *,
    doc: Mapping[str, object],
    model,
    tokenizer,
    device: str,
) -> torch.Tensor:
    oracle_block_sizes = _normalize_oracle_block_sizes(doc.get("oracle_block_sizes") or [])
    solution_text = str(doc.get("solution_text") or "")
    solution_token_ids = tokenize_without_special_tokens(tokenizer, solution_text)
    if not solution_token_ids:
        raise ValueError("local_corrector_path requires non-empty solution_text token ids")
    solution_token_ids = solution_token_ids[: max(1, sum(oracle_block_sizes))]
    input_tensor = torch.tensor(solution_token_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        hidden_states = extract_early_hidden_states(model, input_tensor)

    return build_transition_feature_matrix(
        hidden_states=hidden_states,
        oracle_block_sizes=oracle_block_sizes,
        prior_boundary_points=list(doc.get("oracle_prior_boundary_points") or []),
        has_final_answer_anchor=bool(doc.get("has_final_answer_anchor")),
    )


def load_local_boundary_corrector_checkpoint(*, checkpoint_path: Path, device: str):
    checkpoint_path = Path(checkpoint_path)
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, Mapping) and "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state_dict = state["state_dict"]
    else:
        state_dict = state
    if not isinstance(state_dict, Mapping):
        raise ValueError("local_corrector checkpoint must be a state_dict mapping")

    backbone_weight = state_dict.get("backbone.0.weight")
    gate_weight = state_dict.get("gate_head.weight")
    delta_weight = state_dict.get("delta_head.weight")
    if backbone_weight is None or gate_weight is None or delta_weight is None:
        raise ValueError("local_corrector checkpoint is missing required weights")

    input_dim = int(backbone_weight.shape[1])
    hidden_dim = int(backbone_weight.shape[0])
    gate_classes = int(gate_weight.shape[0])
    delta_classes = int(delta_weight.shape[0])
    if gate_classes != 2:
        raise ValueError("local_corrector checkpoint must expose 2 gate classes")
    if delta_classes != len(LOCAL_CORRECTION_DELTA_CLASS_VALUES):
        raise ValueError(
            f"local_corrector checkpoint must expose {len(LOCAL_CORRECTION_DELTA_CLASS_VALUES)} delta classes"
        )

    local_corrector = LocalBoundaryCorrector(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        delta_classes=delta_classes,
    )
    local_corrector.load_state_dict(state_dict)
    local_corrector.eval().to(device)
    return local_corrector


def infer_predicted_deltas_from_feature_matrix(*, local_corrector, feature_matrix: torch.Tensor) -> Dict[int, int]:
    if feature_matrix.dim() != 2:
        raise ValueError("feature_matrix must be 2D (boundary_count, feature_dim)")
    if feature_matrix.shape[0] == 0:
        return {}

    local_device = next(local_corrector.parameters()).device
    with torch.no_grad():
        gate_logits, delta_logits = local_corrector(feature_matrix.to(device=local_device, dtype=torch.float32))
    gate_predictions = gate_logits.argmax(dim=-1).tolist()
    delta_predictions = delta_logits.argmax(dim=-1).tolist()

    predicted_deltas: Dict[int, int] = {}
    for boundary_index, (gate_class, delta_class) in enumerate(zip(gate_predictions, delta_predictions)):
        if int(gate_class) != 1:
            continue
        delta_value = int(LOCAL_CORRECTION_DELTA_CLASS_VALUES[int(delta_class)])
        if delta_value != 0:
            predicted_deltas[boundary_index] = delta_value
    return predicted_deltas


def infer_boundary_carry_mask_from_feature_matrix(*, local_corrector, feature_matrix: torch.Tensor) -> List[int]:
    if feature_matrix.dim() != 2:
        raise ValueError("feature_matrix must be 2D (transition_count, feature_dim)")
    if feature_matrix.shape[0] == 0:
        return []

    local_device = next(local_corrector.parameters()).device
    with torch.no_grad():
        gate_logits, _ = local_corrector(feature_matrix.to(device=local_device, dtype=torch.float32))
    gate_predictions = gate_logits.argmax(dim=-1).tolist()
    return [1 if int(prediction) == 1 else 0 for prediction in gate_predictions]


def decode_generation(
    *,
    tokenizer,
    generated_tokens: torch.Tensor,
    input_length: int,
    stop_tokens: Sequence[str],
    is_instruct: bool,
    doc: Mapping[str, object],
    gsm8k_landing_control: bool,
    gsm8k_landing_tail_lines: int,
) -> str:
    if should_use_raw_completion_decode(is_instruct=is_instruct, doc=doc):
        text = tokenizer.decode(generated_tokens[0][input_length:], skip_special_tokens=True)
        return truncate_generated_text(
            text,
            stop_tokens=stop_tokens,
            is_instruct=is_instruct,
            doc=doc,
        )

    text = tokenizer.decode(generated_tokens[0][input_length:], skip_special_tokens=False)
    text = truncate_generated_text(
        text,
        stop_tokens=stop_tokens,
        is_instruct=is_instruct,
        doc=doc,
    )
    generated_ids = torch.tensor(tokenizer(text)["input_ids"])
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    if str(doc.get("source_dataset", "")).lower() == "gsm8k":
        text = maybe_apply_gsm8k_landing(
            text,
            doc,
            enabled=gsm8k_landing_control,
            tail_line_budget=gsm8k_landing_tail_lines,
        )
    return text


def build_gsm8k_documents(tokenizer, num_fewshot: int, limit: int | None, gen_length: int) -> List[Dict[str, object]]:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    fewshots = load_gsm8k_cot_fewshots(num_fewshot)
    documents: List[Dict[str, object]] = []
    for index, row in enumerate(dataset):
        prompt_text = build_gsm8k_cot_prompt(str(row["question"]), fewshots)
        doc = build_math_oracle_document(
            sample_id=f"gsm8k/test/{index}",
            source_dataset="gsm8k",
            prompt_text=prompt_text,
            solution_text=str(row["answer"]),
            tokenizer=tokenizer,
            max_length=gen_length,
        )
        doc["question"] = str(row["question"])
        doc["gold_solution"] = str(row["answer"])
        doc["task_id"] = f"gsm8k/test/{index}"
        documents.append(doc)
    return documents


def sample_math_rows_stratified(
    rows: Sequence[Mapping[str, object]],
    *,
    limit: int | None,
    seed: int = 0,
) -> List[Dict[str, object]]:
    normalized_rows = [dict(row) for row in list(rows or [])]
    if limit is None:
        return normalized_rows

    limit = max(0, int(limit))
    if limit == 0:
        return []
    if limit >= len(normalized_rows):
        return normalized_rows

    del seed  # Reserved for future deterministic within-stratum shuffling if needed.

    grouped_rows: "OrderedDict[tuple[str, str], List[Dict[str, object]]]" = OrderedDict()
    for row in normalized_rows:
        stratum_key = (
            str(row.get("subject") or ""),
            str(row.get("level") or ""),
        )
        grouped_rows.setdefault(stratum_key, []).append(row)

    sampled_rows: List[Dict[str, object]] = []
    while len(sampled_rows) < limit:
        progressed = False
        for bucket in grouped_rows.values():
            if len(sampled_rows) >= limit:
                break
            if not bucket:
                continue
            sampled_rows.append(bucket.pop(0))
            progressed = True
        if not progressed:
            break
    return sampled_rows


def build_math_documents(
    tokenizer,
    train_path: Path,
    test_path: Path,
    num_fewshot: int,
    limit: int | None,
    gen_length: int,
    seed: int = 0,
) -> List[Dict[str, object]]:
    train_rows = load_jsonl(train_path)
    all_test_rows = load_jsonl(test_path)
    test_rows = sample_math_rows_stratified(all_test_rows, limit=limit, seed=seed)

    fewshots = select_math_fewshots(
        train_rows,
        num_fewshot,
        excluded_problem_texts=[str(row.get("problem") or "") for row in all_test_rows],
    )
    documents: List[Dict[str, object]] = []
    for index, row in enumerate(test_rows):
        prompt_text = build_math_cot_prompt(str(row["problem"]), fewshots)
        sample_id = str(row.get("unique_id") or f"math/test/{index}")
        doc = build_math_oracle_document(
            sample_id=sample_id,
            source_dataset="math",
            prompt_text=prompt_text,
            solution_text=str(row["solution"]),
            tokenizer=tokenizer,
            max_length=gen_length,
        )
        doc["problem"] = str(row["problem"])
        doc["gold_solution"] = str(row["solution"])
        doc["task_id"] = sample_id
        doc["subject"] = str(row.get("subject") or "")
        doc["level"] = str(row.get("level") or "")
        documents.append(doc)
    return documents


def resolve_oracle_generator(*, use_cache: bool, dual_cache: bool, boundary_gate: bool = False):
    if boundary_gate:
        return generate_oracle_blocks_boundary_gate
    if dual_cache:
        return generate_oracle_blocks_dual_cache
    if use_cache:
        return generate_oracle_blocks_prefix_cache
    return generate_oracle_blocks


def oracle_cache_mode_label(*, use_cache: bool, dual_cache: bool) -> str:
    if dual_cache:
        return "dual_cache"
    if use_cache:
        return "prefix_cache"
    return "no_cache"


def requested_cache_mode_label(*, use_cache: bool, dual_cache: bool, cache_min_block_count: int) -> str:
    base_mode = oracle_cache_mode_label(use_cache=use_cache, dual_cache=dual_cache)
    if not use_cache or cache_min_block_count <= 0:
        return base_mode
    return f"{base_mode}_min_blocks_{int(cache_min_block_count)}"


def should_use_cache_for_doc(*, use_cache: bool, oracle_block_sizes: Sequence[int], cache_min_block_count: int) -> bool:
    if not use_cache:
        return False
    if cache_min_block_count <= 0:
        return True
    return len(list(oracle_block_sizes or [])) >= int(cache_min_block_count)


def _normalize_oracle_block_sizes(raw_sizes: Sequence[object] | None) -> List[int]:
    normalized: List[int] = []
    for index, size in enumerate(list(raw_sizes or [])):
        if isinstance(size, bool):
            raise ValueError(f"oracle_block_sizes[{index}] must be a positive integer")
        try:
            numeric_value = float(size)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"oracle_block_sizes[{index}] must be a positive integer") from exc
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError(f"oracle_block_sizes[{index}] must be a finite integer")
        size_value = int(numeric_value)
        if size_value < 1:
            raise ValueError(f"oracle_block_sizes[{index}] must be >= 1")
        normalized.append(size_value)
    return normalized


def _normalize_local_boundary_delta(raw_delta: object, *, boundary_index: int) -> int:
    if isinstance(raw_delta, bool):
        raise ValueError(f"predicted_deltas[{boundary_index}] must be an integer delta")
    try:
        numeric_value = float(raw_delta)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"predicted_deltas[{boundary_index}] must be an integer delta") from exc
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise ValueError(f"predicted_deltas[{boundary_index}] must be a finite integer delta")
    delta_value = int(numeric_value)
    if abs(delta_value) > LOCAL_CORRECTION_DELTA_LIMIT:
        raise ValueError(
            f"predicted_deltas[{boundary_index}] must stay within +/-{LOCAL_CORRECTION_DELTA_LIMIT}"
        )
    return delta_value


def validate_cache_configuration(
    *,
    use_cache: bool,
    dual_cache: bool,
    cache_min_block_count: int,
    cache_policy: str | None,
) -> None:
    if cache_policy is None:
        return
    if use_cache or dual_cache or int(cache_min_block_count) != 0:
        raise ValueError("cache_policy cannot be combined with legacy cache flags")


def apply_local_boundary_correction(
    doc: Mapping[str, object],
    predicted_deltas: Mapping[int | str, int] | None,
) -> List[int]:
    oracle_block_sizes = _normalize_oracle_block_sizes(doc.get("oracle_block_sizes") or [])
    if predicted_deltas is None:
        return oracle_block_sizes
    if not isinstance(predicted_deltas, Mapping):
        raise ValueError("predicted_deltas must be a mapping of boundary_index to delta")
    if len(predicted_deltas) == 0:
        return oracle_block_sizes

    normalized_deltas: Dict[int, int] = {}
    for key, value in predicted_deltas.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        normalized_deltas[index] = _normalize_local_boundary_delta(value, boundary_index=index)

    corrected: List[int] = []
    for index, size in enumerate(oracle_block_sizes):
        delta_value = normalized_deltas.get(index, 0)
        corrected_size = size + delta_value
        corrected.append(max(1, corrected_size))
    return corrected


def evaluate_documents(
    *,
    task_name: str,
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
    use_cache: bool,
    dual_cache: bool,
    cache_min_block_count: int = 0,
    cache_policy: str | None = None,
    local_corrector_path: Path | None = None,
    gsm8k_landing_control: bool,
    gsm8k_landing_tail_lines: int,
    output_dir: Path,
) -> Dict[str, object]:
    validate_cache_configuration(
        use_cache=use_cache,
        dual_cache=dual_cache,
        cache_min_block_count=cache_min_block_count,
        cache_policy=cache_policy,
    )
    if local_corrector_path is not None and cache_policy not in ("prior_correction", "boundary_gate"):
        raise ValueError("local_corrector_path requires cache_policy in {'prior_correction', 'boundary_gate'}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / f"{task_name}_predictions.jsonl"

    stop_tokens = ["Q:"] if task_name == "gsm8k" else ["Problem:"]
    records: List[Dict[str, object]] = []
    apply_prior_correction = False
    apply_boundary_gate = False
    effective_use_cache = use_cache
    effective_dual_cache = dual_cache
    effective_cache_min_block_count = cache_min_block_count
    if cache_policy is not None:
        if cache_policy == "no_cache":
            effective_use_cache = False
            effective_dual_cache = False
            effective_cache_min_block_count = 0
        elif cache_policy == "prefix_cache":
            effective_use_cache = True
            effective_dual_cache = False
            effective_cache_min_block_count = 0
        elif cache_policy == "gate4":
            effective_use_cache = True
            effective_dual_cache = False
            effective_cache_min_block_count = 4
        elif cache_policy == "prior_correction":
            effective_use_cache = True
            effective_dual_cache = False
            effective_cache_min_block_count = 0
            apply_prior_correction = True
        elif cache_policy == "boundary_gate":
            effective_use_cache = True
            effective_dual_cache = False
            effective_cache_min_block_count = 0
            apply_boundary_gate = True
        else:
            raise ValueError(f"Unknown cache_policy={cache_policy}")

    cache_mode = requested_cache_mode_label(
        use_cache=effective_use_cache,
        dual_cache=effective_dual_cache,
        cache_min_block_count=effective_cache_min_block_count,
    )
    requested_cache_policy = cache_policy or cache_mode
    local_corrector = None
    prior_correction_requested_doc_count = 0
    prior_correction_applied_doc_count = 0
    boundary_gate_applied_doc_count = 0

    with open(prediction_path, "w", encoding="utf-8") as prediction_handle:
        for doc in tqdm(documents, desc=f"oracle-{task_name}"):
            prompt_text = str(doc["prompt_text"])
            user_input = build_generation_prompt(
                tokenizer,
                question=prompt_text,
                is_instruct=is_instruct,
                doc=doc,
            )
            input_ids = tokenizer(user_input)["input_ids"]
            input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
            oracle_block_sizes = _normalize_oracle_block_sizes(doc.get("oracle_block_sizes") or [])
            prior_correction_applied = False
            predicted_delta_count = 0
            boundary_carry_mask: List[int] = []
            boundary_gate_applied = False
            carry_on_boundary_count = 0
            doc_boundary_count = max(0, len(oracle_block_sizes) - 1)
            if apply_prior_correction:
                prior_correction_requested_doc_count += 1
                if local_corrector_path is not None:
                    if local_corrector is None:
                        local_corrector = load_local_boundary_corrector_checkpoint(
                            checkpoint_path=local_corrector_path,
                            device=device,
                        )
                    feature_matrix = build_local_corrector_feature_matrix_for_doc(
                        doc=doc,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    predicted_deltas = infer_predicted_deltas_from_feature_matrix(
                        local_corrector=local_corrector,
                        feature_matrix=feature_matrix,
                    )
                else:
                    if "predicted_deltas" not in doc:
                        raise ValueError(
                            "prior_correction requires predicted_deltas in each document when local_corrector_path is unset"
                        )
                    predicted_deltas = doc.get("predicted_deltas")
                predicted_delta_count = len(predicted_deltas or {})
                prior_correction_applied = predicted_delta_count > 0
                oracle_block_sizes = apply_local_boundary_correction(doc, predicted_deltas)
                if prior_correction_applied:
                    prior_correction_applied_doc_count += 1
            if apply_boundary_gate:
                if local_corrector_path is not None:
                    if local_corrector is None:
                        local_corrector = load_local_boundary_corrector_checkpoint(
                            checkpoint_path=local_corrector_path,
                            device=device,
                        )
                    transition_feature_matrix = build_local_corrector_transition_feature_matrix_for_doc(
                        doc=doc,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    boundary_carry_mask = infer_boundary_carry_mask_from_feature_matrix(
                        local_corrector=local_corrector,
                        feature_matrix=transition_feature_matrix,
                    )
                else:
                    if "boundary_carry_mask" not in doc:
                        raise ValueError(
                            "boundary_gate requires boundary_carry_mask in each document when local_corrector_path is unset"
                        )
                    boundary_carry_mask = [1 if bool(value) else 0 for value in list(doc.get("boundary_carry_mask") or [])]
                if len(boundary_carry_mask) != doc_boundary_count:
                    raise ValueError(
                        f"boundary_carry_mask must have length {doc_boundary_count} for sample_id={doc.get('sample_id')}"
                    )
                boundary_gate_applied = len(boundary_carry_mask) > 0
                carry_on_boundary_count = int(sum(boundary_carry_mask))
                if boundary_gate_applied:
                    boundary_gate_applied_doc_count += 1
            sample_gen_length = compute_oracle_generation_budget(oracle_block_sizes, default_gen_length=gen_length)
            if apply_boundary_gate:
                doc_use_cache = True
                doc_dual_cache = False
                doc_generate_fn = resolve_oracle_generator(
                    use_cache=doc_use_cache,
                    dual_cache=doc_dual_cache,
                    boundary_gate=True,
                )
                doc_cache_mode = "boundary_gate"
            else:
                doc_use_cache = should_use_cache_for_doc(
                    use_cache=effective_use_cache,
                    oracle_block_sizes=oracle_block_sizes,
                    cache_min_block_count=effective_cache_min_block_count,
                )
                doc_dual_cache = bool(doc_use_cache and effective_dual_cache)
                doc_generate_fn = resolve_oracle_generator(
                    use_cache=doc_use_cache,
                    dual_cache=doc_dual_cache,
                )
                doc_cache_mode = oracle_cache_mode_label(use_cache=doc_use_cache, dual_cache=doc_dual_cache)
            generated_tokens, nfe_history, block_history = doc_generate_fn(
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
                boundary_carry_mask=boundary_carry_mask,
            )
            prediction = decode_generation(
                tokenizer=tokenizer,
                generated_tokens=generated_tokens,
                input_length=input_tensor.shape[1],
                stop_tokens=stop_tokens,
                is_instruct=is_instruct,
                doc=doc,
                gsm8k_landing_control=gsm8k_landing_control,
                gsm8k_landing_tail_lines=gsm8k_landing_tail_lines,
            )
            if task_name == "gsm8k":
                is_correct = is_gsm8k_correct(str(doc["gold_solution"]), prediction)
            else:
                is_correct = is_hendrycks_math_correct(str(doc["gold_solution"]), prediction)

            record = {
                "sample_id": str(doc["sample_id"]),
                "task_name": task_name,
                "source_dataset": str(doc["source_dataset"]),
                "is_correct": bool(is_correct),
                "prediction": prediction,
                "gold_solution": str(doc["gold_solution"]),
                "oracle_block_sizes": list(oracle_block_sizes),
                "oracle_block_count": len(list(oracle_block_sizes)),
                "block_history": list(block_history),
                "nfe_history": list(nfe_history),
                "has_final_answer_anchor": bool(doc.get("has_final_answer_anchor")),
                "segment_count": len(list(doc.get("segments") or [])),
                "cache_mode": doc_cache_mode,
                "cache_policy": requested_cache_policy,
                "prior_correction_applied": prior_correction_applied,
                "predicted_delta_count": predicted_delta_count,
                "boundary_gate_applied": boundary_gate_applied,
                "boundary_carry_mask": list(boundary_carry_mask),
                "carry_on_boundary_count": carry_on_boundary_count,
                "carry_rate": float(carry_on_boundary_count / doc_boundary_count) if doc_boundary_count else 0.0,
            }
            if task_name == "gsm8k":
                record["question"] = str(doc.get("question") or "")
            else:
                record["problem"] = str(doc.get("problem") or "")
                record["subject"] = str(doc.get("subject") or "")
                record["level"] = str(doc.get("level") or "")

            prediction_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)

    oracle_summary = summarize_oracle_documents(documents)
    generation_summary = summarize_generation_records(records)
    total_boundary_count = sum(max(0, len(list(record.get("oracle_block_sizes") or [])) - 1) for record in records)
    total_carry_on_boundary_count = sum(int(record.get("carry_on_boundary_count", 0)) for record in records)
    summary: Dict[str, object] = {
        "task_name": task_name,
        **oracle_summary,
        **generation_summary,
        "prediction_path": str(prediction_path),
        "cache_mode": "boundary_gate" if apply_boundary_gate else cache_mode,
        "cache_policy": requested_cache_policy,
        "prior_correction_requested_doc_count": prior_correction_requested_doc_count,
        "prior_correction_applied_doc_count": prior_correction_applied_doc_count,
        "boundary_gate_applied_doc_count": boundary_gate_applied_doc_count,
        "carry_on_boundary_count": total_carry_on_boundary_count,
        "carry_rate": float(total_carry_on_boundary_count / total_boundary_count) if total_boundary_count else 0.0,
    }
    summary_path = output_dir / f"{task_name}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    success_examples = [record for record in records if record["is_correct"]][:10]
    failure_examples = [record for record in records if not record["is_correct"]][:10]
    return {
        "summary": summary,
        "records": records,
        "success_examples": success_examples,
        "failure_examples": failure_examples,
        "summary_path": str(summary_path),
        "prediction_path": str(prediction_path),
    }


def percent(value: float) -> float:
    return round(value * 100.0, 2)


def build_report_markdown(
    *,
    model_path: str,
    run_dir: Path,
    mode: str,
    args_dict: Dict[str, object],
    task_outputs: Mapping[str, Mapping[str, object]],
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# \u6570\u5b66\u8bed\u4e49\u8fb9\u754c Oracle \u8bc4\u6d4b\u62a5\u544a\uff08{timestamp}\uff09")
    lines.append("")
    lines.append(f"- \u6a21\u578b\uff1a`{model_path}`")
    lines.append(f"- \u8fd0\u884c\u76ee\u5f55\uff1a`{run_dir}`")
    lines.append(f"- \u8bc4\u6d4b\u6a21\u5f0f\uff1a`{mode}`")
    lines.append(
        f"- \u53c2\u6570\uff1a`B0={args_dict['block_length']}`\uff0c`steps={args_dict['steps']}`\uff0c`gen_length={args_dict['gen_length']}`\uff0c`threshold={args_dict['threshold']}`"
    )
    lines.append(
        "- \u8bf4\u660e\uff1a\u672c\u811a\u672c\u4ec5\u4f7f\u7528 `block_strategy=oracle` \u7684\u72ec\u7acb\u94fe\u8def\uff0c\u4e0d\u4f7f\u7528 Ada entropy\uff0c\u4e0d\u52a0\u8f7d Ada boundary head\u3002"
    )
    lines.append("")
    lines.append("## \u603b\u8868")
    lines.append("")
    lines.append("| \u4efb\u52a1 | Vanilla | Dynamic | +Ada | New Oracle | \u5907\u6ce8 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | --- |")
    for task_name, output in task_outputs.items():
        summary = output["summary"]
        oracle_score = percent(float(summary["exact_match"]))
        baseline = BASELINE_SCORES[task_name]
        remark = "smoke \u5b50\u96c6" if mode == "smoke" else "full"
        lines.append(
            f"| {task_name.upper()} | {baseline['Vanilla']:.1f} | {baseline['Dynamic']:.1f} | {baseline['+Ada']:.1f} | {oracle_score:.2f} | {remark} |"
        )
    lines.append("")

    for task_name, output in task_outputs.items():
        summary = output["summary"]
        lines.append(f"## {task_name.upper()}")
        lines.append("")
        lines.append(f"- exact_match / strict_match\uff1a`{percent(float(summary['exact_match'])):.2f}%`")
        lines.append(f"- \u5e73\u5747 NFE\uff1a`{float(summary['avg_nfe']):.2f}`")
        lines.append(f"- \u5e73\u5747\u751f\u6210\u5757\u6570\uff1a`{float(summary['avg_generated_block_count']):.2f}`")
        lines.append(f"- \u5e73\u5747\u751f\u6210\u5757\u957f\uff1a`{float(summary['avg_generated_block_length']):.2f}`")
        lines.append(f"- oracle \u5e73\u5747\u5757\u957f\uff1a`{float(summary['avg_block_length']):.2f}`")
        lines.append(f"- oracle \u5e73\u5747\u5757\u6570\uff1a`{float(summary['avg_block_count']):.2f}`")
        lines.append(f"- oracle \u8fb9\u754c\u8986\u76d6\u7387\uff1a`{percent(float(summary['boundary_coverage_rate'])):.2f}%`")
        lines.append(f"- \u6700\u7ec8\u7b54\u6848\u951a\u70b9\u547d\u4e2d\u7387\uff1a`{percent(float(summary['final_answer_anchor_hit_rate'])):.2f}%`")
        lines.append(f"- cache policy\uff1a`{summary['cache_policy']}`")
        lines.append(f"- \u6267\u884c cache mode\uff1a`{summary['cache_mode']}`")
        lines.append(
            f"- prior correction \u751f\u6548\u6587\u6863\u6570\uff1a`{int(summary['prior_correction_applied_doc_count'])}` / `{int(summary['prior_correction_requested_doc_count'])}`"
        )
        lines.append(f"- \u5757\u6570\u5206\u5e03\uff1a`{json.dumps(summary['block_count_distribution'], ensure_ascii=False)}`")
        lines.append("")
        lines.append("### \u6210\u529f\u6837\u4f8b\uff08\u6700\u591a 10 \u6761\uff09")
        lines.append("")
        if output["success_examples"]:
            for record in output["success_examples"]:
                prompt_key = "question" if task_name == "gsm8k" else "problem"
                lines.append(f"- \u6837\u4f8b `{record['sample_id']}`\uff1a{str(record.get(prompt_key, ''))[:180]}")
                lines.append(f"  - \u9884\u6d4b\uff1a`{record['prediction'][:180]}`")
                lines.append(
                    f"  - oracle \u5757\uff1a`{record['oracle_block_sizes']}`\uff1b\u751f\u6210\u5757\uff1a`{record['block_history']}`\uff1bNFE\uff1a`{record['nfe_history']}`"
                )
        else:
            lines.append("- \u65e0")
        lines.append("")
        lines.append("### \u5931\u8d25\u6837\u4f8b\uff08\u6700\u591a 10 \u6761\uff09")
        lines.append("")
        if output["failure_examples"]:
            for record in output["failure_examples"]:
                prompt_key = "question" if task_name == "gsm8k" else "problem"
                lines.append(f"- \u6837\u4f8b `{record['sample_id']}`\uff1a{str(record.get(prompt_key, ''))[:180]}")
                lines.append(f"  - \u9884\u6d4b\uff1a`{record['prediction'][:180]}`")
                lines.append(f"  - gold\uff1a`{record['gold_solution'][:180]}`")
                lines.append(
                    f"  - oracle \u5757\uff1a`{record['oracle_block_sizes']}`\uff1b\u751f\u6210\u5757\uff1a`{record['block_history']}`\uff1bNFE\uff1a`{record['nfe_history']}`"
                )
        else:
            lines.append("- \u65e0")
        lines.append("")

    if mode == "smoke":
        lines.append("## \u8bf4\u660e")
        lines.append("")
        lines.append("- \u5f53\u524d\u7ed3\u679c\u662f smoke \u5b50\u96c6\uff0c\u7528\u6765\u9a8c\u8bc1\u72ec\u7acb oracle \u94fe\u8def\u4e0e\u65b0\u5207\u5206\u5b9a\u4e49\u662f\u5426\u6210\u7acb\u3002")
        lines.append("- \u82e5\u8981\u548c\u56fe\u7247\u4e2d\u7684 Ada \u6307\u6807\u505a\u4e25\u683c\u4e3b\u7ed3\u8bba\u5bf9\u6bd4\uff0c\u4e0b\u4e00\u6b65\u5e94\u5728\u76f8\u540c\u914d\u7f6e\u4e0a\u8865\u8dd1 full\u3002")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="\u72ec\u7acb\u6570\u5b66 oracle \u8bc4\u6d4b\u811a\u672c\uff08GSM8K / MATH\uff09\u3002")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="math_oracle_v2")
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "math"], choices=["gsm8k", "math"])
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--gsm8k-limit", type=int, default=None)
    parser.add_argument("--math-limit", type=int, default=None)
    parser.add_argument("--smoke-limit", type=int, default=16)
    parser.add_argument("--gsm8k-fewshot", type=int, default=5)
    parser.add_argument("--math-fewshot", type=int, default=4)
    parser.add_argument("--math-train-path", type=Path, default=DEFAULT_MATH_TRAIN_PATH)
    parser.add_argument("--math-test-path", type=Path, default=DEFAULT_MATH_TEST_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--gen-length", type=int, default=512)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--dual-cache", action="store_true")
    parser.add_argument("--cache-min-block-count", type=int, default=0)
    parser.add_argument(
        "--cache-policy",
        choices=["no_cache", "prefix_cache", "gate4", "prior_correction", "boundary_gate"],
        default=None,
    )
    parser.add_argument("--local-corrector-path", type=Path, default=None)
    parser.add_argument("--gsm8k-landing-control", action="store_true")
    parser.add_argument("--gsm8k-landing-tail-lines", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dual_cache:
        args.use_cache = True
    set_seed(args.seed)

    timestamp = timestamp_slug()
    run_dir = args.output_root / f"{timestamp}_{args.run_name}_{args.mode}"
    eval_dir = run_dir / "eval"
    docs_dir = run_dir / "docs"
    eval_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = docs_dir / f"run_manifest_{timestamp}.json"
    manifest_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    model, tokenizer, is_instruct = load_model_and_tokenizer(args.model_path, args.device)

    task_outputs: Dict[str, Dict[str, object]] = {}
    for task_name in args.tasks:
        if task_name == "gsm8k":
            limit = args.gsm8k_limit if args.gsm8k_limit is not None else (args.smoke_limit if args.mode == "smoke" else None)
            documents = build_gsm8k_documents(
                tokenizer=tokenizer,
                num_fewshot=args.gsm8k_fewshot,
                limit=limit,
                gen_length=args.gen_length,
            )
        else:
            limit = args.math_limit if args.math_limit is not None else (args.smoke_limit if args.mode == "smoke" else None)
            documents = build_math_documents(
                tokenizer=tokenizer,
                train_path=args.math_train_path,
                test_path=args.math_test_path,
                num_fewshot=args.math_fewshot,
                limit=limit,
                gen_length=args.gen_length,
                seed=args.seed,
            )

        output = evaluate_documents(
            task_name=task_name,
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
            use_cache=args.use_cache,
            dual_cache=args.dual_cache,
            cache_min_block_count=args.cache_min_block_count,
            cache_policy=args.cache_policy,
            local_corrector_path=args.local_corrector_path,
            gsm8k_landing_control=args.gsm8k_landing_control,
            gsm8k_landing_tail_lines=args.gsm8k_landing_tail_lines,
            output_dir=eval_dir,
        )
        task_outputs[task_name] = output

    report = build_report_markdown(
        model_path=args.model_path,
        run_dir=run_dir,
        mode=args.mode,
        args_dict=vars(args),
        task_outputs=task_outputs,
    )
    report_path = docs_dir / f"oracle_report_{timestamp}.md"
    report_path.write_text(report, encoding="utf-8")

    final_summary = {
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "tasks": {task_name: output["summary"] for task_name, output in task_outputs.items()},
    }
    summary_path = eval_dir / f"oracle_run_summary_{timestamp}.json"
    summary_path.write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
