from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from codecontests_functionization import functionize_python_program
from label_phase_boundary_python import BOUNDARY_TYPE_VOCAB, PHASE_LABEL_VOCAB, label_python_phase_boundary_spans
from task_condition_serialization import serialize_task_condition


DEFAULT_SMOKE_SAMPLING_STRATEGY = "equidistant"
DEFAULT_TOKENIZER_PATH = "GSAI-ML/LLaDA-8B-Instruct"


@dataclass(frozen=True)
class FunctionCompletionParts:
    signature: str
    body_text: str
    function_text: str
    body_start_char: int


def _tokenize(tokenizer, text: str) -> List[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _token_count(tokenizer, text: str) -> int:
    return len(_tokenize(tokenizer, text))


def resolve_tokenizer_path(tokenizer_path: Optional[str]) -> str:
    resolved = str(tokenizer_path or DEFAULT_TOKENIZER_PATH).strip()
    if not resolved:
        raise ValueError("tokenizer_path must not be empty.")
    return resolved


def summarize_public_tests(public_tests: Optional[Iterable[Dict[str, str]]], max_cases: int = 3) -> str:
    summaries: List[str] = []
    for index, test in enumerate(public_tests or []):
        if index >= max_cases:
            break
        test_input = str((test or {}).get("input", "")).strip().replace("\n", "\\n")
        test_output = str((test or {}).get("output", "")).strip().replace("\n", "\\n")
        summaries.append(f"case_{index + 1}: {test_input} -> {test_output}")
    return "\n".join(summaries)


def _char_to_token_index(tokenizer, text: str, char_index: int) -> int:
    return _token_count(tokenizer, text[: max(0, int(char_index))])


def _project_span_to_tokens(
    *,
    tokenizer,
    serialized_text: str,
    global_char_start: int,
    global_char_end: int,
    total_tokens: int,
) -> tuple[int, int]:
    token_start = _char_to_token_index(tokenizer, serialized_text, global_char_start)
    token_end = _char_to_token_index(tokenizer, serialized_text, global_char_end)
    token_start = max(0, min(token_start, total_tokens))
    token_end = max(token_start, min(token_end, total_tokens))
    if global_char_end > global_char_start and token_end == token_start and token_start < total_tokens:
        token_end = token_start + 1
    return token_start, token_end


def select_equidistant_indices(total_count: int, take_count: int) -> List[int]:
    total = int(total_count)
    take = int(take_count)
    if total < 0:
        raise ValueError("total_count must be non-negative")
    if take < 0:
        raise ValueError("take_count must be non-negative")
    if take == 0 or total == 0:
        return []
    if take >= total:
        return list(range(total))
    if take == 1:
        return [0]

    raw_indices = [round(index * (total - 1) / (take - 1)) for index in range(take)]
    selected: List[int] = []
    used = set()
    for candidate in raw_indices:
        resolved = None
        for offset in range(total):
            right = candidate + offset
            left = candidate - offset
            if 0 <= right < total and right not in used:
                resolved = right
                break
            if 0 <= left < total and left not in used:
                resolved = left
                break
        if resolved is None:
            raise RuntimeError("Unable to resolve a unique equidistant index.")
        used.add(resolved)
        selected.append(resolved)
    return sorted(selected)


def _join_segments_as_code(segments: Iterable[str]) -> str:
    normalized_segments = [str(segment) for segment in segments if str(segment).strip()]
    if not normalized_segments:
        return ""

    code_parts: List[str] = []
    for index, segment in enumerate(normalized_segments):
        if index == 0:
            code_parts.append(segment.rstrip("\n"))
            continue
        previous = code_parts[-1]
        prefix = "" if previous.endswith(("\n", ":", " ")) or segment.startswith(("\n", " ", "\t")) else "\n"
        code_parts.append(prefix + segment.rstrip("\n"))
    return "".join(code_parts).rstrip() + "\n"


def _build_public_test_token_span(
    *,
    tokenizer,
    serialized_text: str,
    section_offsets: Dict[str, int],
) -> List[int]:
    public_start = section_offsets.get("public_tests")
    code_start = section_offsets.get("code")
    if public_start is None or code_start is None:
        return [-1, -1]
    token_start = _char_to_token_index(tokenizer, serialized_text, int(public_start))
    token_end = _char_to_token_index(tokenizer, serialized_text, int(code_start))
    return [int(token_start), int(token_end)]


def _normalize_docstring(text: Optional[str]) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return '"""Implement the requested function."""'
    escaped = normalized.replace('"""', '\\"\\"\\"')
    return f'"""{escaped}"""'


def _build_line_offsets(text: str) -> List[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _char_index(line_offsets: List[int], lineno: int, col_offset: int) -> int:
    safe_line = max(1, min(int(lineno), len(line_offsets) - 1))
    return line_offsets[safe_line - 1] + max(0, int(col_offset))


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    args = args.strip()
    if args == "self":
        args = ""
    else:
        args = args.removeprefix("self,").strip()
        args = args.removeprefix("self ,").strip()
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args}){returns}:"


def _is_docstring_node(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = getattr(node, "value", None)
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def extract_function_completion_parts(
    code_text: str,
    *,
    synthetic_signature: Optional[str] = None,
) -> Optional[FunctionCompletionParts]:
    source = str(code_text or "").strip("\n")
    if not source.strip():
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None

    functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return None

    line_offsets = _build_line_offsets(source)

    def ranked_key(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
        start = _char_index(line_offsets, node.lineno, node.col_offset)
        end = _char_index(
            line_offsets,
            getattr(node, "end_lineno", node.lineno),
            getattr(node, "end_col_offset", 0),
        )
        return (end - start, int(node.lineno))

    target = max(functions, key=ranked_key)
    body_nodes = list(target.body)
    if body_nodes and _is_docstring_node(body_nodes[0]):
        body_nodes = body_nodes[1:]
    if not body_nodes:
        return None

    body_start = _char_index(
        line_offsets,
        getattr(body_nodes[0], "lineno", target.lineno),
        0,
    )
    function_end = _char_index(
        line_offsets,
        getattr(target, "end_lineno", target.lineno),
        getattr(target, "end_col_offset", len(source)),
    )
    body_text = source[body_start:function_end].rstrip()
    if not body_text.strip():
        return None

    inferred_signature = _function_signature(target)
    signature = str(synthetic_signature or "").strip()
    if not signature or "..." in signature:
        signature = inferred_signature
    function_text = source[
        _char_index(line_offsets, target.lineno, target.col_offset):function_end
    ].rstrip()
    return FunctionCompletionParts(
        signature=signature,
        body_text=body_text,
        function_text=function_text,
        body_start_char=body_start,
    )


def _infer_signature_from_sample(sample: Dict[str, object]) -> Optional[str]:
    explicit_signature = str(sample.get("signature") or "").strip()
    if explicit_signature:
        return explicit_signature
    func_name = str(sample.get("func_name") or "").strip()
    if not func_name:
        return None
    return f"def {func_name}(...):"


def _synthesize_mbpp_tests(*, task_description: str, synthetic_signature: Optional[str]) -> str:
    function_name = "candidate"
    if synthetic_signature and synthetic_signature.startswith("def "):
        function_name = synthetic_signature[4:].split("(", 1)[0].strip() or function_name
    task_line = str(task_description or "").strip() or "implement the function"
    return (
        f"assert callable({function_name})\n"
        f"# Task intent: {task_line}\n"
        f"# Return value must satisfy the described contract."
    )


def _infer_hard_mining_cluster(
    *,
    code_text: str,
    task_description: str,
    synthetic_signature: Optional[str],
) -> str:
    normalized_code = str(code_text or "")
    normalized_task = str(task_description or "").lower()
    normalized_signature = str(synthetic_signature or "").lower()
    if normalized_code.count("def ") > 1:
        return "helper_definition"
    if any(token in normalized_code for token in ("sorted(", ".sort(", "strip()", "lower()", "upper()")):
        return "normalization_sorting"
    if any(token in normalized_code for token in ("+=", "-=", "append(", "update(", "extend(", "best =", "ans =")):
        return "state_update"
    if "return " in normalized_code and any(token in normalized_code for token in ("join(", "format(", "tuple(", "list(", "str(")):
        return "return_shaping"
    if "," in normalized_signature and any(token in normalized_task for token in ("order", "position", "first", "second", "swap")):
        return "signature_order"
    return "general"


def _serialize_completion_aligned_prompt(
    *,
    prompt_style: str,
    task_description: str,
    synthetic_signature: Optional[str],
    synthetic_tests: Optional[str],
    code_text: str,
) -> Dict[str, object]:
    style = str(prompt_style).strip().lower()
    if style == "humaneval":
        signature_line = str(synthetic_signature or "def solve(...):").rstrip()
        docstring_block = _normalize_docstring(task_description)
        condition_text = f"{signature_line}\n    {docstring_block}\n"
        serialized_text = condition_text + str(code_text or "").rstrip("\n")
        return {
            "serialized_text": serialized_text,
            "condition_mask": 0,
            "code_start_char": len(condition_text),
            "section_offsets": {"signature": 0, "task": len(signature_line) + 1, "code": len(condition_text)},
        }
    if style == "mbpp":
        task_block = str(task_description or "").strip() or "Implement the requested function."
        tests_block = str(synthetic_tests or "").strip() or "# Synthetic tests unavailable"
        condition_text = f"[TASK]\n{task_block}\n[TESTS]\n{tests_block}\n[BEGIN]\n"
        serialized_text = condition_text + str(code_text or "").rstrip("\n")
        return {
            "serialized_text": serialized_text,
            "condition_mask": 0,
            "code_start_char": len(condition_text),
            "section_offsets": {"task": 0, "tests": len(f'[TASK]\n{task_block}\n'), "code": len(condition_text)},
        }
    raise ValueError(f"Unsupported prompt_style: {prompt_style}")


def build_serialized_training_record(
    *,
    tokenizer,
    serialized_text: str,
    code_text: str,
    code_start_char: int,
    section_offsets: Dict[str, int],
    source_domain: str,
    source_view: str,
    label_confidence: str,
    condition_mask: int,
    max_length: int = 4096,
    task_id: Optional[str] = None,
    public_test_metadata: Optional[Dict[str, object]] = None,
    hard_mining_cluster: Optional[str] = None,
    annotation_code_text: Optional[str] = None,
    annotation_target_start_char: int = 0,
) -> Dict[str, object]:
    input_ids = _tokenize(tokenizer, serialized_text)
    total_tokens = len(input_ids)

    phase_labels = [0] * total_tokens
    boundary_labels = [0] * total_tokens
    typed_transition_labels = [0] * total_tokens
    boundary_type_labels = [0] * total_tokens
    phase_loss_mask = [0] * total_tokens
    transition_loss_mask = [0] * total_tokens
    boundary_loss_mask = [0] * total_tokens

    condition_token_count = _token_count(tokenizer, serialized_text[:code_start_char])
    for index in range(condition_token_count, total_tokens):
        boundary_loss_mask[index] = 1

    target_text = str(code_text or "")
    annotation_text = str(annotation_code_text) if annotation_code_text is not None else target_text
    annotation_target_start = max(0, int(annotation_target_start_char or 0))
    annotation_target_end = annotation_target_start + len(target_text)
    annotations = label_python_phase_boundary_spans(annotation_text)
    for span in annotations["phase_spans"]:
        if span.get("label_confidence") == "ignore":
            continue
        span_start = int(span["char_start"])
        span_end = int(span["char_end"])
        overlap_start = max(span_start, annotation_target_start)
        overlap_end = min(span_end, annotation_target_end)
        if overlap_end <= overlap_start:
            continue
        token_start, token_end = _project_span_to_tokens(
            tokenizer=tokenizer,
            serialized_text=serialized_text,
            global_char_start=code_start_char + (overlap_start - annotation_target_start),
            global_char_end=code_start_char + (overlap_end - annotation_target_start),
            total_tokens=total_tokens,
        )
        if token_end <= token_start:
            continue
        phase_id = int(span["phase_id"])
        for token_index in range(token_start, token_end):
            if phase_loss_mask[token_index] and phase_labels[token_index] != phase_id:
                phase_loss_mask[token_index] = 0
                phase_labels[token_index] = 0
                continue
            phase_labels[token_index] = phase_id
            phase_loss_mask[token_index] = 1

    boundary_edges = list(annotations.get("boundary_edges") or [])
    if not boundary_edges:
        boundary_edges = [
            {
                "char_position": int(boundary_char),
                "boundary_type": "phase_change",
                "boundary_type_id": 1,
            }
            for boundary_char in annotations.get("boundary_positions", [])
        ]

    for boundary_edge in boundary_edges:
        annotation_boundary_position = int(boundary_edge["char_position"])
        if not (annotation_target_start <= annotation_boundary_position < annotation_target_end):
            continue
        token_position = _char_to_token_index(
            tokenizer,
            serialized_text,
            code_start_char + (annotation_boundary_position - annotation_target_start) + 1,
        ) - 1
        if 0 <= token_position < total_tokens:
            boundary_type_id = int(boundary_edge.get("boundary_type_id", 0))
            boundary_labels[token_position] = 1
            typed_transition_labels[token_position] = boundary_type_id
            boundary_loss_mask[token_position] = 1
            transition_loss_mask[token_position] = 1
            if boundary_type_labels[token_position] == 0:
                boundary_type_labels[token_position] = boundary_type_id

    if total_tokens > max_length:
        input_ids = input_ids[:max_length]
        phase_labels = phase_labels[:max_length]
        boundary_labels = boundary_labels[:max_length]
        typed_transition_labels = typed_transition_labels[:max_length]
        boundary_type_labels = boundary_type_labels[:max_length]
        phase_loss_mask = phase_loss_mask[:max_length]
        transition_loss_mask = transition_loss_mask[:max_length]
        boundary_loss_mask = boundary_loss_mask[:max_length]
        if boundary_labels:
            boundary_labels[-1] = 0
            typed_transition_labels[-1] = 0
            boundary_type_labels[-1] = 0
        total_tokens = max_length
        condition_token_count = min(condition_token_count, total_tokens)

    cluster_name = hard_mining_cluster or _infer_hard_mining_cluster(
        code_text=code_text,
        task_description="",
        synthetic_signature=None,
    )
    return {
        "task_id": task_id,
        "serialized_text": serialized_text,
        "input_ids": input_ids,
        "phase_labels": phase_labels,
        "typed_transition_labels": typed_transition_labels,
        "boundary_labels": boundary_labels,
        "boundary_type_labels": boundary_type_labels,
        "phase_loss_mask": phase_loss_mask,
        "transition_loss_mask": transition_loss_mask,
        "boundary_loss_mask": boundary_loss_mask,
        "source_domain": source_domain,
        "source_view": source_view,
        "condition_mask": int(condition_mask),
        "public_test_token_span": _build_public_test_token_span(
            tokenizer=tokenizer,
            serialized_text=serialized_text,
            section_offsets=section_offsets,
        ),
        "label_confidence": label_confidence,
        "phase_label_vocab": list(PHASE_LABEL_VOCAB),
        "boundary_type_vocab": list(BOUNDARY_TYPE_VOCAB),
        "condition_token_count": condition_token_count,
        "code_text": code_text,
        "hard_mining_cluster": cluster_name,
        **(public_test_metadata or {}),
    }


def build_domain_training_record(
    *,
    tokenizer,
    task_description: str,
    synthetic_signature: Optional[str],
    public_test_summary: Optional[str],
    code_text: str,
    source_domain: str,
    source_view: str,
    label_confidence: str,
    include_public_tests: bool,
    max_length: int = 4096,
    task_id: Optional[str] = None,
    public_test_metadata: Optional[Dict[str, object]] = None,
    hard_mining_cluster: Optional[str] = None,
) -> Dict[str, object]:
    serialized = serialize_task_condition(
        task_description=task_description,
        synthetic_signature=synthetic_signature,
        public_test_summary=public_test_summary,
        code_text=code_text,
        include_public_tests=include_public_tests,
    )
    record = build_serialized_training_record(
        tokenizer=tokenizer,
        serialized_text=str(serialized["serialized_text"]),
        code_text=code_text,
        code_start_char=int(serialized["code_start_char"]),
        section_offsets=dict(serialized.get("section_offsets") or {}),
        source_domain=source_domain,
        source_view=source_view,
        label_confidence=label_confidence,
        condition_mask=int(serialized["condition_mask"]),
        max_length=max_length,
        task_id=task_id,
        public_test_metadata=public_test_metadata,
        hard_mining_cluster=hard_mining_cluster or _infer_hard_mining_cluster(
            code_text=code_text,
            task_description=task_description,
            synthetic_signature=synthetic_signature,
        ),
    )
    record["task_description"] = task_description
    record["synthetic_signature"] = synthetic_signature
    record["public_test_summary"] = public_test_summary
    return record


def build_codecontests_training_rows(
    sample: Dict[str, object],
    *,
    tokenizer,
    max_length: int = 4096,
    disable_contest_functionized: bool = False,
) -> List[Dict[str, object]]:
    task_description = str(sample.get("description") or sample.get("prompt") or "")
    public_tests = sample.get("public_tests") or []
    public_test_summary = summarize_public_tests(public_tests)
    task_id = sample.get("task_id")
    solutions = [str(code) for code in (sample.get("solutions") or []) if str(code).strip()]
    if not solutions:
        return []

    public_test_metadata = {
        "functionization_passed_public": False,
        "functionization_passed_private": False,
        "public_test_summary": public_test_summary,
        **{
            key: sample[key]
            for key in ("original_split", "internal_split")
            if key in sample
        },
    }

    primary_solution = solutions[0]
    return [
        build_domain_training_record(
            tokenizer=tokenizer,
            task_description=task_description,
            synthetic_signature=None,
            public_test_summary=None,
            code_text=primary_solution,
            source_domain="codecontests",
            source_view="contest_description_only",
            label_confidence="silver",
            include_public_tests=False,
            max_length=max_length,
            task_id=task_id,
            public_test_metadata=public_test_metadata,
            hard_mining_cluster=_infer_hard_mining_cluster(
                code_text=primary_solution,
                task_description=task_description,
                synthetic_signature=None,
            ),
        )
    ]


def build_codesearchnet_humaneval_bridge_row(
    sample: Dict[str, object],
    *,
    tokenizer,
    max_length: int = 4096,
) -> Optional[Dict[str, object]]:
    language = str(sample.get("language", "")).lower()
    if language and language != "python":
        return None
    code_text = str(sample.get("code") or sample.get("code_text") or "")
    if not code_text.strip() and sample.get("segments"):
        code_text = _join_segments_as_code(sample.get("segments") or [])
    if not code_text.strip():
        return None
    task_description = str(sample.get("docstring") or sample.get("comment") or sample.get("task_prompt") or "")
    synthetic_signature = _infer_signature_from_sample(sample)
    completion_parts = extract_function_completion_parts(
        code_text,
        synthetic_signature=synthetic_signature,
    )
    completion_text = completion_parts.body_text if completion_parts is not None else code_text
    annotation_code_text = completion_parts.function_text if completion_parts is not None else None
    annotation_target_start_char = completion_parts.body_start_char if completion_parts is not None else 0
    if completion_parts is not None:
        synthetic_signature = completion_parts.signature
    serialized = _serialize_completion_aligned_prompt(
        prompt_style="humaneval",
        task_description=task_description,
        synthetic_signature=synthetic_signature,
        synthetic_tests=None,
        code_text=completion_text,
    )
    record = build_serialized_training_record(
        tokenizer=tokenizer,
        serialized_text=str(serialized["serialized_text"]),
        code_text=completion_text,
        code_start_char=int(serialized["code_start_char"]),
        section_offsets=dict(serialized["section_offsets"]),
        source_domain="codesearchnet",
        source_view="codesearchnet_humaneval_bridge",
        label_confidence="silver",
        condition_mask=int(serialized["condition_mask"]),
        max_length=max_length,
        task_id=sample.get("task_id") or sample.get("sha"),
        hard_mining_cluster=_infer_hard_mining_cluster(
            code_text=completion_text,
            task_description=task_description,
            synthetic_signature=synthetic_signature,
        ),
        annotation_code_text=annotation_code_text,
        annotation_target_start_char=annotation_target_start_char,
    )
    record["synthetic_signature"] = synthetic_signature
    if annotation_code_text is not None:
        record["full_function_text"] = annotation_code_text
    return record


def build_codesearchnet_mbpp_bridge_row(
    sample: Dict[str, object],
    *,
    tokenizer,
    max_length: int = 4096,
) -> Optional[Dict[str, object]]:
    language = str(sample.get("language", "")).lower()
    if language and language != "python":
        return None
    code_text = str(sample.get("code") or sample.get("code_text") or "")
    if not code_text.strip() and sample.get("segments"):
        code_text = _join_segments_as_code(sample.get("segments") or [])
    if not code_text.strip():
        return None
    task_description = str(sample.get("docstring") or sample.get("comment") or sample.get("task_prompt") or "")
    synthetic_signature = _infer_signature_from_sample(sample)
    synthetic_tests = _synthesize_mbpp_tests(
        task_description=task_description,
        synthetic_signature=synthetic_signature,
    )
    serialized = _serialize_completion_aligned_prompt(
        prompt_style="mbpp",
        task_description=task_description,
        synthetic_signature=synthetic_signature,
        synthetic_tests=synthetic_tests,
        code_text=code_text,
    )
    return build_serialized_training_record(
        tokenizer=tokenizer,
        serialized_text=str(serialized["serialized_text"]),
        code_text=code_text,
        code_start_char=int(serialized["code_start_char"]),
        section_offsets=dict(serialized["section_offsets"]),
        source_domain="codesearchnet",
        source_view="codesearchnet_mbpp_bridge",
        label_confidence="silver",
        condition_mask=int(serialized["condition_mask"]),
        max_length=max_length,
        task_id=sample.get("task_id") or sample.get("sha"),
        hard_mining_cluster=_infer_hard_mining_cluster(
            code_text=code_text,
            task_description=task_description,
            synthetic_signature=synthetic_signature,
        ),
    )


def build_codecontests_internal_split(
    *,
    train_rows: List[Dict[str, object]],
    valid_rows: List[Dict[str, object]],
    resplit_ratio: float,
    resplit_seed: int,
) -> Dict[str, object]:
    if not 0 < float(resplit_ratio) < 1:
        raise ValueError("resplit_ratio must be between 0 and 1.")

    filtered_train = _filter_rows_for_source_domain(list(train_rows), source_domain="codecontests")
    filtered_valid = _filter_rows_for_source_domain(list(valid_rows), source_domain="codecontests")
    combined_rows: List[Dict[str, object]] = []
    for original_index, row in enumerate(filtered_train):
        combined_rows.append(
            {
                **row,
                "original_split": "train",
                "_combined_order": original_index,
            }
        )
    for valid_index, row in enumerate(filtered_valid):
        combined_rows.append(
            {
                **row,
                "original_split": "valid",
                "_combined_order": len(filtered_train) + valid_index,
            }
        )

    def ordering_key(row: Dict[str, object]) -> tuple[int, str, int]:
        task_id = str(row.get("task_id", "")).strip()
        if task_id:
            return (0, task_id, int(row.get("_combined_order", 0)))
        return (1, "", int(row.get("_combined_order", 0)))

    ordered_rows = sorted(combined_rows, key=ordering_key)
    dev_count = int(round(len(ordered_rows) * float(resplit_ratio)))
    dev_indices = set(select_equidistant_indices(len(ordered_rows), dev_count))

    internal_train: List[Dict[str, object]] = []
    internal_dev: List[Dict[str, object]] = []
    for ordered_index, row in enumerate(ordered_rows):
        materialized = {key: value for key, value in row.items() if key != "_combined_order"}
        materialized["internal_split"] = "dev" if ordered_index in dev_indices else "train"
        if ordered_index in dev_indices:
            internal_dev.append(materialized)
        else:
            internal_train.append(materialized)

    shadow_valid = []
    for row in filtered_valid:
        materialized = dict(row)
        materialized["original_split"] = "valid"
        materialized["internal_split"] = "shadow_valid"
        shadow_valid.append(materialized)

    return {
        "split_mode": "combined_representative_80_20",
        "resplit_ratio": float(resplit_ratio),
        "resplit_seed": int(resplit_seed),
        "combined_count": len(ordered_rows),
        "train": internal_train,
        "dev": internal_dev,
        "shadow_valid": shadow_valid,
        "dev_indices": sorted(dev_indices),
        "ordered_task_ids": [row.get("task_id") for row in ordered_rows],
    }


def build_codesearchnet_bridge_row(
    sample: Dict[str, object],
    *,
    tokenizer,
    max_length: int = 4096,
) -> Optional[Dict[str, object]]:
    return build_codesearchnet_humaneval_bridge_row(
        sample,
        tokenizer=tokenizer,
        max_length=max_length,
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def _discover_jsonl_files(input_root: Path) -> List[Path]:
    return sorted(path for path in input_root.rglob("*.jsonl") if path.is_file())


def _load_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    return list(_iter_jsonl(path))


def _ordered_source_samples(
    rows: List[Dict[str, object]],
    *,
    source_domain: str,
) -> List[Dict[str, object]]:
    if source_domain == "codecontests":
        if rows and all(str(row.get("task_id", "")).strip() for row in rows):
            return sorted(rows, key=lambda row: str(row.get("task_id", "")))
    return list(rows)


def _filter_rows_for_source_domain(
    rows: List[Dict[str, object]],
    *,
    source_domain: str,
) -> List[Dict[str, object]]:
    filtered_rows: List[Dict[str, object]] = []
    for row in rows:
        language = str(row.get("language") or row.get("lang") or "").strip().lower()
        if source_domain == "codesearchnet":
            if language != "python":
                continue
        elif source_domain == "codecontests" and language and language != "python":
            continue
        filtered_rows.append(row)
    return filtered_rows


def _sample_source_examples(
    rows: List[Dict[str, object]],
    *,
    source_domain: str,
    take_count: int,
    sampling_strategy: str,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    filtered_rows = _filter_rows_for_source_domain(rows, source_domain=source_domain)
    ordered_rows = _ordered_source_samples(filtered_rows, source_domain=source_domain)
    if str(sampling_strategy).lower() != DEFAULT_SMOKE_SAMPLING_STRATEGY:
        raise ValueError(f"Unsupported smoke sampling strategy: {sampling_strategy}")
    selected_indices = select_equidistant_indices(len(ordered_rows), take_count)
    sampled_rows = [ordered_rows[index] for index in selected_indices]
    return sampled_rows, {
        "available_count": len(ordered_rows),
        "selected_count": len(sampled_rows),
        "selected_indices": selected_indices,
        "selected_task_ids": [
            row.get("task_id") or row.get("sha") or row.get("func_name")
            for row in sampled_rows
        ],
    }


def _build_records_from_sample(sample: Dict[str, object], tokenizer, max_length: int) -> List[Dict[str, object]]:
    sample_type = str(sample.get("source_domain") or sample.get("dataset") or "").lower()
    if "contest" in sample_type or "public_tests" in sample or "solutions" in sample:
        return build_codecontests_training_rows(sample, tokenizer=tokenizer, max_length=max_length)
    rows = [
        build_codesearchnet_humaneval_bridge_row(sample, tokenizer=tokenizer, max_length=max_length),
        build_codesearchnet_mbpp_bridge_row(sample, tokenizer=tokenizer, max_length=max_length),
    ]
    return [row for row in rows if row is not None]


def _build_leetcode_bridge_rows(
    *,
    leetcode_paths: Iterable[Path],
    tokenizer,
    max_length: int,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    from build_leetcode_humaneval_bridge_jsonl import (
        build_leetcode_humaneval_bridge_row,
        iter_json_records,
    )

    rows: List[Dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    rejected = Counter()
    for path in leetcode_paths:
        for sample in iter_json_records(path):
            row = build_leetcode_humaneval_bridge_row(sample, tokenizer=tokenizer, max_length=max_length)
            if row is None:
                rejected.update(["unusable_or_non_python"])
                continue
            key = (str(row.get("task_id") or ""), str(row.get("synthetic_signature") or ""))
            if key in seen:
                rejected.update(["duplicate_task_signature"])
                continue
            seen.add(key)
            rows.append(row)
    return rows, {
        "input_paths": [str(path) for path in leetcode_paths],
        "available_count": len(rows),
        "rejected_counts": dict(rejected),
    }


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    phase_counts = Counter()
    boundary_positive = 0
    boundary_total = 0
    view_counts = Counter()
    domain_counts = Counter()
    for row in rows:
        view_counts.update([str(row.get("source_view", "unknown"))])
        domain_counts.update([str(row.get("source_domain", "unknown"))])
        for label, mask in zip(row.get("phase_labels", []), row.get("phase_loss_mask", [])):
            if int(mask):
                phase_counts.update([PHASE_LABEL_VOCAB[int(label)]])
        boundary_positive += int(sum(int(x) for x in row.get("boundary_labels", [])))
        boundary_total += len(row.get("boundary_labels", []))
    return {
        "num_rows": len(rows),
        "source_view_counts": dict(view_counts),
        "source_domain_counts": dict(domain_counts),
        "phase_label_counts": dict(phase_counts),
        "boundary_positive_rate": 0.0 if boundary_total == 0 else boundary_positive / boundary_total,
        "functionization_pass_rate": 0.0 if not rows else sum(bool(row.get("functionization_passed_public")) for row in rows) / len(rows),
    }


def _sample_selected_rows(
    rows: List[Dict[str, object]],
    *,
    source_domain: str,
    take_count: int,
    sampling_strategy: str,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    filtered_rows = _filter_rows_for_source_domain(rows, source_domain=source_domain)
    ordered_rows = _ordered_source_samples(filtered_rows, source_domain=source_domain)
    requested_take = int(take_count)
    selected_indices = select_equidistant_indices(len(ordered_rows), requested_take or len(ordered_rows))
    selected_rows = [ordered_rows[index] for index in selected_indices]
    return selected_rows, {
        "available_count": len(ordered_rows),
        "selected_count": len(selected_rows),
        "selected_indices": selected_indices,
        "selected_task_ids": [
            row.get("task_id") or row.get("sha") or row.get("func_name")
            for row in selected_rows
        ],
        "sampling_strategy": str(sampling_strategy),
    }


def build_mixed_domain_dataset(
    *,
    codecontests_root: Path,
    codesearchnet_root: Path,
    output_root: Path,
    tokenizer,
    max_length: int,
    codecontests_train_take: int = 0,
    codecontests_valid_take: int = 0,
    codesearchnet_train_take: int = 0,
    codesearchnet_valid_take: int = 0,
    sampling_strategy: str = DEFAULT_SMOKE_SAMPLING_STRATEGY,
    codecontests_split_mode: str = "external_split",
    codecontests_resplit_ratio: float = 0.20,
    codecontests_resplit_seed: int = 20260419,
    disable_contest_functionized: bool = False,
    max_per_view: int = 0,
    allowed_views: set | None = None,
    leetcode_paths: Optional[List[Path]] = None,
    leetcode_train_take: int = 0,
    leetcode_valid_take: int = 0,
    leetcode_valid_ratio: float = 0.05,
) -> Dict[str, object]:
    if str(sampling_strategy).lower() != DEFAULT_SMOKE_SAMPLING_STRATEGY:
        raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")

    raw_contest_train = _load_jsonl_rows(codecontests_root / "train.jsonl")
    raw_contest_valid = _load_jsonl_rows(codecontests_root / "valid.jsonl")
    if str(codecontests_split_mode) == "combined_representative_80_20":
        split_payload = build_codecontests_internal_split(
            train_rows=raw_contest_train,
            valid_rows=raw_contest_valid,
            resplit_ratio=codecontests_resplit_ratio,
            resplit_seed=codecontests_resplit_seed,
        )
        selected_contest_train, contest_train_manifest = _sample_selected_rows(
            split_payload["train"],
            source_domain="codecontests",
            take_count=codecontests_train_take,
            sampling_strategy=sampling_strategy,
        )
        selected_contest_valid, contest_valid_manifest = _sample_selected_rows(
            split_payload["dev"],
            source_domain="codecontests",
            take_count=codecontests_valid_take,
            sampling_strategy=sampling_strategy,
        )
        selected_shadow_valid, shadow_valid_manifest = _sample_selected_rows(
            split_payload["shadow_valid"],
            source_domain="codecontests",
            take_count=0,
            sampling_strategy=sampling_strategy,
        )
    else:
        selected_contest_train, contest_train_manifest = _sample_selected_rows(
            raw_contest_train,
            source_domain="codecontests",
            take_count=codecontests_train_take,
            sampling_strategy=sampling_strategy,
        )
        for row in selected_contest_train:
            row.setdefault("original_split", "train")
            row.setdefault("internal_split", "train")
        selected_contest_valid, contest_valid_manifest = _sample_selected_rows(
            raw_contest_valid,
            source_domain="codecontests",
            take_count=codecontests_valid_take,
            sampling_strategy=sampling_strategy,
        )
        for row in selected_contest_valid:
            row.setdefault("original_split", "valid")
            row.setdefault("internal_split", "dev")
        selected_shadow_valid = list(selected_contest_valid)
        shadow_valid_manifest = dict(contest_valid_manifest)

    raw_bridge_train = _load_jsonl_rows(codesearchnet_root / "train.jsonl")
    raw_bridge_valid = _load_jsonl_rows(codesearchnet_root / "valid.jsonl")
    selected_bridge_train, bridge_train_manifest = _sample_selected_rows(
        raw_bridge_train,
        source_domain="codesearchnet",
        take_count=codesearchnet_train_take,
        sampling_strategy=sampling_strategy,
    )
    selected_bridge_valid, bridge_valid_manifest = _sample_selected_rows(
        raw_bridge_valid,
        source_domain="codesearchnet",
        take_count=codesearchnet_valid_take,
        sampling_strategy=sampling_strategy,
    )
    leetcode_train_manifest: Dict[str, object] = {"available_count": 0, "selected_count": 0}
    leetcode_valid_manifest: Dict[str, object] = {"available_count": 0, "selected_count": 0}
    selected_leetcode_train: List[Dict[str, object]] = []
    selected_leetcode_valid: List[Dict[str, object]] = []
    leetcode_manifest_extra: Dict[str, object] = {}
    if leetcode_paths:
        built_leetcode_rows, leetcode_manifest_extra = _build_leetcode_bridge_rows(
            leetcode_paths=leetcode_paths,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        valid_count = max(1, round(len(built_leetcode_rows) * float(leetcode_valid_ratio))) if built_leetcode_rows else 0
        raw_leetcode_train = built_leetcode_rows[:-valid_count] if valid_count else built_leetcode_rows
        raw_leetcode_valid = built_leetcode_rows[-valid_count:] if valid_count else []
        selected_leetcode_train, leetcode_train_manifest = _sample_selected_rows(
            raw_leetcode_train,
            source_domain="leetcode",
            take_count=leetcode_train_take,
            sampling_strategy=sampling_strategy,
        )
        selected_leetcode_valid, leetcode_valid_manifest = _sample_selected_rows(
            raw_leetcode_valid,
            source_domain="leetcode",
            take_count=leetcode_valid_take,
            sampling_strategy=sampling_strategy,
        )

    train_rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    shadow_valid_rows: List[Dict[str, object]] = []
    for sample in selected_contest_train:
        train_rows.extend(
            build_codecontests_training_rows(
                sample,
                tokenizer=tokenizer,
                max_length=max_length,
                disable_contest_functionized=disable_contest_functionized,
            )
        )
    for sample in selected_contest_valid:
        valid_rows.extend(
            build_codecontests_training_rows(
                sample,
                tokenizer=tokenizer,
                max_length=max_length,
                disable_contest_functionized=disable_contest_functionized,
            )
        )
    for sample in selected_shadow_valid:
        shadow_valid_rows.extend(
            build_codecontests_training_rows(
                sample,
                tokenizer=tokenizer,
                max_length=max_length,
                disable_contest_functionized=disable_contest_functionized,
            )
        )

    for sample in selected_bridge_train:
        bridge_rows = _build_records_from_sample(sample, tokenizer=tokenizer, max_length=max_length)
        train_rows.extend(bridge_rows)
    for sample in selected_bridge_valid:
        bridge_rows = _build_records_from_sample(sample, tokenizer=tokenizer, max_length=max_length)
        valid_rows.extend(bridge_rows)
    train_rows.extend(selected_leetcode_train)
    valid_rows.extend(selected_leetcode_valid)

    if allowed_views is not None:
        train_rows = [r for r in train_rows if str(r.get("source_view", "")) in allowed_views]
        valid_rows = [r for r in valid_rows if str(r.get("source_view", "")) in allowed_views]
        shadow_valid_rows = [r for r in shadow_valid_rows if str(r.get("source_view", "")) in allowed_views]
    if max_per_view and max_per_view > 0:
        view_counts: Dict[str, int] = {}
        filtered_train: List[Dict[str, object]] = []
        for r in train_rows:
            v = str(r.get("source_view", ""))
            if view_counts.get(v, 0) < max_per_view:
                filtered_train.append(r)
                view_counts[v] = view_counts.get(v, 0) + 1
        train_rows = filtered_train

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "train.jsonl", train_rows)
    _write_jsonl(output_root / "valid.jsonl", valid_rows)
    _write_jsonl(output_root / "shadow_valid.jsonl", shadow_valid_rows)

    summary = {
        "sampling_strategy": str(sampling_strategy),
        "codecontests_split_mode": str(codecontests_split_mode),
        "codecontests_resplit_ratio": float(codecontests_resplit_ratio),
        "codecontests_resplit_seed": int(codecontests_resplit_seed),
        "disable_contest_functionized": bool(disable_contest_functionized),
        "leetcode_valid_ratio": float(leetcode_valid_ratio),
        "train": _summarize_rows(train_rows),
        "valid": _summarize_rows(valid_rows),
        "shadow_valid": _summarize_rows(shadow_valid_rows),
    }
    manifest = {
        "sampling_strategy": str(sampling_strategy),
        "codecontests": {
            "train": contest_train_manifest,
            "valid": contest_valid_manifest,
            "shadow_valid": shadow_valid_manifest,
            "split_mode": str(codecontests_split_mode),
            "resplit_ratio": float(codecontests_resplit_ratio),
            "resplit_seed": int(codecontests_resplit_seed),
        },
        "codesearchnet": {
            "train": bridge_train_manifest,
            "valid": bridge_valid_manifest,
        },
        "leetcode": {
            **leetcode_manifest_extra,
            "train": leetcode_train_manifest,
            "valid": leetcode_valid_manifest,
        },
    }
    with open(output_root / "metadata_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_root / "sampling_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return summary


def build_smoke_dataset(
    *,
    codecontests_root: Path,
    codesearchnet_root: Path,
    output_root: Path,
    tokenizer,
    max_length: int,
    codecontests_train_take: int,
    codecontests_valid_take: int,
    codesearchnet_train_take: int,
    codesearchnet_valid_take: int,
    smoke_sampling_strategy: str = DEFAULT_SMOKE_SAMPLING_STRATEGY,
) -> Dict[str, object]:
    split_specs = [
        ("train", codecontests_train_take, codesearchnet_train_take),
        ("valid", codecontests_valid_take, codesearchnet_valid_take),
    ]

    manifest: Dict[str, object] = {
        "sampling_strategy": str(smoke_sampling_strategy),
        "codecontests": {},
        "codesearchnet": {},
    }
    built_rows_by_split: Dict[str, List[Dict[str, object]]] = {"train": [], "valid": []}

    for split_name, contest_take, bridge_take in split_specs:
        contest_rows = _load_jsonl_rows(codecontests_root / f"{split_name}.jsonl")
        selected_contests, contest_manifest = _sample_source_examples(
            contest_rows,
            source_domain="codecontests",
            take_count=contest_take,
            sampling_strategy=smoke_sampling_strategy,
        )
        manifest["codecontests"][split_name] = contest_manifest
        for sample in selected_contests:
            built_rows_by_split[split_name].extend(
                build_codecontests_training_rows(sample, tokenizer=tokenizer, max_length=max_length)
            )

        bridge_rows = _load_jsonl_rows(codesearchnet_root / f"{split_name}.jsonl")
        selected_bridge_rows, bridge_manifest = _sample_source_examples(
            bridge_rows,
            source_domain="codesearchnet",
            take_count=bridge_take,
            sampling_strategy=smoke_sampling_strategy,
        )
        manifest["codesearchnet"][split_name] = bridge_manifest
        for sample in selected_bridge_rows:
            built_rows_by_split[split_name].extend(
                _build_records_from_sample(sample, tokenizer=tokenizer, max_length=max_length)
            )

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "train.jsonl", built_rows_by_split["train"])
    _write_jsonl(output_root / "valid.jsonl", built_rows_by_split["valid"])

    summary = {
        "sampling_strategy": str(smoke_sampling_strategy),
        "train": _summarize_rows(built_rows_by_split["train"]),
        "valid": _summarize_rows(built_rows_by_split["valid"]),
    }
    with open(output_root / "metadata_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with open(output_root / "smoke_sampling_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build task-conditioned phase/boundary JSONL datasets.")
    parser.add_argument("--input_root", type=str, default=None)
    parser.add_argument("--codecontests_root", type=str, default=None)
    parser.add_argument("--codesearchnet_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--split_ratio", type=float, default=0.9)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--smoke_mode", action="store_true")
    parser.add_argument("--codecontests_train_take", type=int, default=0)
    parser.add_argument("--codecontests_valid_take", type=int, default=0)
    parser.add_argument("--codesearchnet_train_take", type=int, default=0)
    parser.add_argument("--codesearchnet_valid_take", type=int, default=0)
    parser.add_argument("--leetcode_paths", nargs="*", default=None,
                        help="Optional LeetCode JSON/JSONL files to convert into leetcode_humaneval_bridge rows.")
    parser.add_argument("--leetcode_train_take", type=int, default=0)
    parser.add_argument("--leetcode_valid_take", type=int, default=0)
    parser.add_argument("--leetcode_valid_ratio", type=float, default=0.05)
    parser.add_argument("--smoke_sampling_strategy", type=str, default=DEFAULT_SMOKE_SAMPLING_STRATEGY)
    parser.add_argument("--codecontests_split_mode", type=str, default="external_split")
    parser.add_argument("--codecontests_resplit_ratio", type=float, default=0.20)
    parser.add_argument("--codecontests_resplit_seed", type=int, default=20260419)
    parser.add_argument("--disable_contest_functionized", action="store_true")
    parser.add_argument("--max_per_view", type=int, default=0,
                        help="Cap samples per source_view (train split). 0 = unlimited.")
    parser.add_argument("--source_views", type=str, default=None,
                        help="Comma-separated list of allowed source_view values. "
                             "E.g. codesearchnet_humaneval_bridge,codesearchnet_mbpp_bridge,contest_description_only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build the dataset manifests.") from exc

    tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    if args.smoke_mode:
        if not args.codecontests_root or not args.codesearchnet_root:
            raise ValueError("--smoke_mode requires --codecontests_root and --codesearchnet_root.")
        build_smoke_dataset(
            codecontests_root=Path(args.codecontests_root),
            codesearchnet_root=Path(args.codesearchnet_root),
            output_root=output_root,
            tokenizer=tokenizer,
            max_length=args.max_length,
            codecontests_train_take=args.codecontests_train_take,
            codecontests_valid_take=args.codecontests_valid_take,
            codesearchnet_train_take=args.codesearchnet_train_take,
            codesearchnet_valid_take=args.codesearchnet_valid_take,
            smoke_sampling_strategy=args.smoke_sampling_strategy,
        )
        return

    if args.codecontests_root and args.codesearchnet_root:
        allowed_views = (
            {v.strip() for v in args.source_views.split(",") if v.strip()}
            if args.source_views
            else None
        )
        build_mixed_domain_dataset(
            codecontests_root=Path(args.codecontests_root),
            codesearchnet_root=Path(args.codesearchnet_root),
            output_root=output_root,
            tokenizer=tokenizer,
            max_length=args.max_length,
            codecontests_train_take=args.codecontests_train_take,
            codecontests_valid_take=args.codecontests_valid_take,
            codesearchnet_train_take=args.codesearchnet_train_take,
            codesearchnet_valid_take=args.codesearchnet_valid_take,
            sampling_strategy=args.smoke_sampling_strategy,
            codecontests_split_mode=args.codecontests_split_mode,
            codecontests_resplit_ratio=args.codecontests_resplit_ratio,
            codecontests_resplit_seed=args.codecontests_resplit_seed,
            disable_contest_functionized=args.disable_contest_functionized,
            max_per_view=args.max_per_view,
            allowed_views=allowed_views,
            leetcode_paths=[Path(path) for path in (args.leetcode_paths or [])],
            leetcode_train_take=args.leetcode_train_take,
            leetcode_valid_take=args.leetcode_valid_take,
            leetcode_valid_ratio=args.leetcode_valid_ratio,
        )
        return

    if not args.input_root:
        raise ValueError("--input_root is required when not running in --smoke_mode.")
    input_root = Path(args.input_root)

    rows: List[Dict[str, object]] = []
    for path in _discover_jsonl_files(input_root):
        for sample in _iter_jsonl(path):
            language = str(sample.get("language") or sample.get("lang") or args.language).lower()
            if language != str(args.language).lower():
                continue
            rows.extend(_build_records_from_sample(sample, tokenizer=tokenizer, max_length=args.max_length))

    split_index = int(len(rows) * float(args.split_ratio))
    train_rows = rows[:split_index]
    valid_rows = rows[split_index:]

    _write_jsonl(output_root / "train.jsonl", train_rows)
    _write_jsonl(output_root / "valid.jsonl", valid_rows)
    summary = {
        "train": _summarize_rows(train_rows),
        "valid": _summarize_rows(valid_rows),
    }
    with open(output_root / "metadata_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
