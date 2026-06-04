from __future__ import annotations

import re
from fractions import Fraction
from typing import Dict, List, Optional

try:
    from .prepare_boundary_corpora import (
        enrich_task_usable_record,
        looks_like_explicit_final_answer,
        looks_like_math_reasoning_step,
        split_math_text,
    )
except ImportError:  # pragma: no cover - direct test import fallback
    from prepare_boundary_corpora import (
        enrich_task_usable_record,
        looks_like_explicit_final_answer,
        looks_like_math_reasoning_step,
        split_math_text,
    )

try:
    from .oracle_boundary_candidates import build_candidate_boundary_points
except ImportError:  # pragma: no cover - direct test import fallback
    from oracle_boundary_candidates import build_candidate_boundary_points

try:
    from math_verify import parse as math_parse, verify as math_verify
except Exception:  # pragma: no cover - optional dependency fallback
    math_parse = None
    math_verify = None

try:
    from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed
except Exception:  # pragma: no cover - dependency fallback
    def last_boxed_only_string(string):
        idx = string.rfind("\\boxed")
        if "\\boxed " in string:
            return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
        if idx < 0:
            return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

    def remove_boxed(s):
        if s is None:
            return None
        if "\\boxed " in s:
            return s[len("\\boxed "):]
        if s.startswith("\\boxed{") and s.endswith("}"):
            return s[len("\\boxed{"):-1]
        return s

    def is_equiv(str1, str2, verbose=False):
        return str1 == str2


STRICT_ANSWER_RE = re.compile(r"####\s*(-?[0-9\.,]+)")
NUMERIC_CANDIDATE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
CONTROL_CHAR_LATEX_REPAIRS = {"\x08oxed": "\\boxed", "\x0crac": "\\frac"}


ACTION_LEAD_RE = re.compile(
    r"^(?:first|then|next|now|finally|thus|therefore|hence|so|we|to |find|calculate|compute|determine|multiply|divide|subtract|add|simplify|convert|count|list|write)\b",
    re.IGNORECASE,
)
LABEL_ONLY_RE = re.compile(r"^\s*(?:\d+|[A-Za-z])\s*[:.)]\s*$")
PRONOUN_HEAVY_ACTION_RE = re.compile(r"\b(?:that|those|this)\b", re.IGNORECASE)
DISCOURSE_PREFIX_RE = re.compile(r"^(?:first|then|next|now|finally|thus|therefore|hence|so)\b[,: ]*", re.IGNORECASE)
TARGET_MARKER_PATTERNS = (
    re.compile(r"\bto find out\b", re.IGNORECASE),
    re.compile(r"\bto figure out\b", re.IGNORECASE),
    re.compile(r"\bto determine\b", re.IGNORECASE),
    re.compile(r"\bto calculate\b", re.IGNORECASE),
    re.compile(r"\bto compute\b", re.IGNORECASE),
    re.compile(r"\bto find\b", re.IGNORECASE),
)
LEADING_ACTION_TARGET_PATTERNS = (
    re.compile(r"^find\s+(.+)$", re.IGNORECASE),
    re.compile(r"^calculate\s+(.+)$", re.IGNORECASE),
    re.compile(r"^compute\s+(.+)$", re.IGNORECASE),
    re.compile(r"^determine\s+(.+)$", re.IGNORECASE),
    re.compile(r"^convert\s+(.+)$", re.IGNORECASE),
    re.compile(r"^count\s+(.+)$", re.IGNORECASE),
)
NARRATIVE_STATE_PATTERNS = (
    (re.compile(r"^the (.+?) came out to (.+)$", re.IGNORECASE), lambda m: (m.group(1), m.group(2))),
    (re.compile(r"^(?:he|she|it) increased the (.+?) by (.+)$", re.IGNORECASE), lambda m: (f"increase in the {m.group(1)}", m.group(2))),
    (re.compile(r"^(?:so|thus|therefore) the (.+?) is (.+)$", re.IGNORECASE), lambda m: (m.group(1), m.group(2))),
    (re.compile(r"^(?:so|thus|therefore) (?:he|she|they|it) made a profit of (.+)$", re.IGNORECASE), lambda m: ("profit", m.group(1))),
)


def normalize_state_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    normalized = str(label).strip()
    normalized = re.sub(r"^(?:out\s+)?", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:how many|how much)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"^(?:the|a|an)\s+", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bare in\b", " in", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bis in\b", " in", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .:;,")
    return normalized or None


def extract_action_target_label(text: Optional[str], result: Optional[str] = None) -> Optional[str]:
    stripped = str(text or "").strip().rstrip(":")
    if not stripped:
        return None
    stripped = DISCOURSE_PREFIX_RE.sub("", stripped)

    lowered = stripped.lower()
    result_text = str(result or "")
    if "pension she gets per year" in lowered and "%" in result_text:
        return "fraction of the full pension she gets"
    if "how much she gets" in lowered:
        return "amount she gets"

    label: Optional[str] = None
    for pattern in TARGET_MARKER_PATTERNS:
        matches = list(pattern.finditer(stripped))
        if matches:
            label = stripped[matches[-1].end():].strip()
            break

    if label is None:
        for pattern in LEADING_ACTION_TARGET_PATTERNS:
            match = pattern.match(stripped)
            if match:
                label = match.group(1).strip()
                break

    if label is None and PRONOUN_HEAVY_ACTION_RE.search(stripped):
        leading_action = re.match(
            r"^(?:multiply|divide|subtract|add|simplify|convert|count|list|write)\s+(.+)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if leading_action:
            label = leading_action.group(1).strip()

    return normalize_state_label(label)


def rewrite_action_pair_with_explicit_target(lead: str, result: str) -> str:
    label = extract_action_target_label(lead, result=result)
    if label:
        return f"{label}: {result.strip()}"
    return f"{lead.strip()}\n{result.strip()}"


def rewrite_narrative_state_segment(segment: str) -> str:
    stripped = str(segment or "").strip()
    if not stripped or looks_like_explicit_final_answer(stripped):
        return stripped
    for pattern, formatter in NARRATIVE_STATE_PATTERNS:
        match = pattern.match(stripped)
        if match:
            label, result = formatter(match)
            normalized_label = normalize_state_label(label)
            if normalized_label:
                return f"{normalized_label}: {result.strip()}"
    return stripped


def looks_like_action_lead_segment(text: Optional[str]) -> bool:
    stripped = str(text or "").strip()
    if not stripped or looks_like_explicit_final_answer(stripped):
        return False
    if stripped.endswith(":"):
        return True
    return bool(ACTION_LEAD_RE.match(stripped)) and not looks_like_math_reasoning_step(stripped)


def looks_like_equation_or_result_segment(text: Optional[str]) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if looks_like_explicit_final_answer(stripped):
        return True
    if looks_like_math_reasoning_step(stripped):
        return True
    return bool(re.search(r"\d", stripped) and re.search(r"[=+\-*/^<>]", stripped))


def build_math_action_segments(text: str) -> List[str]:
    raw_segments = [segment.strip() for segment in split_math_text(text) if segment and segment.strip()]

    labeled_segments: List[str] = []
    pending_label: Optional[str] = None
    for segment in raw_segments:
        if LABEL_ONLY_RE.fullmatch(segment):
            pending_label = segment
            continue
        if pending_label:
            segment = f"{pending_label} {segment}"
            pending_label = None
        if segment.startswith("(") and labeled_segments:
            labeled_segments[-1] = f"{labeled_segments[-1]} {segment}"
            continue
        labeled_segments.append(segment)
    if pending_label:
        if labeled_segments:
            labeled_segments[-1] = f"{labeled_segments[-1]} {pending_label}"
        else:
            labeled_segments.append(pending_label)

    merged_segments: List[str] = []
    idx = 0
    while idx < len(labeled_segments):
        current = labeled_segments[idx]
        if idx + 1 < len(labeled_segments):
            nxt = labeled_segments[idx + 1]
            if (
                looks_like_action_lead_segment(current)
                and looks_like_equation_or_result_segment(nxt)
                and not looks_like_explicit_final_answer(nxt)
            ):
                current = rewrite_action_pair_with_explicit_target(current, nxt)
                idx += 1
        if current.startswith("(") and merged_segments:
            merged_segments[-1] = f"{merged_segments[-1]} {current}"
        else:
            merged_segments.append(current)
        idx += 1

    return [rewrite_narrative_state_segment(segment) for segment in merged_segments if segment.strip()]


def repair_common_latex_escapes(text: Optional[str]) -> str:
    if text is None:
        return ""
    repaired = text
    for bad, good in CONTROL_CHAR_LATEX_REPAIRS.items():
        repaired = repaired.replace(bad, good)
    return repaired


def extract_oracle_block_sizes_from_segments(segments: List[str], tokenizer, max_length: int = 512) -> List[int]:
    block_sizes: List[int] = []
    total_tokens = 0
    for segment in segments:
        if not segment.strip():
            continue
        token_ids = tokenizer(segment + "\n", add_special_tokens=False)["input_ids"]
        if not token_ids:
            continue
        remaining = max_length - total_tokens
        if remaining <= 0:
            break
        block_size = min(len(token_ids), remaining)
        block_sizes.append(block_size)
        total_tokens += block_size
    return block_sizes


def build_math_oracle_document(
    sample_id: str,
    source_dataset: str,
    prompt_text: str,
    solution_text: str,
    tokenizer,
    max_length: int = 512,
) -> Dict[str, object]:
    base_source = source_dataset if source_dataset in {"proofnet", "lean_workbook"} else "proofnet"
    normalized_solution_text = repair_common_latex_escapes(solution_text)
    record = enrich_task_usable_record(
        {
            "source": base_source,
            "source_dataset": source_dataset,
            "segments": build_math_action_segments(normalized_solution_text),
            "prompt_text": prompt_text,
        }
    )
    record["sample_id"] = sample_id
    record["source_dataset"] = source_dataset
    record["prompt_text"] = prompt_text
    record["solution_text"] = normalized_solution_text
    record["oracle_block_sizes"] = extract_oracle_block_sizes_from_segments(
        record.get("segments", []),
        tokenizer,
        max_length=max_length,
    )
    segment_token_lengths = list(record.get("oracle_block_sizes") or [])
    keep_count = len(segment_token_lengths)
    record["segments"] = list(record.get("segments", []))[:keep_count]
    record["segment_boundary_types"] = list(record.get("segment_boundary_types", []))[:keep_count]
    record["oracle_prior_boundary_points"] = [
        build_candidate_boundary_points(
            segment_token_lengths=segment_token_lengths,
            boundary_index=boundary_index,
            radius=2,
        )
        for boundary_index in range(len(segment_token_lengths))
    ]
    return record


def normalize_gsm8k_number(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text.strip().lstrip("$").replace(",", "")
    normalized = re.sub(r"(?<=\d)[\.\!\?,:;]+$", "", normalized)
    return normalized or None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    strict_match = STRICT_ANSWER_RE.search(text or "")
    if strict_match:
        return normalize_gsm8k_number(strict_match.group(1))
    matches = NUMERIC_CANDIDATE_RE.findall(text or "")
    if not matches:
        return None
    raw_candidate = next((group for group in matches[-1] if group), "")
    return normalize_gsm8k_number(raw_candidate)


def is_gsm8k_correct(gold_text: str, prediction_text: str) -> bool:
    gold = extract_gsm8k_answer(gold_text)
    pred = extract_gsm8k_answer(prediction_text)
    return gold is not None and pred is not None and gold == pred


def normalize_math_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"(?<!\\)[\.\!\?,:;]+$", "", normalized)
    return normalized or None


def _canonicalize_simple_math_answer(text: Optional[str]) -> Optional[str]:
    """Lightweight fallback for common boxed numeric/fraction answers."""
    if text is None:
        return None
    normalized = repair_common_latex_escapes(text)
    normalized = normalize_math_answer(normalized)
    if normalized is None:
        return None
    normalized = re.sub(
        r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}",
        r"\1/\2",
        normalized,
    )
    normalized = normalized.replace("\\dfrac", "").replace("\\tfrac", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.replace("\\,", "").replace(",", "")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized or None


def _answers_match_with_simple_fraction_fallback(pred: str, gold: str) -> bool:
    pred_norm = _canonicalize_simple_math_answer(pred)
    gold_norm = _canonicalize_simple_math_answer(gold)
    if pred_norm is None or gold_norm is None:
        return False
    if pred_norm == gold_norm:
        return True
    try:
        return Fraction(pred_norm) == Fraction(gold_norm)
    except Exception:
        return False


def extract_hendrycks_math_answer(text: str) -> Optional[str]:
    cleaned_text = repair_common_latex_escapes(text or "")
    boxed = last_boxed_only_string(cleaned_text)
    if boxed is not None:
        return normalize_math_answer(remove_boxed(boxed))
    dollars = [idx for idx, char in enumerate(cleaned_text) if char == "$"]
    if len(dollars) > 1:
        return normalize_math_answer(cleaned_text[dollars[0] + 1 : dollars[-1]])
    stripped = cleaned_text.strip()
    return normalize_math_answer(stripped or None)


def is_hendrycks_math_correct(gold_solution: str, prediction_text: str) -> bool:
    cleaned_gold = repair_common_latex_escapes(gold_solution)
    cleaned_pred = repair_common_latex_escapes(prediction_text)

    if math_parse is not None and math_verify is not None:
        try:
            gold_parsed = math_parse(cleaned_gold)
            pred_parsed = math_parse(cleaned_pred)
            if gold_parsed and pred_parsed:
                return bool(math_verify(gold_parsed, pred_parsed))
        except Exception:
            pass

    gold = extract_hendrycks_math_answer(cleaned_gold)
    pred = extract_hendrycks_math_answer(cleaned_pred)
    if gold is None or pred is None:
        return False
    if bool(is_equiv(pred, gold)):
        return True
    return _answers_match_with_simple_fraction_fallback(pred, gold)
