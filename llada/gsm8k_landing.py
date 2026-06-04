import re
from typing import Any


STRICT_ANSWER_RE = re.compile(r"####\s*(-?[0-9\.,]+)")
NUMERIC_CANDIDATE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
ANSWER_CUE_RE = re.compile(r"(?i)^(?:final answer|answer|therefore|thus|hence|so)\b")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_numeric_candidate(candidate: str) -> str | None:
    normalized = candidate.strip().lstrip("$").replace(",", "")
    normalized = re.sub(r"(?<=\d)[\.\!\?,:;]+$", "", normalized)
    if not normalized or normalized in {"-", ".", "-."}:
        return None
    return normalized


def extract_tail_numeric_candidate(text: str, tail_line_budget: int = 4) -> str | None:
    lines = [line.rstrip() for line in text.rstrip().splitlines() if line.strip()]
    if not lines:
        return None

    tail_lines = lines[-max(1, tail_line_budget) :]
    for line in reversed(tail_lines):
        stripped = line.strip()
        if not ANSWER_CUE_RE.match(stripped):
            continue
        matches = NUMERIC_CANDIDATE_RE.findall(stripped)
        if not matches:
            continue
        raw_candidate = next((group for group in matches[-1] if group), "")
        candidate = normalize_numeric_candidate(raw_candidate)
        if candidate is not None:
            return candidate

    tail = "\n".join(tail_lines).strip()
    sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", tail) if segment.strip()]
    for sentence in reversed(sentences):
        if not re.search(r"[.!?]$", sentence):
            continue
        matches = NUMERIC_CANDIDATE_RE.findall(sentence)
        if not matches:
            continue
        raw_candidate = next((group for group in matches[-1] if group), "")
        candidate = normalize_numeric_candidate(raw_candidate)
        if candidate is not None:
            return candidate

    last_line = lines[-1].strip()
    if re.fullmatch(r"\$?-?[\d,]+(?:\.\d+)?\.?", last_line):
        return normalize_numeric_candidate(last_line)
    return None


def is_gsm8k_request(doc: Any) -> bool:
    return (
        isinstance(doc, dict)
        and isinstance(doc.get("question"), str)
        and isinstance(doc.get("answer"), str)
        and "####" in doc["answer"]
    )


def maybe_apply_gsm8k_landing(
    generated_answer: str,
    doc: Any,
    *,
    enabled: bool,
    tail_line_budget: int = 4,
) -> str:
    if not enabled or not is_gsm8k_request(doc):
        return generated_answer

    stripped = generated_answer.rstrip()
    if not stripped or STRICT_ANSWER_RE.search(stripped):
        return generated_answer

    candidate = extract_tail_numeric_candidate(stripped, tail_line_budget=tail_line_budget)
    if candidate is None:
        return generated_answer

    return f"{stripped}\n#### {candidate}"
