from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


TASK_HEADER = "[TASK]"
SIGNATURE_HEADER = "[SIGNATURE]"
PUBLIC_TESTS_HEADER = "[PUBLIC_TESTS]"
CODE_HEADER = "[CODE]"


@dataclass(frozen=True)
class SerializedTaskCondition:
    serialized_text: str
    condition_text: str
    code_prefix_text: str
    code_text: str
    condition_mask: int
    code_start_char: int
    section_offsets: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return {
            "serialized_text": self.serialized_text,
            "condition_text": self.condition_text,
            "code_prefix_text": self.code_prefix_text,
            "code_text": self.code_text,
            "condition_mask": self.condition_mask,
            "code_start_char": self.code_start_char,
            "section_offsets": dict(self.section_offsets),
        }


def _normalize_block(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = str(text).strip("\n")
    return normalized if normalized.strip() else None


def _render_section(header: str, body: Optional[str]) -> Optional[str]:
    normalized_body = _normalize_block(body)
    if normalized_body is None:
        return None
    return f"{header}\n{normalized_body}\n"


def serialize_task_condition(
    *,
    task_description: str,
    synthetic_signature: Optional[str],
    public_test_summary: Optional[str],
    code_text: str,
    include_public_tests: bool = True,
) -> Dict[str, object]:
    task_section = _render_section(TASK_HEADER, task_description) or f"{TASK_HEADER}\n\n"
    signature_section = _render_section(SIGNATURE_HEADER, synthetic_signature)
    public_tests_section = None
    if include_public_tests:
        public_tests_section = _render_section(PUBLIC_TESTS_HEADER, public_test_summary)
    code_prefix_text = f"{CODE_HEADER}\n"
    code_body = str(code_text or "").rstrip("\n")
    serialized_parts = [task_section]
    section_offsets: Dict[str, int] = {"task": 0}

    if signature_section is not None:
        section_offsets["signature"] = sum(len(part) for part in serialized_parts)
        serialized_parts.append(signature_section)
    if public_tests_section is not None:
        section_offsets["public_tests"] = sum(len(part) for part in serialized_parts)
        serialized_parts.append(public_tests_section)

    condition_text = "".join(serialized_parts)
    code_prefix_offset = len(condition_text)
    section_offsets["code"] = code_prefix_offset
    serialized_text = condition_text + code_prefix_text + code_body

    payload = SerializedTaskCondition(
        serialized_text=serialized_text,
        condition_text=condition_text,
        code_prefix_text=code_prefix_text,
        code_text=code_body,
        condition_mask=1 if public_tests_section is not None else 0,
        code_start_char=code_prefix_offset + len(code_prefix_text),
        section_offsets=section_offsets,
    )
    return payload.to_dict()
