from __future__ import annotations

import ast
import re
from typing import Mapping, Optional


def is_code_completion_doc(doc: Optional[Mapping[str, object]]) -> bool:
    if not doc:
        return False

    task_id = str(doc.get("task_id", "")).lower()
    if task_id.startswith("humaneval"):
        return True

    if "entry_point" in doc and "canonical_solution" in doc:
        return True

    if "test_list" in doc and "text" in doc:
        return True

    return False


def is_mbpp_doc(doc: Optional[Mapping[str, object]]) -> bool:
    if not doc:
        return False
    task_id = str(doc.get("task_id", "")).lower()
    if task_id.startswith("humaneval"):
        return False
    return "test_list" in doc and "text" in doc


def should_use_chat_template(
    *,
    is_instruct: bool,
    doc: Optional[Mapping[str, object]],
) -> bool:
    if not is_instruct:
        return False
    return not is_code_completion_doc(doc)


def should_use_raw_completion_decode(
    *,
    is_instruct: bool,
    doc: Optional[Mapping[str, object]],
) -> bool:
    if not is_instruct or not doc:
        return False
    return str(doc.get("task_id", "")).lower().startswith("humaneval")


def build_generation_prompt(
    tokenizer,
    *,
    question: str,
    is_instruct: bool,
    doc: Optional[Mapping[str, object]],
    code_prompt_style: str = "raw",
) -> str:
    style = str(code_prompt_style or "raw").strip().lower()
    if is_code_completion_doc(doc) and style in {"body_comment", "humaneval_body_comment"}:
        return (
            question.rstrip()
            + "\n    # Complete the function body. Return only valid Python code for the body.\n"
        )
    if is_code_completion_doc(doc) and style in {"instruction_prefix", "humaneval_instruction_prefix"}:
        return (
            "Complete the following Python function. Return only the indented function body; "
            "do not include markdown, tests, or explanations.\n\n"
            + question
        )
    if should_use_chat_template(is_instruct=is_instruct, doc=doc):
        chat = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
        )
    return question


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _extract_function_body(text: str, entry_point: str) -> Optional[str]:
    try:
        module = ast.parse(text)
    except SyntaxError:
        return None
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            if not node.body:
                return None
            lines = text.splitlines()
            start = min(getattr(child, "lineno", 1) for child in node.body) - 1
            end = max(getattr(child, "end_lineno", getattr(child, "lineno", 1)) for child in node.body)
            return "\n".join(lines[start:end]).rstrip()
    return None


def normalize_code_completion_text(text: str, *, doc: Optional[Mapping[str, object]]) -> str:
    if not is_code_completion_doc(doc):
        return text

    normalized = _strip_code_fence(text).strip("\n")
    prompt = str((doc or {}).get("prompt", ""))
    if prompt and normalized.startswith(prompt):
        normalized = normalized[len(prompt):].lstrip("\n")

    entry_point = str((doc or {}).get("entry_point", ""))
    if entry_point and f"def {entry_point}" in normalized:
        extracted = _extract_function_body(normalized, entry_point)
        if extracted:
            normalized = extracted

    lines = normalized.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if not lines:
        return ""

    if is_mbpp_doc(doc):
        return "\n".join(lines).rstrip()

    first = lines[0]
    if first.strip() and not first.startswith((" ", "\t")) and not first.lstrip().startswith(("def ", "class ", "@")):
        lines = [("    " + line if line.strip() else line) for line in lines]
    return "\n".join(lines).rstrip()


def strip_code_prompt_echo(text: str, *, doc: Optional[Mapping[str, object]]) -> str:
    if not is_code_completion_doc(doc):
        return text

    prompt = str((doc or {}).get("prompt", "")).rstrip()
    if not prompt:
        return _strip_code_fence(text)

    if text.startswith(prompt):
        stripped = text[len(prompt):].lstrip("\n")
        return stripped or text

    prompt_lines = prompt.splitlines()
    text_lines = text.splitlines()
    shared = 0
    for prompt_line, text_line in zip(prompt_lines, text_lines):
        if prompt_line.rstrip() != text_line.rstrip():
            break
        shared += 1
    if shared > 0:
        stripped = "\n".join(text_lines[shared:]).lstrip("\n")
        return stripped or text

    return _strip_code_fence(text)


def ensure_code_body_indentation(text: str, *, doc: Optional[Mapping[str, object]]) -> str:
    if not is_code_completion_doc(doc):
        return text
    if is_mbpp_doc(doc):
        return ensure_mbpp_code_indentation(text, doc=doc)

    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return text

    positive_later_indents = []
    seen_first = False
    for line in lines:
        if not line.strip():
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if not seen_first:
            seen_first = True
            continue
        if current_indent > 0:
            positive_later_indents.append(current_indent)

    extra_shift = 0
    if positive_later_indents:
        extra_shift = max(min(positive_later_indents) - 4, 0)

    rebuilt = []
    first_written = False
    for line in lines:
        if not line.strip():
            rebuilt.append("")
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        if not first_written:
            rebuilt.append("    " + stripped)
            first_written = True
            continue
        normalized_indent = max(current_indent - extra_shift, 4)
        rebuilt.append((" " * normalized_indent) + stripped)
    return "\n".join(rebuilt).strip("\n")


def ensure_mbpp_code_indentation(text: str, *, doc: Optional[Mapping[str, object]] = None) -> str:
    lines = text.splitlines()
    if not any(line.strip() for line in lines):
        return text

    nonempty = [line for line in lines if line.strip()]
    min_indent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
    dedented = []
    for line in lines:
        if not line.strip():
            dedented.append("")
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        stripped = line.lstrip(" ")
        dedented.append((" " * max(current_indent - min_indent, 0)) + stripped)

    rebuilt = []
    in_def_block = False
    prev_indent = 0
    prev_stripped = ""
    for line in dedented:
        stripped = line.lstrip(" ")
        current_indent = len(line) - len(stripped)

        if not stripped:
            rebuilt.append("")
            continue
        if stripped.startswith(("import ", "from ", "@")) and not in_def_block:
            rebuilt.append(stripped)
            continue
        if stripped.startswith(("def ", "class ")):
            rebuilt.append(stripped)
            in_def_block = stripped.endswith(":")
            prev_indent = 0
            prev_stripped = stripped
            continue
        normalized_indent = current_indent
        if in_def_block and current_indent == 0:
            normalized_indent = 4
        elif (
            prev_stripped.endswith(":")
            and current_indent <= prev_indent
            and not stripped.startswith(("elif ", "else:", "except", "finally:"))
        ):
            normalized_indent = prev_indent + 4
        rebuilt.append((" " * normalized_indent) + stripped)
        prev_indent = normalized_indent
        prev_stripped = stripped

    return _normalize_mbpp_python3_snippets("\n".join(rebuilt).strip("\n"), doc=doc)


def _normalize_mbpp_python3_snippets(text: str, *, doc: Optional[Mapping[str, object]] = None) -> str:
    text = re.sub(r"len\(filter\((.*?)\)\)", r"len(list(filter(\1)))", text)
    text = re.sub(r"len\(map\((.*?)\)\)", r"len(list(map(\1)))", text)
    text = re.sub(r"int\(([^,()]+),\s*([268])\)", r"int(str(\1), \2)", text)
    text = re.sub(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*\*\*", r"{**\1, **", text)
    return _cast_digit_indexed_args(text, doc=doc)


def _cast_digit_indexed_args(text: str, *, doc: Optional[Mapping[str, object]]) -> str:
    prompt_text = str((doc or {}).get("text") or "").lower()
    if "digit" not in prompt_text:
        return text

    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        match = re.match(r"(\s*)def\s+\w+\(([^)]*)\)\s*:", line)
        if not match:
            continue
        body_indent = match.group(1) + "    "
        rest = "\n".join(lines[line_index + 1 :])
        inserts = []
        for raw_arg in match.group(2).split(","):
            arg_name = raw_arg.strip().split("=", 1)[0].strip()
            if not re.match(r"^[A-Za-z_]\w*$", arg_name):
                continue
            uses_sequence_ops = re.search(rf"\blen\({arg_name}\)|\b{arg_name}\s*\[", rest)
            already_assigned = re.search(rf"^\s*{arg_name}\s*=", rest, re.MULTILINE)
            if uses_sequence_ops and not already_assigned:
                inserts.append(f"{body_indent}{arg_name} = str({arg_name})")
        if inserts:
            return "\n".join(lines[: line_index + 1] + inserts + lines[line_index + 1 :])
    return text


def truncate_generated_text(
    text: str,
    *,
    stop_tokens,
    is_instruct: bool,
    doc: Optional[Mapping[str, object]],
    code_completion_postprocess: bool = False,
) -> str:
    cut_text = text

    extra_stop_tokens = []
    if should_use_raw_completion_decode(is_instruct=is_instruct, doc=doc):
        extra_stop_tokens = ["\n```", "\n###"]

    for stop_seq in list(stop_tokens) + extra_stop_tokens:
        if stop_seq and stop_seq in cut_text:
            cut_text = cut_text.split(stop_seq)[0]

    if code_completion_postprocess:
        cut_text = normalize_code_completion_text(
            ensure_code_body_indentation(
                strip_code_prompt_echo(cut_text, doc=doc),
                doc=doc,
            ),
            doc=doc,
        )

    return cut_text.rstrip()
