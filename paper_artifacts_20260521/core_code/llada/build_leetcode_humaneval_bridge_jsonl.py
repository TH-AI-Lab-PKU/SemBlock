from __future__ import annotations

import argparse
import ast
import html
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from build_task_conditioned_phase_boundary_jsonl import (
    _infer_hard_mining_cluster,
    _serialize_completion_aligned_prompt,
    build_serialized_training_record,
    resolve_tokenizer_path,
)


LEETCODE_VIEW = "leetcode_humaneval_bridge"
LEETCODE_DOMAIN = "leetcode"


@dataclass(frozen=True)
class ExtractedFunction:
    signature: str
    body_text: str
    function_text: str
    body_start_char: int


@dataclass(frozen=True)
class DatasetFilters:
    include_difficulties: frozenset[str]
    include_sources: frozenset[str]
    include_types: frozenset[str]
    include_tags_any: frozenset[str]
    exclude_tags: frozenset[str]
    exclude_custom_types: bool
    max_problem_chars: int
    max_tests: int
    max_solution_lines: int
    max_body_lines: int
    min_tokens: int
    max_tokens: int


def strip_html(text: object) -> str:
    raw = html.unescape(str(text or ""))
    raw = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", raw)
    raw = re.sub(r"(?is)</\s*(p|div|li|pre|code|h[1-6]|tr)\s*>", "\n", raw)
    raw = re.sub(r"(?is)<\s*(pre|code)[^>]*>", "\n", raw)
    raw = re.sub(r"(?is)<[^>]+>", " ", raw)
    raw = raw.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _first_nonempty(record: Dict[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _message_role(message: Dict[str, object]) -> str:
    return str(message.get("role") or message.get("from") or "").strip().lower()


def _message_content(message: Dict[str, object]) -> str:
    return str(message.get("content") or message.get("value") or "").strip()


def _iter_chat_messages(record: Dict[str, object]) -> Iterator[Dict[str, object]]:
    for key in ("messages", "conversations"):
        value = record.get(key)
        if isinstance(value, list):
            for message in value:
                if isinstance(message, dict):
                    yield message


def _last_chat_content(record: Dict[str, object], roles: Sequence[str]) -> str:
    role_set = set(roles)
    matched = ""
    for message in _iter_chat_messages(record):
        if _message_role(message) in role_set:
            content = _message_content(message)
            if content:
                matched = content
    return matched


def extract_markdown_code(text: object) -> str:
    value = str(text or "")
    fenced = re.findall(r"```(?:python|py)?\s*\n(.*?)```", value, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return max((block.strip() for block in fenced), key=len)
    return value.strip()


def _strip_starter_code_from_prompt(text: str) -> str:
    return re.split(r"(?i)\n\s*starter code\s*:", text, maxsplit=1)[0].strip()


def extract_task_description(record: Dict[str, object]) -> str:
    direct = _first_nonempty(
        record,
        (
            "problem",
            "description",
            "question_content",
            "question",
            "prompt",
            "problem_statement",
            "content",
            "title",
        ),
    )
    if not direct:
        direct = _last_chat_content(record, ("user", "human"))
    cleaned = strip_html(_strip_starter_code_from_prompt(direct))
    return cleaned


def _normalized_csv_set(raw_value: str | None) -> frozenset[str]:
    if not raw_value:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw_value.split(",") if part.strip())


def _sample_tags(record: Dict[str, object]) -> frozenset[str]:
    value = record.get("tags")
    if isinstance(value, list):
        return frozenset(str(tag).strip().lower() for tag in value if str(tag).strip())
    if isinstance(value, str):
        return _normalized_csv_set(value)
    return frozenset()


def _has_custom_problem_types(sample: Dict[str, object], code_text: str) -> bool:
    haystack = "\n".join(
        str(sample.get(key) or "")
        for key in ("starter_code", "problem", "solution", "prompt", "description")
    )
    haystack += "\n" + code_text
    return bool(
        re.search(
            r"\b(?:ListNode|TreeNode|Node|NestedInteger|Sea|MountainArray|ArrayReader|Reader4|ImmutableListNode)\b",
            haystack,
        )
    )


def _count_tests(sample: Dict[str, object]) -> int:
    tests = sample.get("tests")
    if isinstance(tests, list):
        return len(tests)
    return 0


def _line_count(text: str) -> int:
    return sum(1 for line in str(text or "").splitlines() if line.strip())


def _passes_pre_filters(sample: Dict[str, object], filters: DatasetFilters, rejected: Counter) -> bool:
    difficulty = str(sample.get("difficulty") or sample.get("level") or "").strip().lower()
    if filters.include_difficulties and difficulty not in filters.include_difficulties:
        rejected.update(["difficulty_filter"])
        return False

    source = str(sample.get("source") or "").strip().lower()
    if filters.include_sources and source not in filters.include_sources:
        rejected.update(["source_filter"])
        return False

    split_type = str(sample.get("type") or "").strip().lower()
    if filters.include_types and split_type not in filters.include_types:
        rejected.update(["type_filter"])
        return False

    tags = _sample_tags(sample)
    if filters.include_tags_any and not (tags & filters.include_tags_any):
        rejected.update(["include_tags_filter"])
        return False
    if filters.exclude_tags and (tags & filters.exclude_tags):
        rejected.update(["exclude_tags_filter"])
        return False

    if filters.max_problem_chars > 0 and len(extract_task_description(sample)) > filters.max_problem_chars:
        rejected.update(["problem_too_long"])
        return False

    if filters.max_tests > 0 and _count_tests(sample) > filters.max_tests:
        rejected.update(["too_many_tests"])
        return False
    return True


def _iter_nested_solutions(value: object) -> Iterator[Dict[str, object]]:
    if isinstance(value, dict):
        yield value
        for nested_key in ("python", "python3", "py", "solutions", "accepted"):
            nested = value.get(nested_key)
            if nested is not None and nested is not value:
                yield from _iter_nested_solutions(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_nested_solutions(item)
    elif isinstance(value, str):
        yield {"code": value}


def extract_python_solution(record: Dict[str, object]) -> str:
    assistant_content = _last_chat_content(record, ("assistant", "gpt"))
    if assistant_content:
        return extract_markdown_code(assistant_content)

    for key in (
        "python",
        "python3",
        "python_code",
        "solution_code",
        "accepted_solution",
        "code",
        "content",
        "solution",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    for candidate in _iter_nested_solutions(record.get("solutions")):
        language = str(candidate.get("language") or candidate.get("lang") or "").lower()
        if language and "python" not in language:
            continue
        code = _first_nonempty(candidate, ("code", "content", "solution", "python", "python3"))
        if code.strip():
            return code
    return ""


def _signature_from_function(node: ast.FunctionDef) -> str:
    args = ast.unparse(node.args)
    args = re.sub(r"^\s*self\s*,\s*", "", args)
    if args.strip() == "self":
        args = ""
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"def {node.name}({args}){returns}:"


def _dedent_body(source: str, node: ast.FunctionDef) -> str:
    if not node.body:
        return ""
    lines = source.splitlines()
    start = max(0, int(node.body[0].lineno) - 1)
    end = int(getattr(node, "end_lineno", len(lines)))
    body_lines = lines[start:end]
    nonempty = [line for line in body_lines if line.strip()]
    if not nonempty:
        return ""
    indent = min(len(line) - len(line.lstrip(" ")) for line in nonempty)
    return "\n".join(line[indent:] if len(line) >= indent else line for line in body_lines).rstrip()


def extract_function_completion(code_text: object) -> Optional[ExtractedFunction]:
    source = str(code_text or "").strip()
    if not source:
        return None
    if re.search(r"\b(?:input\s*\(|print\s*\(|sys\.stdin|stdin\.read|stdout\.write)", source):
        return None
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None

    target: Optional[ast.FunctionDef] = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and not member.name.startswith("__"):
                    target = member
                    break
        if target is not None:
            break
    if target is None:
        for node in module.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__"):
                target = node
                break
    if target is None:
        return None

    signature = _signature_from_function(target)
    body = _dedent_body(source, target)
    if not body.strip():
        return None
    body = textwrap.dedent(body).rstrip()
    function_text = signature + "\n" + textwrap.indent(body, "    ")
    try:
        ast.parse(function_text)
    except SyntaxError:
        return None
    return ExtractedFunction(
        signature=signature,
        body_text=textwrap.indent(body, "    "),
        function_text=function_text,
        body_start_char=len(signature + "\n"),
    )


def build_leetcode_humaneval_bridge_row(
    sample: Dict[str, object],
    *,
    tokenizer,
    max_length: int = 4096,
) -> Optional[Dict[str, object]]:
    language = str(sample.get("language") or sample.get("lang") or "").lower()
    if language and "python" not in language:
        return None

    extracted = extract_function_completion(extract_python_solution(sample))
    if extracted is None:
        return None

    task_description = extract_task_description(sample)
    if not task_description:
        task_description = "Implement the requested LeetCode function."

    serialized = _serialize_completion_aligned_prompt(
        prompt_style="humaneval",
        task_description=task_description,
        synthetic_signature=extracted.signature,
        synthetic_tests=None,
        code_text=extracted.body_text,
    )
    task_id = _first_nonempty(
        sample,
        ("task_id", "title_slug", "titleSlug", "slug", "frontend_question_id", "question_id", "id", "title"),
    )
    metadata = {
        "leetcode_title": _first_nonempty(sample, ("title", "question_title", "name")),
        "leetcode_slug": _first_nonempty(sample, ("title_slug", "titleSlug", "slug")),
        "leetcode_difficulty": _first_nonempty(sample, ("difficulty", "level")),
        "leetcode_source": _first_nonempty(sample, ("source",)),
        "leetcode_type": _first_nonempty(sample, ("type",)),
        "leetcode_tags": sorted(_sample_tags(sample)),
        "leetcode_num_tests": _count_tests(sample),
        "solution_line_count": _line_count(extracted.function_text),
        "body_line_count": _line_count(extracted.body_text),
        "synthetic_signature": extracted.signature,
        "task_description": task_description,
    }
    row = build_serialized_training_record(
        tokenizer=tokenizer,
        serialized_text=str(serialized["serialized_text"]),
        code_text=extracted.body_text,
        code_start_char=int(serialized["code_start_char"]),
        section_offsets=dict(serialized["section_offsets"]),
        source_domain=LEETCODE_DOMAIN,
        source_view=LEETCODE_VIEW,
        label_confidence="silver",
        condition_mask=int(serialized["condition_mask"]),
        max_length=max_length,
        task_id=task_id or None,
        public_test_metadata=metadata,
        hard_mining_cluster=_infer_hard_mining_cluster(
            code_text=extracted.function_text,
            task_description=task_description,
            synthetic_signature=extracted.signature,
        ),
        annotation_code_text=extracted.function_text,
        annotation_target_start_char=extracted.body_start_char,
    )
    row["synthetic_signature"] = extracted.signature
    row["completion_target_text"] = extracted.body_text
    row["full_function_text"] = extracted.function_text
    row["leetcode_difficulty"] = metadata["leetcode_difficulty"]
    row["leetcode_source"] = metadata["leetcode_source"]
    row["leetcode_type"] = metadata["leetcode_type"]
    row["leetcode_tags"] = metadata["leetcode_tags"]
    row["leetcode_num_tests"] = metadata["leetcode_num_tests"]
    row["solution_line_count"] = metadata["solution_line_count"]
    row["body_line_count"] = metadata["body_line_count"]
    return row


def iter_json_records(path: Path) -> Iterator[Dict[str, object]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to read parquet LeetCode shards.") from exc
        for row in pq.read_table(path).to_pylist():
            if isinstance(row, dict):
                yield row
        return

    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        yield payload
            return
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(payload, dict):
        for key in ("data", "rows", "items", "train", "valid", "test"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
        if not any(isinstance(payload.get(key), list) for key in ("data", "rows", "items", "train", "valid", "test")):
            yield payload


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    view_counts = Counter(str(row.get("source_view", "unknown")) for row in rows)
    domain_counts = Counter(str(row.get("source_domain", "unknown")) for row in rows)
    difficulty_counts = Counter(str(row.get("leetcode_difficulty", "unknown")) for row in rows)
    source_counts = Counter(str(row.get("leetcode_source", "unknown")) for row in rows)
    tag_counts = Counter(
        str(tag)
        for row in rows
        for tag in row.get("leetcode_tags", [])
        if str(tag).strip()
    )
    token_lengths = [len(row.get("input_ids", [])) for row in rows]
    body_line_counts = [int(row.get("body_line_count", 0)) for row in rows]
    boundary_total = sum(len(row.get("boundary_labels", [])) for row in rows)
    boundary_positive = sum(sum(int(x) for x in row.get("boundary_labels", [])) for row in rows)
    return {
        "num_rows": len(rows),
        "source_view_counts": dict(view_counts),
        "source_domain_counts": dict(domain_counts),
        "leetcode_difficulty_counts": dict(difficulty_counts),
        "leetcode_source_counts": dict(source_counts),
        "leetcode_top_tag_counts": dict(tag_counts.most_common(25)),
        "avg_tokens": 0.0 if not token_lengths else sum(token_lengths) / len(token_lengths),
        "min_tokens": 0 if not token_lengths else min(token_lengths),
        "max_tokens": 0 if not token_lengths else max(token_lengths),
        "avg_body_lines": 0.0 if not body_line_counts else sum(body_line_counts) / len(body_line_counts),
        "max_body_lines": 0 if not body_line_counts else max(body_line_counts),
        "boundary_positive_rate": 0.0 if boundary_total == 0 else boundary_positive / boundary_total,
    }


def build_dataset(
    *,
    input_paths: Sequence[Path],
    output_root: Path,
    tokenizer,
    max_length: int,
    valid_ratio: float,
    max_rows: int,
    filters: DatasetFilters,
) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    seen: set[Tuple[str, str]] = set()
    rejected = Counter()
    for path in input_paths:
        for sample in iter_json_records(path):
            if max_rows and len(rows) >= max_rows:
                break
            if not _passes_pre_filters(sample, filters, rejected):
                continue
            raw_solution = extract_python_solution(sample)
            if filters.exclude_custom_types and _has_custom_problem_types(sample, raw_solution):
                rejected.update(["custom_problem_type"])
                continue
            if filters.max_solution_lines > 0 and _line_count(raw_solution) > filters.max_solution_lines:
                rejected.update(["solution_too_long"])
                continue
            extracted = extract_function_completion(raw_solution)
            if extracted is None:
                rejected.update(["unusable_or_non_python"])
                continue
            if filters.max_body_lines > 0 and _line_count(extracted.body_text) > filters.max_body_lines:
                rejected.update(["body_too_long"])
                continue
            row = build_leetcode_humaneval_bridge_row(sample, tokenizer=tokenizer, max_length=max_length)
            if row is None:
                rejected.update(["unusable_or_non_python"])
                continue
            token_count = len(row.get("input_ids", []))
            if filters.min_tokens > 0 and token_count < filters.min_tokens:
                rejected.update(["too_few_tokens"])
                continue
            if filters.max_tokens > 0 and token_count > filters.max_tokens:
                rejected.update(["too_many_tokens"])
                continue
            key = (str(row.get("task_id") or ""), str(row.get("synthetic_signature") or ""))
            if key in seen:
                rejected.update(["duplicate_task_signature"])
                continue
            seen.add(key)
            rows.append(row)

    valid_count = max(1, round(len(rows) * valid_ratio)) if rows else 0
    train_rows = rows[:-valid_count] if valid_count else rows
    valid_rows = rows[-valid_count:] if valid_count else []
    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "train.jsonl", train_rows)
    _write_jsonl(output_root / "valid.jsonl", valid_rows)
    metadata = {
        "dataset_type": LEETCODE_VIEW,
        "input_paths": [str(path) for path in input_paths],
        "max_length": max_length,
        "valid_ratio": valid_ratio,
        "filters": {
            "include_difficulties": sorted(filters.include_difficulties),
            "include_sources": sorted(filters.include_sources),
            "include_types": sorted(filters.include_types),
            "include_tags_any": sorted(filters.include_tags_any),
            "exclude_tags": sorted(filters.exclude_tags),
            "exclude_custom_types": filters.exclude_custom_types,
            "max_problem_chars": filters.max_problem_chars,
            "max_tests": filters.max_tests,
            "max_solution_lines": filters.max_solution_lines,
            "max_body_lines": filters.max_body_lines,
            "min_tokens": filters.min_tokens,
            "max_tokens": filters.max_tokens,
        },
        "rejected_counts": dict(rejected),
        "train": _summarize_rows(train_rows),
        "valid": _summarize_rows(valid_rows),
    }
    with open(output_root / "metadata_summary.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LeetCode HumanEval-style phase/boundary dataset.")
    parser.add_argument("--input_paths", nargs="+", required=True, help="LeetCode JSON or JSONL files.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--max_rows", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--include_difficulties", default="")
    parser.add_argument("--include_sources", default="")
    parser.add_argument("--include_types", default="")
    parser.add_argument("--include_tags_any", default="")
    parser.add_argument("--exclude_tags", default="")
    parser.add_argument("--exclude_custom_types", action="store_true")
    parser.add_argument("--max_problem_chars", type=int, default=0)
    parser.add_argument("--max_tests", type=int, default=0)
    parser.add_argument("--max_solution_lines", type=int, default=0)
    parser.add_argument("--max_body_lines", type=int, default=0)
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required to build the dataset manifests.") from exc

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(args.tokenizer_path), trust_remote_code=True)
    metadata = build_dataset(
        input_paths=[Path(path) for path in args.input_paths],
        output_root=Path(args.output_root),
        tokenizer=tokenizer,
        max_length=args.max_length,
        valid_ratio=args.valid_ratio,
        max_rows=args.max_rows,
        filters=DatasetFilters(
            include_difficulties=_normalized_csv_set(args.include_difficulties),
            include_sources=_normalized_csv_set(args.include_sources),
            include_types=_normalized_csv_set(args.include_types),
            include_tags_any=_normalized_csv_set(args.include_tags_any),
            exclude_tags=_normalized_csv_set(args.exclude_tags),
            exclude_custom_types=bool(args.exclude_custom_types),
            max_problem_chars=int(args.max_problem_chars),
            max_tests=int(args.max_tests),
            max_solution_lines=int(args.max_solution_lines),
            max_body_lines=int(args.max_body_lines),
            min_tokens=int(args.min_tokens),
            max_tokens=int(args.max_tokens),
        ),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
