from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

TRIPLE_QUOTE_START_PATTERN = re.compile(r'(?i)(?:[rubf]{0,2})?("""|\'\'\')')


def has_standalone_triple_quoted_block(segment: str) -> bool:
    for raw_line in segment.splitlines():
        stripped = raw_line.lstrip()
        match = TRIPLE_QUOTE_START_PATTERN.match(stripped)
        if match is None:
            continue

        delimiter = match.group(1)
        suffix = stripped[match.end() :]
        if delimiter in suffix:
            closing_index = suffix.find(delimiter)
            trailing = suffix[closing_index + len(delimiter) :].strip()
            if not trailing:
                return True
            continue

        trailing = suffix.strip()
        if not trailing:
            return True
        if trailing.startswith((")", "]", "}", ",", "+", ".", ";")):
            continue
        return True

    return False


def classify_segment(segment: str) -> Optional[str]:
    stripped = segment.strip()
    if not stripped:
        return None

    if has_standalone_triple_quoted_block(segment):
        return "docstring_like"
    if stripped.startswith("/**"):
        return "docstring_like"
    if stripped.startswith("=begin"):
        return "docstring_like"
    return None


def summarize_jsonl(path: Path) -> Dict[str, object]:
    segment_hits: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_method: Counter[str] = Counter()
    summary: Dict[str, object] = {
        "path": str(path),
        "total_rows": 0,
        "rows_with_docstring_meta": 0,
        "rows_with_docstring_like_segments": 0,
        "docstring_like_segments": 0,
    }

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            summary["total_rows"] += 1
            if row.get("has_docstring"):
                summary["rows_with_docstring_meta"] += 1

            by_language[str(row.get("language") or "unknown")] += 1
            by_method[str(row.get("segmentation_method") or "unknown")] += 1

            row_has_docstring_like_segment = False
            for segment in row.get("segments") or []:
                label = classify_segment(segment)
                if label is None:
                    continue
                segment_hits[label] += 1
                if label == "docstring_like":
                    row_has_docstring_like_segment = True

            if row_has_docstring_like_segment:
                summary["rows_with_docstring_like_segments"] += 1

    summary["docstring_like_segments"] = segment_hits.get("docstring_like", 0)
    summary["segment_hits"] = dict(segment_hits)
    summary["by_language"] = dict(by_language)
    summary["by_method"] = dict(by_method)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit processed CodeSearchNet JSONL files for docstring-like leakage.")
    parser.add_argument("--input", type=Path, required=True, help="Processed JSONL file to audit.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path for a machine-readable JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_jsonl(args.input)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
