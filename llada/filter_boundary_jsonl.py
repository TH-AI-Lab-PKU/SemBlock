from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def filter_boundary_jsonl(
    *,
    input_path: str | Path,
    output_path: str | Path,
    language: str,
    metadata_path: str | Path | None = None,
) -> dict[str, object]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_language = str(language).strip().lower()
    if not normalized_language:
        raise ValueError("language must be a non-empty string")

    total_examples = 0
    written_examples = 0
    missing_language = 0

    with open(input_path, "r", encoding="utf-8") as source, open(
        output_path, "w", encoding="utf-8"
    ) as sink:
        for raw_line in source:
            line = raw_line.strip()
            if not line:
                continue
            total_examples += 1
            record = json.loads(line)
            record_language = record.get("language")
            if record_language is None:
                missing_language += 1
                continue
            if str(record_language).strip().lower() != normalized_language:
                continue
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_examples += 1

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "language": normalized_language,
        "total_examples": int(total_examples),
        "written_examples": int(written_examples),
        "skipped_examples": int(total_examples - written_examples),
        "missing_language_examples": int(missing_language),
    }

    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter semantic-boundary JSONL rows by language.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--metadata_path", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = filter_boundary_jsonl(
        input_path=args.input_path,
        output_path=args.output_path,
        language=args.language,
        metadata_path=args.metadata_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
