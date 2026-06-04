from __future__ import annotations

import argparse
import json
from pathlib import Path


def _iter_trace_files(path: Path):
    if path.is_file():
        yield path
        return
    yield from sorted(path.rglob("rank_*.jsonl"))


def _sample_key(record: dict) -> str | None:
    value = record.get("sample_id")
    if value is None:
        value = record.get("request_index")
    if value is None:
        return None
    return str(value)


def _read_selected(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter trace JSONL files to the first N samples or a selected sample-id list.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--sample-id-input", type=Path)
    parser.add_argument("--sample-id-output", type=Path)
    args = parser.parse_args()

    if args.max_samples is None and args.sample_id_input is None:
        raise SystemExit("Provide --max-samples or --sample-id-input")

    selected: list[str] = _read_selected(args.sample_id_input) if args.sample_id_input else []
    selected_set = set(selected)
    records_by_sample: dict[str, list[dict]] = {}

    for trace_file in _iter_trace_files(args.input):
        with trace_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = _sample_key(record)
                if key is None:
                    continue
                if args.sample_id_input is None and key not in records_by_sample:
                    if args.max_samples is not None and len(records_by_sample) >= args.max_samples:
                        continue
                    selected.append(key)
                    selected_set.add(key)
                if key in selected_set:
                    records_by_sample.setdefault(key, []).append(record)

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "rank_0.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for key in selected:
            for record in records_by_sample.get(key, []):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.sample_id_output is not None:
        args.sample_id_output.parent.mkdir(parents=True, exist_ok=True)
        args.sample_id_output.write_text("\n".join(selected) + "\n", encoding="utf-8")

    print(f"wrote {output_path}")
    print(f"samples={len(selected)}")
    print(f"events={sum(len(records_by_sample.get(key, [])) for key in selected)}")


if __name__ == "__main__":
    main()
