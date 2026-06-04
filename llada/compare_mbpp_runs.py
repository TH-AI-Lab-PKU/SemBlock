from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "regex": r"regex|regular expression",
    "string": r"string|substring|character|word|sentence|vowel|consonant",
    "list_tuple_set": r"list|array|tuple|set|similar|common elements",
    "sort": r"sort|ascending|descending",
    "math": r"prime|factorial|divisor|number|integer|sum|product|power|square|cube|multiple|decimal|binary",
    "counter_heap": r"frequency|common|counter|heap|largest items",
}


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current_delta", required=True)
    parser.add_argument("--other", action="append", required=True, help="name:path")
    args = parser.parse_args()

    current = load_jsonl(args.current_delta)
    current_ok = {int(row["doc_id"]): bool(row["new_pass"]) for row in current}
    current_rows = {int(row["doc_id"]): row for row in current}

    for spec in args.other:
        name, path = spec.split(":", 1)
        rows = load_jsonl(path)
        other_ok = {int(row["doc_id"]): bool(row.get("pass_at_1")) for row in rows}
        wins = [idx for idx, ok in other_ok.items() if ok and not current_ok.get(idx, False)]
        losses = [idx for idx, ok in other_ok.items() if current_ok.get(idx, False) and not ok]
        print(f"--- {name}")
        print(f"score={sum(other_ok.values()) / max(len(other_ok), 1):.3f}")
        print(f"other_pass_current_fail={len(wins)} current_pass_other_fail={len(losses)}")
        for pattern_name, pattern in PATTERNS.items():
            win_count = sum(1 for idx in wins if re.search(pattern, str(current_rows[idx]["text"]), re.I))
            loss_count = sum(1 for idx in losses if re.search(pattern, str(current_rows[idx]["text"]), re.I))
            print(f"{pattern_name}: wins={win_count} losses={loss_count}")
        print("sample wins:")
        for idx in wins[:20]:
            row = current_rows[idx]
            print(f"  {idx} task={row['task_id']} {str(row['text'])[:90]}")


if __name__ == "__main__":
    main()
