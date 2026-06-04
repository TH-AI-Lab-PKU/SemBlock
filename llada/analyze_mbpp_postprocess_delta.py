from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from eval_prompting import truncate_generated_text


def _run_tests(code: str, tests: list[str], queue: mp.Queue) -> None:
    namespace: dict[str, object] = {}
    try:
        exec(code, namespace)
        for test in tests:
            exec(test, namespace)
    except BaseException as exc:
        queue.put({"passed": False, "error": f"{type(exc).__name__}: {exc}"})
        return
    queue.put({"passed": True, "error": ""})


def run_with_timeout(code: str, tests: list[str], timeout_seconds: float) -> dict[str, object]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_run_tests, args=(code, tests, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(1.0)
        return {"passed": False, "error": "Timeout"}
    if queue.empty():
        return {"passed": False, "error": f"No result, exitcode={process.exitcode}"}
    return dict(queue.get())


def maybe_add_missing_stdlib_imports(code: str) -> str:
    additions = []
    if "math." in code and "import math" not in code and "from math import" not in code:
        additions.append("import math")
    if "re." in code and "import re" not in code and "from re import" not in code:
        additions.append("import re")
    if "Counter(" in code and "Counter" not in "\n".join(
        line for line in code.splitlines() if line.lstrip().startswith(("import ", "from "))
    ):
        additions.append("from collections import Counter")
    if "heapq." in code and "import heapq" not in code and "import heapq as" not in code:
        additions.append("import heapq")
    if "hq." in code and "import heapq as hq" not in code:
        additions.append("import heapq as hq")
    if not additions:
        return code
    return "\n".join(additions + [code])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout_seconds", type=float, default=2.0)
    parser.add_argument("--auto_imports", action="store_true")
    args = parser.parse_args()

    sample_path = Path(args.samples)
    rows = [json.loads(line) for line in sample_path.read_text().splitlines() if line.strip()]
    output_rows = []
    old_pass = 0
    new_pass = 0
    rescued = []
    regressed = []

    for row in rows:
        old_ok = bool(row.get("pass_at_1"))
        old_pass += int(old_ok)
        original = (row.get("filtered_resps") or [""])[0]
        doc = row.get("doc") or {}
        repaired = truncate_generated_text(
            original,
            stop_tokens=["[DONE]"],
            is_instruct=True,
            doc=doc,
            code_completion_postprocess=True,
        )
        if args.auto_imports:
            repaired = maybe_add_missing_stdlib_imports(repaired)
        result = run_with_timeout(
            repaired,
            [str(test) for test in doc.get("test_list", [])],
            timeout_seconds=float(args.timeout_seconds),
        )
        new_ok = bool(result["passed"])
        new_pass += int(new_ok)
        record = {
            "doc_id": row.get("doc_id"),
            "task_id": doc.get("task_id"),
            "text": doc.get("text"),
            "old_pass": old_ok,
            "new_pass": new_ok,
            "error": result.get("error", ""),
            "old_first_line": original.splitlines()[0] if original.splitlines() else "",
            "new_first_line": repaired.splitlines()[0] if repaired.splitlines() else "",
        }
        output_rows.append(record)
        if new_ok and not old_ok:
            rescued.append(record)
        elif old_ok and not new_ok:
            regressed.append(record)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in output_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "total": len(rows),
        "old_pass": old_pass,
        "old_score": old_pass / max(len(rows), 1),
        "new_pass": new_pass,
        "new_score": new_pass / max(len(rows), 1),
        "rescued": len(rescued),
        "regressed": len(regressed),
        "rescued_examples": rescued[:20],
        "regressed_examples": regressed[:20],
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
