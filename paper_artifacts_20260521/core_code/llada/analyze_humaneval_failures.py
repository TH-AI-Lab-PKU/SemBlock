from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


def _extract_first_response_text(record: dict) -> str:
    filtered_resps = record.get("filtered_resps")
    if not isinstance(filtered_resps, list) or not filtered_resps:
        return ""
    first_group = filtered_resps[0]
    if not isinstance(first_group, list) or not first_group:
        return ""
    return str(first_group[0])


def _infer_failure_type(stderr_text: str, candidate_code: str) -> str:
    if "AssertionError" in stderr_text:
        return "AssertionError"
    if "TypeError" in stderr_text:
        return "TypeError"
    if "NameError" in stderr_text:
        return "NameError"
    if "SyntaxError" in stderr_text:
        return "SyntaxError"
    if "IndentationError" in stderr_text:
        return "IndentationError"
    if "ValueError" in stderr_text or "IndexError" in stderr_text or "KeyError" in stderr_text:
        return "RuntimeError"

    lowered = candidate_code.lower()
    prompt_markers = (
        "to solve this problem",
        "to implement the function",
        "fix =",
        "add more test cases",
    )
    if any(marker in lowered for marker in prompt_markers):
        return "PromptLeakage"
    return "UnknownError"


_CHECK_CALL_LINE_PATTERN = re.compile(r"^\s+check\([A-Za-z_][A-Za-z0-9_]*\)\s*$")


def _normalize_test_code(test_code: str) -> str:
    normalized_lines: list[str] = []
    for line in str(test_code).splitlines():
        if _CHECK_CALL_LINE_PATTERN.match(line):
            normalized_lines.append(line.lstrip())
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def classify_failure(
    candidate_code: str,
    test_code: str,
    *,
    python_executable: str = sys.executable,
    timeout_seconds: int = 10,
) -> dict[str, str | int | None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "humaneval_exec.py"
        normalized_test_code = _normalize_test_code(test_code)
        script_path.write_text(
            candidate_code.rstrip() + "\n\n" + normalized_test_code.rstrip() + "\n",
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [python_executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "failure_type": "TimeoutError",
                "return_code": None,
                "stderr": "TimeoutExpired",
            }

    if completed.returncode == 0:
        return {
            "failure_type": None,
            "return_code": int(completed.returncode),
            "stderr": completed.stderr,
        }

    failure_type = _infer_failure_type(completed.stderr, candidate_code)
    return {
        "failure_type": failure_type,
        "return_code": int(completed.returncode),
        "stderr": completed.stderr,
    }


def analyze_samples_file(
    samples_path: str | Path,
    *,
    python_executable: str = sys.executable,
    timeout_seconds: int = 10,
) -> dict[str, object]:
    samples_path = Path(samples_path)
    failure_type_counts: Counter[str] = Counter()
    failed_examples: list[dict[str, object]] = []
    total_samples = 0
    passed_samples = 0

    with open(samples_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            total_samples += 1
            record = json.loads(line)
            pass_at_1 = float(record.get("pass@1", 0.0))
            if pass_at_1 >= 1.0:
                passed_samples += 1
                continue

            candidate_code = _extract_first_response_text(record)
            test_code = str(record.get("target", ""))
            classification = classify_failure(
                candidate_code,
                test_code,
                python_executable=python_executable,
                timeout_seconds=timeout_seconds,
            )
            failure_type = str(classification.get("failure_type") or "UnknownError")
            failure_type_counts[failure_type] += 1
            failed_examples.append(
                {
                    "task_id": record.get("doc", {}).get("task_id"),
                    "failure_type": failure_type,
                    "return_code": classification.get("return_code"),
                }
            )

    return {
        "samples_path": str(samples_path),
        "total_samples": int(total_samples),
        "passed_samples": int(passed_samples),
        "failed_samples": int(total_samples - passed_samples),
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "failed_examples": failed_examples,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze failure types from lm-eval HumanEval samples JSONL.")
    parser.add_argument("--samples_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--timeout_seconds", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = analyze_samples_file(
        args.samples_path,
        python_executable=args.python_executable,
        timeout_seconds=args.timeout_seconds,
    )
    output_text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
