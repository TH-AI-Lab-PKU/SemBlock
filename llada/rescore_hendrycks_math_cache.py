from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import Dataset, load_dataset


DATASET_PATH = "EleutherAI/hendrycks_math"
SUBTASKS: List[Tuple[str, str]] = [
    ("algebra", "algebra"),
    ("counting_and_prob", "counting_and_probability"),
    ("geometry", "geometry"),
    ("intermediate_algebra", "intermediate_algebra"),
    ("num_theory", "number_theory"),
    ("prealgebra", "prealgebra"),
    ("precalc", "precalculus"),
]


def last_boxed_only_string(string: str | None) -> str | None:
    if string is None:
        return None

    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(string: str | None) -> str | None:
    if string is None:
        return None

    if "\\boxed " in string:
        left = "\\boxed "
        if not string.startswith(left):
            raise ValueError(f"Malformed boxed string: {string!r}")
        return string[len(left) :]

    left = "\\boxed{"
    if not string.startswith(left) or not string.endswith("}"):
        raise ValueError(f"Malformed boxed string: {string!r}")
    return string[len(left) : -1]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a = substr[0]
            b = substr[1]
            if b != "{":
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    new_str += "{" + a + "}{" + b + "}"
            else:
                if len(substr) > 2:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
                else:
                    new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a_int = int(a)
        b_int = int(b)
    except ValueError:
        return string
    if string != f"{a_int}/{b_int}":
        return string
    return f"\\frac{{{a_int}}}{{{b_int}}}"


def remove_right_units(string: str) -> str:
    if "\\text{ " not in string:
        return string
    splits = string.split("\\text{ ")
    if len(splits) != 2:
        return string
    return splits[0]


def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def is_equiv(prediction: str | None, target: str | None) -> bool:
    if prediction is None and target is None:
        return True
    if prediction is None or target is None:
        return False
    try:
        return strip_string(prediction) == strip_string(target)
    except Exception:
        return prediction == target


def process_docs(dataset: Dataset) -> Dataset:
    def _process_doc(doc: Dict[str, str]) -> Dict[str, str]:
        return {
            "problem": doc["problem"],
            "solution": doc["solution"],
            "answer": remove_boxed(last_boxed_only_string(doc["solution"])),
        }

    return dataset.map(_process_doc)


def load_cached_outputs(cache_jsonl: Path) -> List[str]:
    outputs: List[str] = []
    with cache_jsonl.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if not isinstance(record, str):
                raise ValueError(
                    f"Expected each cache line to decode to a string, got {type(record).__name__} at line {line_number}"
                )
            outputs.append(record)
    if not outputs:
        raise ValueError(f"No outputs found in {cache_jsonl}")
    return outputs


def score_cached_outputs(
    *,
    cached_outputs: Sequence[str],
    max_samples: int | None,
) -> Dict[str, object]:
    total_expected = 0
    by_subset: Dict[str, Dict[str, object]] = {}
    output_index = 0
    overall_correct = 0
    overall_boxed = 0

    for subset_alias, dataset_name in SUBTASKS:
        dataset = process_docs(load_dataset(DATASET_PATH, dataset_name, split="test", trust_remote_code=True))
        subset_total = len(dataset)
        if max_samples is not None:
            remaining = max_samples - total_expected
            if remaining <= 0:
                break
            subset_total = min(subset_total, remaining)
            dataset = dataset.select(range(subset_total))

        subset_correct = 0
        subset_boxed = 0
        for doc in dataset:
            if output_index >= len(cached_outputs):
                raise ValueError(
                    f"Cache ended early at output {output_index}, expected at least {total_expected + 1} outputs"
                )
            output_text = cached_outputs[output_index]
            boxed = last_boxed_only_string(output_text)
            extracted = remove_boxed(boxed) if boxed is not None else None
            correct = is_equiv(extracted, doc["answer"])
            if boxed is not None:
                subset_boxed += 1
            if correct:
                subset_correct += 1
            output_index += 1
            total_expected += 1

        by_subset[subset_alias] = {
            "samples": subset_total,
            "boxed_predictions": subset_boxed,
            "correct": subset_correct,
            "exact_match": (subset_correct / subset_total) if subset_total else 0.0,
        }
        overall_correct += subset_correct
        overall_boxed += subset_boxed

    if max_samples is None and output_index != len(cached_outputs):
        raise ValueError(
            f"Cache has {len(cached_outputs)} outputs, but hendrycks_math scoring consumed {output_index} outputs"
        )

    return {
        "dataset_path": DATASET_PATH,
        "subtasks": [alias for alias, _ in SUBTASKS],
        "samples": output_index,
        "boxed_predictions": overall_boxed,
        "correct": overall_correct,
        "exact_match": (overall_correct / output_index) if output_index else 0.0,
        "by_subset": by_subset,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline re-score cached hendrycks_math generations by extracting the final "
            "\\boxed{...} answer from each cached output."
        )
    )
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        required=True,
        help="Path to the lm-eval cache file, typically rank_0.jsonl.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to save the score summary as JSON.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional cap for smoke checks; by default all cached outputs are scored.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    cached_outputs = load_cached_outputs(args.cache_jsonl)
    summary = score_cached_outputs(cached_outputs=cached_outputs, max_samples=args.max_samples)
    summary["cache_jsonl"] = str(args.cache_jsonl.resolve())
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
        summary["output_json"] = str(args.output_json.resolve())

    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
