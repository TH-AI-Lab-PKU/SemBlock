#!/usr/bin/env python3
"""Append targeted non-HumanEval semantic-boundary rows."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from transformers import AutoTokenizer

from build_task_conditioned_phase_boundary_jsonl import (
    _serialize_completion_aligned_prompt,
    build_serialized_training_record,
    extract_function_completion_parts,
    resolve_tokenizer_path,
)


SAMPLES = [
    {
        "task_id": "synthetic_digit_sum_abs_int",
        "signature": "def digit_sum_abs(n: int) -> int:",
        "description": "Return the sum of decimal digits in the absolute value of n.",
        "code": """def digit_sum_abs(n: int) -> int:
    n = abs(n)
    total = 0
    while n:
        total += n % 10
        n //= 10
    return total""",
    },
    {
        "task_id": "synthetic_digit_product_skip_zero",
        "signature": "def digit_product_nonzero(n: int) -> int:",
        "description": "Return the product of the nonzero decimal digits of n. Return 0 when n has no nonzero digits.",
        "code": """def digit_product_nonzero(n: int) -> int:
    n = abs(n)
    product = 1
    seen = False
    for ch in str(n):
        digit = int(ch)
        if digit:
            product *= digit
            seen = True
    return product if seen else 0""",
    },
    {
        "task_id": "synthetic_largest_number_digit_sum",
        "signature": "def largest_digit_sum(values: list[int]) -> int:",
        "description": "Find the largest positive integer in values and return the sum of its decimal digits. Return 0 if none exists.",
        "code": """def largest_digit_sum(values: list[int]) -> int:
    best = None
    for value in values:
        if value > 0 and (best is None or value > best):
            best = value
    if best is None:
        return 0
    return sum(int(ch) for ch in str(best))""",
    },
    {
        "task_id": "synthetic_count_even_digits",
        "signature": "def count_even_digits(n: int) -> int:",
        "description": "Count how many decimal digits of n are even.",
        "code": """def count_even_digits(n: int) -> int:
    count = 0
    for ch in str(abs(n)):
        if int(ch) % 2 == 0:
            count += 1
    return count""",
    },
    {
        "task_id": "synthetic_words_with_k_consonants",
        "signature": "def words_with_k_consonants(text: str, k: int) -> list[str]:",
        "description": "Return words from text that contain exactly k English consonant letters, preserving order.",
        "code": """def words_with_k_consonants(text: str, k: int) -> list[str]:
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    result = []
    for word in text.split():
        count = sum(1 for char in word if char in consonants)
        if count == k:
            result.append(word)
    return result""",
    },
    {
        "task_id": "synthetic_count_letter_classes",
        "signature": "def count_letter_classes(text: str) -> tuple[int, int]:",
        "description": "Return the number of vowels and consonants in text, ignoring non letters.",
        "code": """def count_letter_classes(text: str) -> tuple[int, int]:
    vowels = set("aeiouAEIOU")
    consonants = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
    vowel_count = 0
    consonant_count = 0
    for char in text:
        if char in vowels:
            vowel_count += 1
        elif char in consonants:
            consonant_count += 1
    return vowel_count, consonant_count""",
    },
    {
        "task_id": "synthetic_filter_mixed_case_words",
        "signature": "def filter_by_uppercase_count(words: list[str], target: int) -> list[str]:",
        "description": "Return words whose number of uppercase letters equals target.",
        "code": """def filter_by_uppercase_count(words: list[str], target: int) -> list[str]:
    result = []
    for word in words:
        uppercase_count = sum(1 for char in word if char.isupper())
        if uppercase_count == target:
            result.append(word)
    return result""",
    },
    {
        "task_id": "synthetic_sign_magnitude_zero",
        "signature": "def signed_magnitude_total(values: list[int]) -> int | None:",
        "description": "Return the sum of magnitudes multiplied by the product of signs. A zero value makes the sign product zero.",
        "code": """def signed_magnitude_total(values: list[int]) -> int | None:
    if not values:
        return None
    magnitude = 0
    sign = 1
    for value in values:
        magnitude += abs(value)
        if value > 0:
            sign *= 1
        elif value < 0:
            sign *= -1
        else:
            sign = 0
    return magnitude * sign""",
    },
    {
        "task_id": "synthetic_product_sign_with_zero",
        "signature": "def product_sign(values: list[int]) -> int:",
        "description": "Return 1, -1, or 0 for the sign of the product of all integers in values.",
        "code": """def product_sign(values: list[int]) -> int:
    sign = 1
    for value in values:
        if value == 0:
            return 0
        if value < 0:
            sign *= -1
    return sign""",
    },
    {
        "task_id": "synthetic_parse_symbolic_notes",
        "signature": "def parse_symbolic_notes(notes: str) -> list[int]:",
        "description": "Map space separated note symbols to durations using full tokens, preserving order.",
        "code": """def parse_symbolic_notes(notes: str) -> list[int]:
    durations = {"o": 4, "o|": 2, ".|": 1}
    result = []
    for token in notes.split():
        result.append(durations[token])
    return result""",
    },
    {
        "task_id": "synthetic_closest_float_pair",
        "signature": "def closest_float_pair(numbers: list[float]) -> tuple[float, float]:",
        "description": "Return the pair of values with the smallest absolute difference from a list with at least two values.",
        "code": """def closest_float_pair(numbers: list[float]) -> tuple[float, float]:
    ordered = sorted(numbers)
    best_pair = (ordered[0], ordered[1])
    best_gap = abs(ordered[1] - ordered[0])
    for i in range(1, len(ordered) - 1):
        gap = abs(ordered[i + 1] - ordered[i])
        if gap < best_gap:
            best_gap = gap
            best_pair = (ordered[i], ordered[i + 1])
    return best_pair""",
    },
    {
        "task_id": "synthetic_largest_proper_divisor",
        "signature": "def largest_proper_divisor(n: int) -> int:",
        "description": "Return the largest positive divisor of n that is smaller than n.",
        "code": """def largest_proper_divisor(n: int) -> int:
    for candidate in range(n - 1, 0, -1):
        if n % candidate == 0:
            return candidate
    return 1""",
    },
    {
        "task_id": "synthetic_sort_every_third",
        "signature": "def sort_every_third(values: list[int]) -> list[int]:",
        "description": "Sort the values at indices divisible by three while leaving all other positions unchanged.",
        "code": """def sort_every_third(values: list[int]) -> list[int]:
    result = list(values)
    indices = list(range(0, len(values), 3))
    sorted_values = sorted(values[index] for index in indices)
    for index, value in zip(indices, sorted_values):
        result[index] = value
    return result""",
    },
    {
        "task_id": "synthetic_cycle_triplets",
        "signature": "def cycle_triplets(text: str) -> str:",
        "description": "Cycle each complete group of three characters left by one position and leave shorter tail groups unchanged.",
        "code": """def cycle_triplets(text: str) -> str:
    pieces = []
    for i in range(0, len(text), 3):
        group = text[i:i + 3]
        if len(group) == 3:
            pieces.append(group[1:] + group[0])
        else:
            pieces.append(group)
    return "".join(pieces)""",
    },
    {
        "task_id": "synthetic_four_term_recurrence",
        "signature": "def four_term_recurrence(n: int) -> int:",
        "description": "Compute a recurrence where each term after the first four is the sum of the previous four terms.",
        "code": """def four_term_recurrence(n: int) -> int:
    values = [0, 0, 2, 0]
    if n < len(values):
        return values[n]
    for i in range(4, n + 1):
        values.append(values[i - 1] + values[i - 2] + values[i - 3] + values[i - 4])
    return values[n]""",
    },
    {
        "task_id": "synthetic_same_character_set",
        "signature": "def same_character_set(left: str, right: str) -> bool:",
        "description": "Return whether two strings contain exactly the same distinct characters, ignoring multiplicity and order.",
        "code": """def same_character_set(left: str, right: str) -> bool:
    return set(left) == set(right)""",
    },
    {
        "task_id": "synthetic_largest_prime_factor_generic",
        "signature": "def largest_prime_factor_generic(n: int) -> int:",
        "description": "Return the largest prime factor of a composite integer greater than one.",
        "code": """def largest_prime_factor_generic(n: int) -> int:
    factor = 2
    largest = 1
    while factor * factor <= n:
        while n % factor == 0:
            largest = factor
            n //= factor
        factor += 1
    return max(largest, n)""",
    },
    {
        "task_id": "synthetic_remaining_fruit_count",
        "signature": "def remaining_fruit_count(text: str, total: int) -> int:",
        "description": "Read two integer counts from a short fruit sentence and return how many fruits remain from total.",
        "code": """def remaining_fruit_count(text: str, total: int) -> int:
    numbers = [int(token) for token in text.split() if token.isdigit()]
    return total - sum(numbers[:2])""",
    },
    {
        "task_id": "synthetic_alternating_min_max_sort",
        "signature": "def alternating_min_max_sort(values: list[int]) -> list[int]:",
        "description": "Return values ordered by repeatedly taking the current minimum and then the current maximum.",
        "code": """def alternating_min_max_sort(values: list[int]) -> list[int]:
    values = sorted(values)
    result = []
    take_min = True
    while values:
        if take_min:
            result.append(values.pop(0))
        else:
            result.append(values.pop())
        take_min = not take_min
    return result""",
    },
    {
        "task_id": "synthetic_binary_digit_sum_as_binary",
        "signature": "def binary_digit_sum_as_binary(n: int) -> str:",
        "description": "Sum the bits in the binary representation of n and return that sum encoded as binary without a prefix.",
        "code": """def binary_digit_sum_as_binary(n: int) -> str:
    bit_sum = sum(int(ch) for ch in bin(n)[2:])
    return bin(bit_sum)[2:]""",
    },
    {
        "task_id": "synthetic_largest_even_in_range",
        "signature": "def largest_even_in_range(start: int, end: int) -> int:",
        "description": "Return the largest even integer in an inclusive range, or -1 if the range contains no even integer.",
        "code": """def largest_even_in_range(start: int, end: int) -> int:
    for value in range(end, start - 1, -1):
        if value % 2 == 0:
            return value
    return -1""",
    },
    {
        "task_id": "synthetic_last_descent_index",
        "signature": "def last_descent_index(values: list[int]) -> int:",
        "description": "Return the largest index whose value is smaller than the immediately previous value, or -1.",
        "code": """def last_descent_index(values: list[int]) -> int:
    result = -1
    for index in range(1, len(values)):
        if values[index] < values[index - 1]:
            result = index
    return result""",
    },
    {
        "task_id": "synthetic_fraction_product_is_integer",
        "signature": "def fraction_product_is_integer(left: str, right: str) -> bool:",
        "description": "Return whether the product of two fraction strings evaluates to an integer.",
        "code": """def fraction_product_is_integer(left: str, right: str) -> bool:
    left_num, left_den = map(int, left.split("/"))
    right_num, right_den = map(int, right.split("/"))
    return (left_num * right_num) % (left_den * right_den) == 0""",
    },
    {
        "task_id": "synthetic_sort_by_abs_digit_sum",
        "signature": "def sort_by_abs_digit_sum(values: list[int]) -> list[int]:",
        "description": "Sort integers by the sum of decimal digits of their absolute value, preserving original order on ties.",
        "code": """def sort_by_abs_digit_sum(values: list[int]) -> list[int]:
    def digit_sum(value: int) -> int:
        return sum(int(ch) for ch in str(abs(value)))
    return sorted(values, key=digit_sum)""",
    },
    {
        "task_id": "synthetic_any_order_right_triangle",
        "signature": "def any_order_right_triangle(a: int, b: int, c: int) -> bool:",
        "description": "Return whether three side lengths can form a right triangle in any side order.",
        "code": """def any_order_right_triangle(a: int, b: int, c: int) -> bool:
    sides = sorted([a, b, c])
    return sides[0] * sides[0] + sides[1] * sides[1] == sides[2] * sides[2]""",
    },
    {
        "task_id": "synthetic_evaluate_operator_chain",
        "signature": "def evaluate_operator_chain(operators: list[str], operands: list[int]) -> int:",
        "description": "Evaluate an arithmetic expression represented by operands with operators between them.",
        "code": """def evaluate_operator_chain(operators: list[str], operands: list[int]) -> int:
    expression = str(operands[0])
    for operator, operand in zip(operators, operands[1:]):
        expression += operator + str(operand)
    return eval(expression)""",
    },
    {
        "task_id": "synthetic_swapcase_or_reverse",
        "signature": "def swapcase_or_reverse(text: str) -> str:",
        "description": "Swap the case of letters in text; if there are no letters, return the reversed text.",
        "code": """def swapcase_or_reverse(text: str) -> str:
    if any(char.isalpha() for char in text):
        return "".join(char.swapcase() if char.isalpha() else char for char in text)
    return text[::-1]""",
    },
    {
        "task_id": "synthetic_even_digits_between_bounds",
        "signature": "def even_digits_between_bounds(a: int, b: int) -> list[int]:",
        "description": "Return even decimal digits between two positive bounds inclusive in ascending order.",
        "code": """def even_digits_between_bounds(a: int, b: int) -> list[int]:
    lower = min(a, b)
    upper = max(a, b)
    return [digit for digit in [2, 4, 6, 8] if lower <= digit <= upper]""",
    },
]


def build_record(tokenizer, sample: dict, repeat_index: int) -> dict:
    code = sample["code"].rstrip()
    parts = extract_function_completion_parts(code, synthetic_signature=sample["signature"])
    completion = parts.body_text if parts is not None else code
    annotation_code = parts.function_text if parts is not None else None
    annotation_start = parts.body_start_char if parts is not None else 0
    serialized = _serialize_completion_aligned_prompt(
        prompt_style="humaneval",
        task_description=sample["description"],
        synthetic_signature=sample["signature"],
        synthetic_tests=None,
        code_text=completion,
    )
    row = build_serialized_training_record(
        tokenizer=tokenizer,
        serialized_text=str(serialized["serialized_text"]),
        code_text=completion,
        code_start_char=int(serialized["code_start_char"]),
        section_offsets=dict(serialized["section_offsets"]),
        source_domain="synthetic_nonhe",
        source_view="leetcode_code_bridge",
        label_confidence="silver",
        condition_mask=int(serialized["condition_mask"]),
        max_length=4096,
        task_id=f"{sample['task_id']}_{repeat_index:04d}",
        public_test_metadata={
            "nonhe_aug_source": "targeted_failure_patterns_20260511",
            "targeted_pattern": sample["task_id"],
        },
        hard_mining_cluster=None,
        annotation_code_text=annotation_code,
        annotation_target_start_char=annotation_start,
    )
    row["task_description"] = sample["description"]
    row["synthetic_signature"] = sample["signature"]
    return row


def append_rows(path: Path, rows: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False)
            if "HumanEval" in encoded:
                raise ValueError("Refusing to write a row containing HumanEval.")
            handle.write(encoded + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--train_repeats", type=int, default=120)
    parser.add_argument("--valid_repeats", type=int, default=8)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base_dir / "train.jsonl", output_dir / "train.jsonl")
    shutil.copyfile(base_dir / "valid.jsonl", output_dir / "valid.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer_path(args.tokenizer_path), trust_remote_code=True)
    train_rows = [build_record(tokenizer, sample, idx) for idx in range(args.train_repeats) for sample in SAMPLES]
    valid_rows = [build_record(tokenizer, sample, idx) for idx in range(args.valid_repeats) for sample in SAMPLES]
    append_rows(output_dir / "train.jsonl", train_rows)
    append_rows(output_dir / "valid.jsonl", valid_rows)
    print(f"wrote {len(train_rows)} train rows and {len(valid_rows)} valid rows to {output_dir}")


if __name__ == "__main__":
    main()
