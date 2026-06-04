# HumanEval Boundary Case Study

Date: 2026-05-23

This folder contains a concrete HumanEval example where AdaBlock fails but our
SOTA code-boundary setup succeeds. The selected task is `HumanEval/128`
(`prod_signs`).

Important caveat: the preserved LLaDA-1.5 HumanEval B0=32 OpenCompass output
does include this task and marks it as passed, but that OpenCompass run did not
save boundary traces. Therefore the boundary visualization below uses the
preserved code-head SOTA trace from the LLaDA-Instruct B0=32 archive and the
corresponding AdaBlock trace. This is the locally verifiable pair with both
sample correctness and boundary decisions.

## Selected Sample

```python
def prod_signs(arr):
    """
    You are given an array arr of integers and you need to return
    sum of magnitudes of integers multiplied by product of all signs
    of each number in the array, represented by 1, -1 or 0.
    Note: return None for empty arr.

    Example:
    >>> prod_signs([1, 2, 2, -4]) == -9
    >>> prod_signs([0, 1]) == 0
    >>> prod_signs([]) == None
    """
```

## Correctness

| Method | Model / run | Result |
|---|---|---:|
| AdaBlock | HE32 dual-cache AdaBlock sample run | failed |
| Ours | code-head SOTA, generalized-router-v3, B0=32 | passed |
| Ours | LLaDA-1.5 B0=32 OpenCompass SOTA output | passed |

## Generated Outputs

AdaBlock output:

```python
    if not arr:
        return None

    result = 0
    for num in arr:
        if num > 0:
            result += 1
        elif num < 0:
            result -= 1
        else:
            result += 0

    return result
```

Why it fails: it accumulates the signs, not the sum of magnitudes multiplied by
the product of signs.

Ours output:

```python
    if not arr:
        return None

    sum_magnitude = 0
    product_sign = 1

    for num in arr:
        sum_magnitude += abs(num)
        if num > 0:
            product_sign *= 1
        elif num < 0:
            product_sign *= -1
        else:
            product_sign *= 0

    return sum_magnitude * product_sign
```

Why it passes: it separately tracks magnitude and sign, then returns the
required product.

## Boundary Visualization

`/B` marks a boundary in the visible generated code. Offsets beyond the visible
decoded completion were omitted because the trace records boundary decisions up
to the fixed 512-token generation horizon.

### Ours: Code Head / Semantic Boundary

Visible token offsets used: `1, 33, 35, 67, 72`

```text
    /B  if not arr:\n        return None\n\n    sum_magnitude = 0\n    product_sign = 1\n\n    for num in arr: /B \n        /B  sum_magnitude += abs(num)\n        if num > 0:\n            product_sign *= 1\n        elif num < 0:\n            /B  product_sign *= -1 /B \n        else:\n            product_sign *= 0\n\n    return sum_magnitude * product_sign
```

Reading: our boundaries preserve the high-level algorithmic phases: empty case,
state initialization plus loop entry, magnitude update, negative-sign branch,
and final sign update. The generation keeps the semantic units of the solution
intact.

### AdaBlock

Visible token offsets used: `10, 11, 43, 46, 52, 56, 62, 63`

```text
    if not arr:\n        return None\n /B \n /B     result = 0\n    for num in arr:\n        if num > 0:\n            result += 1\n        elif num <  /B 0:\n /B             result -= 1\n /B         else:\n /B             result += 0\n /B \n /B     return result
```

Reading: AdaBlock places several boundaries around local syntax and line-break
cues, including a split inside the condition `num < 0`. This produces a much
more surface-form-driven segmentation and does not protect the core semantic
distinction between "sum of magnitudes" and "product of signs".

## Boundary Overlap for This Sample

From `humaneval_sota_vs_adablock_all.json`:

```text
semantic_boundary_count = 26
adablock_boundary_count = 35
exact_intersection_count = 2
exact_jaccard = 0.03389830508474576
```

This is a useful paper example because the behavior matches the broader
overlap result: the two boundary systems are not simply equivalent, and the
semantic boundary corresponds better to the successful program structure.

## Source Files

| Evidence | Path |
|---|---|
| Ours SOTA sample output | `llada/experiments/semantic_boundary_indep/SOTA_ARCHIVE_20260512_generalized_v3_4695/results/full_humaneval_router_generalized_v3_sharded/merged_samples_humaneval.jsonl` |
| AdaBlock sample output | `llada/experiments/semantic_boundary_indep/20260510_adablock_he32_dual_cache/__home__nvme03__workspace__models__GSAI-ML__LLaDA-8B-Instruct/samples_llada_humaneval_subset_2026-05-10T15-05-35.251917.jsonl` |
| Boundary overlap summary | `llada/experiments/semantic_boundary_indep/20260516_boundary_overlap/humaneval_sota_vs_adablock_all.json` |
| Ours SOTA traces | `llada/experiments/semantic_boundary_indep/20260512_full_humaneval_router_generalized_v3_sharded/shard*/traces/rank_0.jsonl` |
| AdaBlock traces | `llada/experiments/semantic_boundary_indep/20260516_boundary_overlap_full_corresponding/traces/humaneval/adablock_full/rank_0.jsonl` |
| LLaDA-1.5 B0=32 OpenCompass result | `/home/nvme02/workspace/wm_mem/wm/opencompass/outputs/llada_1p5_humaneval/20260519_093200/results/llada-1.5-8b-instruct/openai_humaneval.json` |

