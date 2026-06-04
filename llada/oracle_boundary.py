"""
Oracle block-size extraction from reference answers.

For the oracle / upper-bound experiment: given a gold reference answer, extract
semantic segment boundaries and convert them into a list of token-level block
sizes that can be fed directly into the AdaBlock generate functions.

Supported task types
--------------------
math  : splits on newlines and GSM8K-style "####" markers (one reasoning step
        per block).
nl    : splits on sentence boundaries (. ? !).
code  : splits on top-level keyword / de-indent boundaries.

Typical usage
-------------
    from oracle_boundary import extract_oracle_block_sizes

    oracle_sizes = extract_oracle_block_sizes(
        reference_text, tokenizer, task_type="math", max_length=512
    )
    # Then pass to generate:
    #   generate_adablock(..., block_strategy="oracle", oracle_block_sizes=oracle_sizes)
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Reference-text splitters
# ---------------------------------------------------------------------------

def _split_math_reference(text: str) -> list:
    """
    Split a math / GSM8K reference into reasoning steps.

    GSM8K gold answers look like::

        Natalia sold 48/2 = <<48/2=24>>24 clips in May.
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
        #### 72

    Each non-empty line becomes one semantic block.
    """
    segments = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            segments.append(line)
    return segments


def _split_nl_reference(text: str) -> list:
    """Split a natural-language reference on sentence boundaries."""
    parts = re.split(r'(?<=[.?!])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


_TOP_LEVEL_KW = re.compile(
    r'^(def|class|if|elif|else|for|while|return|with|try|except|finally'
    r'|import|from|async|yield)\b'
)

def _split_code_reference(text: str) -> list:
    """Split a code reference on top-level statement / dedent boundaries."""
    lines = text.split('\n')
    segments = []
    current: list = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and current and _TOP_LEVEL_KW.match(stripped):
            joined = '\n'.join(current)
            if joined.strip():
                segments.append(joined)
            current = [line]
        else:
            current.append(line)
    if current:
        joined = '\n'.join(current)
        if joined.strip():
            segments.append(joined)
    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_oracle_block_sizes(
    reference_text: str,
    tokenizer,
    task_type: str = 'math',
    max_length: int = 512,
) -> list:
    """
    Extract oracle block sizes from a reference answer.

    Each semantic segment is tokenized (with a trailing newline to match how
    the model will actually generate the text).  The function returns a list of
    integer token counts – one entry per segment – that sum to at most
    ``max_length``.

    Args:
        reference_text : Gold / reference answer text.
        tokenizer      : HuggingFace tokenizer.
        task_type      : ``'math'`` | ``'nl'`` | ``'code'``.
        max_length     : Total token budget; blocks are truncated once reached.

    Returns:
        List[int] of block sizes.  Empty list if reference is empty or
        tokenization yields zero tokens.
    """
    if not reference_text or not reference_text.strip():
        return []

    if task_type == 'math':
        segments = _split_math_reference(reference_text)
    elif task_type == 'code':
        segments = _split_code_reference(reference_text)
    else:
        segments = _split_nl_reference(reference_text)

    block_sizes = []
    total_tokens = 0
    for seg in segments:
        if not seg.strip():
            continue
        # Include a trailing newline so the token count matches how the model
        # generates each reasoning step (each step ends with \n in practice).
        seg_with_sep = seg + '\n'
        token_ids = tokenizer(seg_with_sep, add_special_tokens=False)['input_ids']
        n = len(token_ids)
        if n == 0:
            continue
        remaining = max_length - total_tokens
        if remaining <= 0:
            break
        block_sizes.append(min(n, remaining))
        total_tokens += min(n, remaining)
        if total_tokens >= max_length:
            break

    return block_sizes


def gsm8k_oracle_block_sizes(doc: dict, tokenizer, max_length: int = 512) -> list:
    """
    Convenience wrapper for GSM8K documents from lm-eval.

    lm-eval exposes the gold answer in ``doc['answer']``.  This function
    extracts oracle block sizes assuming math/step-by-line segmentation.

    Args:
        doc        : lm-eval document dict (must contain ``'answer'`` key).
        tokenizer  : HuggingFace tokenizer.
        max_length : Total token budget.

    Returns:
        List[int] of oracle block sizes, or ``[]`` if not applicable.
    """
    reference = doc.get('answer', '') or doc.get('target', '')
    if not reference:
        return []
    return extract_oracle_block_sizes(
        str(reference), tokenizer, task_type='math', max_length=max_length
    )
