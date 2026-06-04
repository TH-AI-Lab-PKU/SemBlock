# Copyright 2025 AdaBlock-dLLM Experiment
#
# Prior-based semantic boundary detection for block-size scheduling.
# Replaces AdaBlock's confidence-based VB logic with domain-prior rules:
#   - code  : AST-level boundaries (tree-sitter) or heuristic keyword/indent
#   - nl    : sentence boundaries (. ? ! \n), then clause boundaries (, ;)
#   - math  : line breaks, logical connectives, step markers

import re

# Try to import tree-sitter for code tasks.  Falls back to heuristics if absent.
try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser as TSParser
    _PY_LANGUAGE = Language(tspython.language())
    _TS_AVAILABLE = True
except Exception:
    _TS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_CODE_KEYWORDS = re.compile(
    r'\b(def|class|if|elif|else|for|while|return|with|try|except|finally|import|from|async|yield)\b'
)

_NL_SENTENCE_END = re.compile(r'[.?!](\s|$)')
_NL_CLAUSE_END   = re.compile(r'[,;](\s|$)')

_MATH_LINE_END   = re.compile(r'\n')
_MATH_CONNECTIVE = re.compile(
    r'\b(therefore|thus|hence|so|since|because|we have|we get|it follows)\b',
    re.IGNORECASE,
)
_MATH_STEP_MARKER = re.compile(r'<<.*?>>|####')


def _find_first_match(text: str, pattern: re.Pattern) -> int:
    """Return the character position right after the first match, or -1."""
    m = pattern.search(text)
    if m:
        return m.end()
    return -1



def choose_boundary_length_from_scores(scores, default_block_length: int, threshold: float | None = None) -> int:
    """
    Pick the earliest semantic block end implied by the highest score.
    Falls back to default_block_length when the best score is below threshold.
    """
    if default_block_length < 1:
        raise ValueError("default_block_length must be positive")
    if scores is None:
        return default_block_length

    values = list(scores)
    if not values:
        return default_block_length

    best_idx, best_score = max(enumerate(values), key=lambda item: float(item[1]))
    if threshold is not None and float(best_score) < float(threshold):
        return default_block_length
    return min(best_idx + 1, default_block_length)


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific boundary finders
# ─────────────────────────────────────────────────────────────────────────────

def _code_boundary_heuristic(text: str) -> int:
    """
    Heuristic code boundary: first newline where the *next* line starts a
    top-level keyword, or where indentation decreases.
    Returns a char position (end of that newline), or -1 if none found.
    """
    lines = text.split('\n')
    # We look at pairs of consecutive lines.
    cumulative_len = 0
    for i, line in enumerate(lines[:-1]):
        cumulative_len += len(line) + 1  # +1 for the '\n'
        next_line = lines[i + 1]
        stripped_next = next_line.lstrip()
        if not stripped_next:
            continue
        # Break before keyword lines
        if _CODE_KEYWORDS.match(stripped_next):
            return cumulative_len
        # Break on de-dentation (indent decreases)
        curr_indent = len(line) - len(line.lstrip())
        next_indent = len(next_line) - len(next_line.lstrip())
        if next_indent < curr_indent and next_indent == 0:
            return cumulative_len
    return -1


def _code_boundary_ast(text: str) -> int:
    """
    Tree-sitter AST boundary: end of the first complete top-level statement.
    Returns char position or -1.
    """
    if not _TS_AVAILABLE:
        return -1
    try:
        parser = TSParser(_PY_LANGUAGE)
        tree = parser.parse(text.encode())
        root = tree.root_node
        # Walk top-level children and return end of the first non-error complete node.
        for child in root.children:
            if child.type in ('ERROR', 'comment'):
                continue
            if not child.has_error and child.end_byte > 0:
                return child.end_byte  # byte offset == char offset for ASCII
    except Exception:
        pass
    return -1


def _find_code_boundary(token_ids: list, tokenizer, default_block_length: int) -> int:
    """
    Scan the window token-by-token, decode cumulative text, and find the
    first code semantic boundary.  Returns a block length (1-indexed).
    """
    text = ""
    best_pos = -1  # char-based best boundary position

    for i, tok_id in enumerate(token_ids):
        token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        text += token_text

        # 1) Try AST boundary on current text
        ast_pos = _code_boundary_ast(text)
        if ast_pos > 0 and best_pos < 0:
            best_pos = ast_pos

        # 2) Heuristic: keyword or de-dent at a newline
        heur_pos = _code_boundary_heuristic(text)
        if heur_pos > 0 and best_pos < 0:
            best_pos = heur_pos

        # If we found a boundary that ends within the current token, stop.
        if best_pos > 0 and best_pos <= len(text):
            return i + 1  # block ends after position i (1-indexed)

    return default_block_length


def _find_nl_boundary(token_ids: list, tokenizer, default_block_length: int) -> int:
    """
    Scan for sentence-boundary (. ? ! \n) and, if none found within the
    window, fall back to clause boundary (, ;).
    """
    text = ""
    sentence_end_pos = -1
    clause_end_pos   = -1

    for i, tok_id in enumerate(token_ids):
        token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        text += token_text

        if sentence_end_pos < 0:
            m = _NL_SENTENCE_END.search(text)
            if m:
                sentence_end_pos = m.end()
        if clause_end_pos < 0:
            m = _NL_CLAUSE_END.search(text)
            if m:
                clause_end_pos = m.end()

        # Prefer sentence end
        if sentence_end_pos > 0 and sentence_end_pos <= len(text):
            return i + 1
    # Fall back to clause end if no sentence end found
    if clause_end_pos > 0:
        # Find which token index corresponds to clause_end_pos
        text2 = ""
        for i, tok_id in enumerate(token_ids):
            token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
            text2 += token_text
            if len(text2) >= clause_end_pos:
                return i + 1

    return default_block_length


def _find_math_boundary(token_ids: list, tokenizer, default_block_length: int) -> int:
    """
    Scan for math step boundaries: newline, logical connectives, step markers.
    Priority order: step markers > connectives > newlines.
    """
    text = ""
    newline_pos     = -1
    connective_pos  = -1
    step_marker_pos = -1

    for i, tok_id in enumerate(token_ids):
        token_text = tokenizer.decode([tok_id], skip_special_tokens=False)
        text += token_text

        if step_marker_pos < 0:
            m = _MATH_STEP_MARKER.search(text)
            if m:
                step_marker_pos = m.end()
        if connective_pos < 0:
            m = _MATH_CONNECTIVE.search(text)
            if m:
                connective_pos = m.start()  # break *before* connective
        if newline_pos < 0:
            m = _MATH_LINE_END.search(text)
            if m:
                newline_pos = m.end()

        # Check in priority order
        target_pos = -1
        if step_marker_pos > 0 and step_marker_pos <= len(text):
            target_pos = step_marker_pos
        elif connective_pos > 0 and connective_pos <= len(text):
            target_pos = connective_pos
        elif newline_pos > 0 and newline_pos <= len(text):
            target_pos = newline_pos

        if target_pos > 0:
            return i + 1

    return default_block_length


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_block_length_prior(
    predicted_tokens,
    prompt,
    gen_length: int,
    generated_length: int,
    default_block_length: int,
    tokenizer,
    task_type: str = 'nl',
    boundary_window_ratio: float = 0.25,
) -> int:
    """
    Compute block length using domain-prior semantic boundaries.

    Args:
        predicted_tokens : (1, full_seq_len) long tensor — argmax of current logits.
        prompt           : (1, prompt_len) long tensor.
        gen_length       : total generation budget (tokens).
        generated_length : tokens already committed.
        default_block_length : fallback block size.
        tokenizer        : HuggingFace tokenizer (for decoding).
        task_type        : 'code' | 'nl' | 'math'.
        boundary_window_ratio : look-ahead window as fraction of gen_length.

    Returns:
        Integer block length in [1, min(default_block_length, remaining)].
    """
    prompt_length = prompt.shape[1]
    block_start   = prompt_length + generated_length
    remaining     = gen_length - generated_length
    window_size   = min(max(int(boundary_window_ratio * gen_length), 1), remaining)
    cap           = min(default_block_length, remaining)

    window_ids = predicted_tokens[0, block_start:block_start + window_size].tolist()

    if task_type == 'code':
        bl = _find_code_boundary(window_ids, tokenizer, cap)
    elif task_type == 'math':
        bl = _find_math_boundary(window_ids, tokenizer, cap)
    else:
        bl = _find_nl_boundary(window_ids, tokenizer, cap)

    return min(bl, remaining)
