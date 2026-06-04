from __future__ import annotations

import ast
from typing import Dict, List, Optional, Tuple


PHASE_LABEL_VOCAB = [
    "signature_or_doc",
    "normalize",
    "init_state",
    "enumerate_or_iterate",
    "check_or_verify",
    "update_state",
    "postprocess",
    "return_emit",
]
PHASE_TO_ID = {name: idx for idx, name in enumerate(PHASE_LABEL_VOCAB)}
BOUNDARY_TYPE_VOCAB = [
    "safe_commit_boundary",
    "phase_change",
    "same_phase_stmt_split",
    "control_header_to_body",
    "block_exit_to_return",
    "helper_def_boundary",
]
BOUNDARY_TYPE_TO_ID = {name: idx + 1 for idx, name in enumerate(BOUNDARY_TYPE_VOCAB)}

NORMALIZE_CALL_NAMES = {
    "abs",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "list",
    "map",
    "set",
    "sorted",
    "str",
    "tuple",
    "zip",
}
NORMALIZE_ATTR_NAMES = {
    "lower",
    "read",
    "readline",
    "replace",
    "rstrip",
    "split",
    "splitlines",
    "strip",
    "upper",
}
MUTATING_ATTR_NAMES = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "remove",
    "reverse",
    "sort",
    "update",
}
POSTPROCESS_CALL_NAMES = {
    "dict",
    "list",
    "repr",
    "set",
    "sorted",
    "str",
    "tuple",
}
POSTPROCESS_ATTR_NAMES = {"format", "join"}


def _build_line_offsets(text: str) -> List[int]:
    offsets = [0]
    running = 0
    for line in text.splitlines(keepends=True):
        running += len(line)
        offsets.append(running)
    return offsets


def _char_index(line_offsets: List[int], lineno: int, col_offset: int) -> int:
    safe_line = max(1, min(lineno, len(line_offsets) - 1))
    return line_offsets[safe_line - 1] + max(0, int(col_offset))


def _node_span(node: ast.AST, line_offsets: List[int], fallback_end: int) -> Tuple[int, int]:
    start = _char_index(line_offsets, getattr(node, "lineno", 1), getattr(node, "col_offset", 0))
    end_lineno = getattr(node, "end_lineno", getattr(node, "lineno", 1))
    end_col = getattr(node, "end_col_offset", 0)
    end = _char_index(line_offsets, end_lineno, end_col)
    if end <= start:
        end = min(max(start + 1, fallback_end), fallback_end)
    return max(0, start), min(max(end, start + 1), fallback_end)


def _append_span(
    spans: List[Dict[str, object]],
    *,
    phase_name: str,
    char_start: int,
    char_end: int,
    confidence: str = "silver",
    segment_kind: str = "statement",
    depth: int = 0,
    is_control_header: bool = False,
    is_helper_definition: bool = False,
    node_type: Optional[str] = None,
) -> None:
    if char_end <= char_start:
        return
    spans.append(
        {
            "phase_id": PHASE_TO_ID[phase_name],
            "phase_name": phase_name,
            "char_start": int(char_start),
            "char_end": int(char_end),
            "label_confidence": confidence,
            "segment_kind": segment_kind,
            "depth": int(depth),
            "is_control_header": bool(is_control_header),
            "is_helper_definition": bool(is_helper_definition),
            "node_type": str(node_type or segment_kind),
        }
    )


def _is_docstring_expr(node: ast.stmt) -> bool:
    if not isinstance(node, ast.Expr):
        return False
    value = getattr(node, "value", None)
    return isinstance(value, ast.Constant) and isinstance(value.value, str)


def _call_identity(func: ast.AST) -> Tuple[Optional[str], Optional[str]]:
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        return None, func.attr
    return None, None


def _iter_call_identities(node: ast.AST) -> List[Tuple[Optional[str], Optional[str]]]:
    call_identities: List[Tuple[Optional[str], Optional[str]]] = []
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Call):
            call_identities.append(_call_identity(candidate.func))
    return call_identities


def _contains_string_literal(node: ast.AST) -> bool:
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            return True
    return False


def _assignment_value(node: ast.stmt) -> Optional[ast.AST]:
    if isinstance(node, ast.Assign):
        return node.value
    if isinstance(node, ast.AnnAssign):
        return node.value
    if isinstance(node, ast.AugAssign):
        return node.value
    return None


def _looks_like_normalization_value(value: Optional[ast.AST]) -> bool:
    if value is None:
        return False
    if isinstance(value, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return True
    for call_name, attr_name in _iter_call_identities(value):
        if call_name in NORMALIZE_CALL_NAMES or attr_name in NORMALIZE_ATTR_NAMES:
            return True
    return False


def _looks_like_mutation_value(value: Optional[ast.AST]) -> bool:
    if value is None:
        return False
    for _, attr_name in _iter_call_identities(value):
        if attr_name in MUTATING_ATTR_NAMES:
            return True
    return False


def _looks_like_postprocess_value(value: Optional[ast.AST]) -> bool:
    if value is None:
        return False
    if isinstance(value, ast.JoinedStr):
        return True
    if isinstance(value, ast.BinOp) and _contains_string_literal(value):
        return True
    for call_name, attr_name in _iter_call_identities(value):
        if call_name in POSTPROCESS_CALL_NAMES or attr_name in POSTPROCESS_ATTR_NAMES:
            return True
    return False


def _loop_header_span(
    node: ast.stmt,
    line_offsets: List[int],
    fallback_end: int,
) -> Optional[Tuple[int, int]]:
    if not isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return None
    start, end = _node_span(node, line_offsets, fallback_end)
    if getattr(node, "body", None):
        body_start = _char_index(
            line_offsets,
            getattr(node.body[0], "lineno", getattr(node, "lineno", 1)),
            getattr(node.body[0], "col_offset", 0),
        )
        end = min(end, max(body_start, start + 1))
    return start, end


def _classify_statement(
    node: ast.stmt,
    *,
    in_loop: bool,
    pre_control_region: bool,
) -> str:
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return "signature_or_doc"
    if isinstance(node, ast.Return):
        return "return_emit"
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return "enumerate_or_iterate"
    if isinstance(node, (ast.If, ast.Assert, ast.Try, ast.With, ast.AsyncWith, ast.Match, ast.Raise)):
        return "check_or_verify"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        value = _assignment_value(node)
        if isinstance(node, ast.AugAssign):
            return "update_state"
        if _looks_like_normalization_value(value) and pre_control_region and not in_loop:
            return "normalize"
        if in_loop:
            return "update_state"
        if pre_control_region:
            return "init_state"
        if _looks_like_postprocess_value(value):
            return "postprocess"
        return "update_state"
    if isinstance(node, ast.AugAssign):
        return "update_state"
    if isinstance(node, ast.Expr):
        if _is_docstring_expr(node):
            return "signature_or_doc"
        value = getattr(node, "value", None)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "print":
            return "return_emit"
        if _looks_like_mutation_value(value):
            if pre_control_region and not in_loop:
                return "normalize" if _looks_like_normalization_value(value) else "init_state"
            return "update_state"
        if _looks_like_normalization_value(value):
            return "normalize" if pre_control_region and not in_loop else ("update_state" if in_loop else "postprocess")
        if in_loop:
            return "update_state"
        if pre_control_region:
            return "init_state"
        return "postprocess"
    return "update_state" if in_loop else ("init_state" if pre_control_region else "postprocess")


def _walk_statements(
    statements: List[ast.stmt],
    *,
    spans: List[Dict[str, object]],
    line_offsets: List[int],
    code_length: int,
    in_loop: bool = False,
    depth: int = 0,
) -> None:
    first_control_seen = False
    for node in statements:
        phase_name = _classify_statement(
            node,
            in_loop=in_loop,
            pre_control_region=not first_control_seen,
        )
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            header_span = _loop_header_span(node, line_offsets, code_length)
            if header_span is not None:
                _append_span(
                    spans,
                    phase_name="enumerate_or_iterate",
                    char_start=header_span[0],
                    char_end=header_span[1],
                    confidence="gold",
                    segment_kind="control_header",
                    depth=depth,
                    is_control_header=True,
                    node_type=type(node).__name__,
                )
            _walk_statements(
                list(node.body),
                spans=spans,
                line_offsets=line_offsets,
                code_length=code_length,
                in_loop=True,
                depth=depth + 1,
            )
            if getattr(node, "orelse", None):
                _walk_statements(
                    list(node.orelse),
                    spans=spans,
                    line_offsets=line_offsets,
                    code_length=code_length,
                    in_loop=in_loop,
                    depth=depth + 1,
                )
            first_control_seen = True
            continue

        if isinstance(node, ast.If):
            start, end = _node_span(node, line_offsets, code_length)
            if node.body:
                body_start = _char_index(
                    line_offsets,
                    getattr(node.body[0], "lineno", getattr(node, "lineno", 1)),
                    getattr(node.body[0], "col_offset", 0),
                )
                end = min(end, max(body_start, start + 1))
            _append_span(
                spans,
                phase_name="check_or_verify",
                char_start=start,
                char_end=end,
                segment_kind="control_header",
                depth=depth,
                is_control_header=True,
                node_type=type(node).__name__,
            )
            _walk_statements(
                list(node.body),
                spans=spans,
                line_offsets=line_offsets,
                code_length=code_length,
                in_loop=in_loop,
                depth=depth + 1,
            )
            if node.orelse:
                _walk_statements(
                    list(node.orelse),
                    spans=spans,
                    line_offsets=line_offsets,
                    code_length=code_length,
                    in_loop=in_loop,
                    depth=depth + 1,
                )
            first_control_seen = True
            continue

        start, end = _node_span(node, line_offsets, code_length)
        confidence = "gold" if phase_name in {"enumerate_or_iterate", "check_or_verify", "update_state", "postprocess", "return_emit"} else "silver"
        _append_span(
            spans,
            phase_name=phase_name,
            char_start=start,
            char_end=end,
            confidence=confidence,
            segment_kind="statement",
            depth=depth,
            is_helper_definition=isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)),
            node_type=type(node).__name__,
        )
        first_control_seen = first_control_seen or isinstance(
            node,
            (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Return, ast.Try, ast.Match),
        )


def _compact_spans(spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda item: (int(item["char_start"]), int(item["char_end"])))
    compacted: List[Dict[str, object]] = [
        {
            "phase_id": int(ordered[0]["phase_id"]),
            "phase_name": ordered[0]["phase_name"],
            "char_start": int(ordered[0]["char_start"]),
            "char_end": int(ordered[0]["char_end"]),
            "label_confidence": ordered[0].get("label_confidence", "silver"),
        }
    ]
    for span in ordered[1:]:
        previous = compacted[-1]
        if (
            span["phase_name"] == previous["phase_name"]
            and int(span["char_start"]) <= int(previous["char_end"])
        ):
            previous["char_end"] = max(int(previous["char_end"]), int(span["char_end"]))
            if previous.get("label_confidence") != "gold" and span.get("label_confidence") == "gold":
                previous["label_confidence"] = "gold"
            continue
        compacted.append(
            {
                "phase_id": int(span["phase_id"]),
                "phase_name": span["phase_name"],
                "char_start": int(span["char_start"]),
                "char_end": int(span["char_end"]),
                "label_confidence": span.get("label_confidence", "silver"),
            }
        )
    return compacted


def _classify_boundary_type(left_span: Dict[str, object], right_span: Dict[str, object]) -> Optional[str]:
    if left_span.get("is_helper_definition") or right_span.get("is_helper_definition"):
        return "helper_def_boundary"
    if left_span.get("is_control_header") and int(right_span.get("depth", 0)) > int(left_span.get("depth", 0)):
        return "control_header_to_body"
    if right_span.get("phase_name") == "return_emit" and int(right_span.get("depth", 0)) < int(left_span.get("depth", 0)):
        return "block_exit_to_return"
    if right_span.get("phase_name") == "return_emit" and left_span.get("is_control_header"):
        return "block_exit_to_return"
    if left_span.get("phase_name") == right_span.get("phase_name"):
        if left_span.get("phase_name") == "signature_or_doc":
            return None
        return "same_phase_stmt_split"
    return "phase_change"


def _is_safe_commit_boundary(left_span: Dict[str, object], right_span: Dict[str, object], semantic_type: str) -> bool:
    if semantic_type == "control_header_to_body":
        return False
    if left_span.get("phase_name") == "signature_or_doc" and right_span.get("phase_name") == "signature_or_doc":
        return False
    if left_span.get("is_control_header") and int(right_span.get("depth", 0)) > int(left_span.get("depth", 0)):
        return False
    return True


def _derive_boundary_edges(code_length: int, spans: List[Dict[str, object]]) -> List[Dict[str, object]]:
    boundary_edges: List[Dict[str, object]] = []
    for index, span in enumerate(spans[:-1]):
        next_span = spans[index + 1]
        semantic_boundary_type = _classify_boundary_type(span, next_span)
        if semantic_boundary_type is None:
            continue
        if not _is_safe_commit_boundary(span, next_span, semantic_boundary_type):
            continue
        boundary_index = min(int(span["char_end"]) - 1, code_length - 1)
        if boundary_index < 0:
            continue
        boundary_type = "safe_commit_boundary"
        boundary_edges.append(
            {
                "char_position": int(boundary_index),
                "boundary_type": boundary_type,
                "boundary_type_id": int(BOUNDARY_TYPE_TO_ID[boundary_type]),
                "semantic_boundary_type": semantic_boundary_type,
                "safe_commit_reason": semantic_boundary_type,
                "left_phase_name": span["phase_name"],
                "right_phase_name": next_span["phase_name"],
                "left_depth": int(span.get("depth", 0)),
                "right_depth": int(next_span.get("depth", 0)),
            }
        )
    return boundary_edges


def _derive_boundary_positions(boundary_edges: List[Dict[str, object]]) -> List[int]:
    return sorted({int(edge["char_position"]) for edge in boundary_edges})


def label_python_phase_boundary_spans(code_text: str) -> Dict[str, object]:
    normalized_code = str(code_text or "")
    code_length = len(normalized_code)
    if not normalized_code.strip():
        return {
            "phase_label_vocab": list(PHASE_LABEL_VOCAB),
            "boundary_type_vocab": list(BOUNDARY_TYPE_VOCAB),
            "phase_spans": [],
            "boundary_positions": [],
            "boundary_edges": [],
        }

    try:
        tree = ast.parse(normalized_code)
    except SyntaxError:
        fallback_phase = "return_emit" if "return" in normalized_code else "init_state"
        return {
            "phase_label_vocab": list(PHASE_LABEL_VOCAB),
            "boundary_type_vocab": list(BOUNDARY_TYPE_VOCAB),
            "phase_spans": [
                {
                    "phase_id": PHASE_TO_ID[fallback_phase],
                    "phase_name": fallback_phase,
                    "char_start": 0,
                    "char_end": code_length,
                    "label_confidence": "ignore",
                }
            ],
            "boundary_positions": [],
            "boundary_edges": [],
        }

    line_offsets = _build_line_offsets(normalized_code)
    spans: List[Dict[str, object]] = []

    preamble_nodes: List[ast.stmt] = []
    trailing_nodes: List[ast.stmt] = []
    target_function: Optional[ast.stmt] = None

    top_level_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if top_level_functions:
        ranked_functions = []
        for order, node in enumerate(top_level_functions):
            char_start, char_end = _node_span(node, line_offsets, code_length)
            ranked_functions.append((char_end - char_start, order, node))
        target_function = max(ranked_functions, key=lambda item: (item[0], item[1]))[2]

    seen_target = False
    for node in tree.body:
        if node is target_function:
            seen_target = True
            continue
        if seen_target:
            trailing_nodes.append(node)
        else:
            preamble_nodes.append(node)

    for node in preamble_nodes:
        start, end = _node_span(node, line_offsets, code_length)
        _append_span(
            spans,
            phase_name="signature_or_doc",
            char_start=start,
            char_end=end,
            segment_kind="statement",
            depth=0,
            is_helper_definition=isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)),
            node_type=type(node).__name__,
        )

    if target_function is None:
        _walk_statements(
            list(tree.body),
            spans=spans,
            line_offsets=line_offsets,
            code_length=code_length,
            in_loop=False,
            depth=0,
        )
    else:
        function_start, _ = _node_span(target_function, line_offsets, code_length)
        if target_function.body:
            first_body = target_function.body[0]
            body_start = _char_index(
                line_offsets,
                getattr(first_body, "lineno", getattr(target_function, "lineno", 1)),
                getattr(first_body, "col_offset", 0),
            )
            _append_span(
                spans,
                phase_name="signature_or_doc",
                char_start=function_start,
                char_end=max(function_start + 1, body_start),
                confidence="gold",
                segment_kind="function_signature",
                depth=0,
                node_type=type(target_function).__name__,
            )
        body_statements = list(target_function.body)
        if body_statements and _is_docstring_expr(body_statements[0]):
            doc_start, doc_end = _node_span(body_statements[0], line_offsets, code_length)
            _append_span(
                spans,
                phase_name="signature_or_doc",
                char_start=doc_start,
                char_end=doc_end,
                confidence="gold",
                segment_kind="docstring",
                depth=1,
                node_type=type(body_statements[0]).__name__,
            )
            body_statements = body_statements[1:]
        _walk_statements(
            body_statements,
            spans=spans,
            line_offsets=line_offsets,
            code_length=code_length,
            in_loop=False,
            depth=1,
        )
        if trailing_nodes:
            _walk_statements(
                trailing_nodes,
                spans=spans,
                line_offsets=line_offsets,
                code_length=code_length,
                in_loop=False,
                depth=0,
            )

    compacted = _compact_spans(spans)
    boundary_edges = _derive_boundary_edges(code_length, spans)
    boundary_positions = _derive_boundary_positions(boundary_edges)
    return {
        "phase_label_vocab": list(PHASE_LABEL_VOCAB),
        "boundary_type_vocab": list(BOUNDARY_TYPE_VOCAB),
        "phase_spans": compacted,
        "boundary_positions": boundary_positions,
        "boundary_edges": boundary_edges,
    }
