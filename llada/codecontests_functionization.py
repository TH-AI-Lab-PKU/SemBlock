from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from typing import Optional


SYNTHETIC_SIGNATURE = "def solve_io(input_data: str) -> str:"


@dataclass(frozen=True)
class FunctionizationResult:
    converted: bool
    function_text: str
    synthetic_signature: str
    wrapper_mode: str = "redirected_stdio"
    notes: Optional[str] = None


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    left = test.left
    comparator = test.comparators[0]
    if not isinstance(left, ast.Name) or left.id != "__name__":
        return False
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _strip_main_guard(code_text: str) -> str:
    try:
        module = ast.parse(code_text)
    except SyntaxError:
        return code_text.strip("\n")

    filtered_body = [node for node in module.body if not _is_main_guard(node)]
    module.body = filtered_body
    try:
        stripped = ast.unparse(module)
    except Exception:
        return code_text.strip("\n")
    return stripped.strip("\n")


def functionize_python_program(code_text: str) -> FunctionizationResult:
    stripped_code = _strip_main_guard(str(code_text or ""))
    indented_body = textwrap.indent(stripped_code or "pass", " " * 8)
    function_text = (
        f"{SYNTHETIC_SIGNATURE}\n"
        "    import io\n"
        "    import sys\n"
        "    _original_stdin = sys.stdin\n"
        "    _original_stdout = sys.stdout\n"
        "    _stdout_capture = io.StringIO()\n"
        "    sys.stdin = io.StringIO(input_data)\n"
        "    sys.stdout = _stdout_capture\n"
        "    try:\n"
        f"{indented_body}\n"
        "    finally:\n"
        "        sys.stdin = _original_stdin\n"
        "        sys.stdout = _original_stdout\n"
        "    return _stdout_capture.getvalue()\n"
    )
    return FunctionizationResult(
        converted=True,
        function_text=function_text,
        synthetic_signature=SYNTHETIC_SIGNATURE,
    )
