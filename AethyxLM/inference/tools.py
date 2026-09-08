"""Validated local tool calls for arithmetic and tightly restricted Python snippets."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any, Mapping


class ToolCallError(ValueError):
    pass


_ARITHMETIC_WRAPPER = re.compile(
    r"^\s*(?:what\s+is|what's|calculate|compute|evaluate|solve)"
    r"(?:\s+the\s+(?:value|result)\s+of)?\s*:?\s*(?P<expression>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"^\d{1,4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,4}$")
_IDENTIFIER_LIKE = re.compile(r"^\d{3,}(?:\s*-\s*\d{2,})+$")
_NUMERIC_MULTIPLICATION = re.compile(
    r"(?<=\d)\s*[xX×]\s*(?=[+-]?(?:\d|\.\d))"
)


def extract_arithmetic_expression(text: str) -> str | None:
    """Extract an unambiguous standalone arithmetic request.

    This intentionally accepts only a complete expression or a small allowlist of
    question wrappers. Prose containing arithmetic, units, dates, identifiers,
    variables, and URLs falls through to normal model generation.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    if not candidate or len(candidate) > 256 or "\n" in candidate:
        return None

    wrapped = _ARITHMETIC_WRAPPER.fullmatch(candidate)
    expression = wrapped.group("expression").strip() if wrapped else candidate
    if expression.endswith("?"):
        expression = expression[:-1].rstrip()
    if not expression or _DATE_LIKE.fullmatch(expression) or _IDENTIFIER_LIKE.fullmatch(expression):
        return None

    # Treat x as multiplication only when both adjacent operands are numeric.
    expression = _NUMERIC_MULTIPLICATION.sub("*", expression)
    if re.search(r"[xX×]", expression):
        return None
    if not re.fullmatch(r"[\d\s.+\-*/%()]+", expression):
        return None

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if not any(isinstance(node, ast.BinOp) for node in ast.walk(tree)):
        return None
    return expression


def route_arithmetic_question(
    text: str, controller: "ToolController"
) -> dict[str, Any] | None:
    """Execute a clear arithmetic request, or return ``None`` to fall through."""
    expression = extract_arithmetic_expression(text)
    if expression is None:
        return None
    result = controller.execute(
        {"tool": "calculator", "arguments": {"expression": expression}}
    )
    return result if result.get("ok") else None


def _decimal(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolCallError("calculator accepts numeric literals only")
    return Decimal(str(value))


def calculate(expression: str) -> str:
    """Evaluate a bounded arithmetic grammar without eval or arbitrary calls."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 256:
        raise ToolCallError("expression must contain 1-256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ToolCallError("invalid arithmetic expression") from error

    def visit(node, depth=0):
        if depth > 24:
            raise ToolCallError("expression is too deeply nested")
        if isinstance(node, ast.Expression):
            return visit(node.body, depth + 1)
        if isinstance(node, ast.Constant):
            return _decimal(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand, depth + 1)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left, depth + 1), visit(node.right, depth + 1)
            if isinstance(node.op, ast.Add): result = left + right
            elif isinstance(node.op, ast.Sub): result = left - right
            elif isinstance(node.op, ast.Mult): result = left * right
            elif isinstance(node.op, ast.Div): result = left / right
            elif isinstance(node.op, ast.FloorDiv): result = left // right
            elif isinstance(node.op, ast.Mod): result = left % right
            elif isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(right) > 12:
                    raise ToolCallError("exponent must be an integer from -12 to 12")
                result = left ** int(right)
            else:
                raise ToolCallError("unsupported calculator operator")
            if not result.is_finite() or abs(result) > Decimal("1e100"):
                raise ToolCallError("calculator result is outside the allowed range")
            return result
        raise ToolCallError("calculator permits only numbers and + - * / // % **")

    try:
        result = visit(tree)
    except (DivisionByZero, InvalidOperation, ZeroDivisionError) as error:
        raise ToolCallError("invalid arithmetic operation") from error
    rendered = format(result.normalize(), "f")
    return "0" if Decimal(rendered) == 0 else rendered


_FORBIDDEN_CODE_NODES = (
    ast.Import, ast.ImportFrom, ast.Attribute, ast.With, ast.AsyncWith, ast.Try,
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Global,
    ast.Nonlocal, ast.Delete, ast.Yield, ast.YieldFrom, ast.Await, ast.Raise,
)
_SAFE_CALLS = {"print", "range", "len", "min", "max", "sum", "abs", "round", "sorted", "enumerate", "zip"}


def _validate_restricted_python(code: str):
    if not isinstance(code, str) or not code.strip() or len(code) > 8000:
        raise ToolCallError("code must contain 1-8000 characters")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise ToolCallError("invalid Python syntax") from error
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_CODE_NODES):
            raise ToolCallError(f"restricted Python rejects {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise ToolCallError("private names are not permitted")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_CALLS:
                raise ToolCallError("only approved pure built-ins may be called")


_PYTHON_WRAPPER = r'''
import sys
limit = int(sys.argv[1])
class LimitedWriter:
    def __init__(self): self.size = 0
    def write(self, value):
        raw = str(value)
        self.size += len(raw.encode("utf-8"))
        if self.size > limit: raise RuntimeError("output limit exceeded")
        return sys.__stdout__.write(raw)
    def flush(self): return sys.__stdout__.flush()
safe = {name: getattr(__builtins__, name) for name in ("range","len","min","max","sum","abs","round","sorted","enumerate","zip")}
writer = LimitedWriter()
safe["print"] = lambda *a, **k: print(*a, file=writer, **{x:y for x,y in k.items() if x in ("sep","end","flush")})
source = sys.stdin.read()
exec(compile(source, "<tool>", "exec"), {"__builtins__": safe}, {})
'''


@dataclass(frozen=True)
class RestrictedPythonBackend:
    timeout_seconds: float = 2.0
    max_output_bytes: int = 16384
    memory_limit_mb: int = 256

    def run(self, code: str) -> dict[str, Any]:
        _validate_restricted_python(code)
        with tempfile.TemporaryDirectory(prefix="aethyx-tool-") as directory:
            environment = {
                "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TMP": directory,
                "TEMP": directory,
            }
            for key in ("SYSTEMROOT", "WINDIR"):
                if key in os.environ:
                    environment[key] = os.environ[key]
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            preexec_fn = None
            if os.name == "posix":
                def apply_posix_limits():
                    import resource

                    memory = self.memory_limit_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
                    resource.setrlimit(
                        resource.RLIMIT_CPU,
                        (max(1, math.ceil(self.timeout_seconds)), max(1, math.ceil(self.timeout_seconds))),
                    )
                    if hasattr(resource, "RLIMIT_NPROC"):
                        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

                preexec_fn = apply_posix_limits
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _PYTHON_WRAPPER, str(self.max_output_bytes)],
                cwd=directory,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=creationflags,
                start_new_session=os.name == "posix",
                preexec_fn=preexec_fn,
            )
            try:
                stdout, stderr = process.communicate(code, timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    # The AST allowlist cannot import or launch child processes.
                    process.kill()
                process.communicate()
                return {
                    "ok": False,
                    "error": "timeout",
                    "timeout_seconds": self.timeout_seconds,
                    "memory_limit_enforced": os.name == "posix",
                }
        if len(stdout.encode("utf-8")) > self.max_output_bytes:
            return {"ok": False, "error": "output_limit", "memory_limit_enforced": os.name == "posix"}
        if process.returncode:
            error = "output_limit" if "output limit exceeded" in stderr else "execution_error"
            return {"ok": False, "error": error, "memory_limit_enforced": os.name == "posix"}
        return {
            "ok": True,
            "stdout": stdout,
            "language": "python-restricted",
            "memory_limit_enforced": os.name == "posix",
        }


def parse_tool_call(payload: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ToolCallError("tool call must be one JSON object") from error
    if not isinstance(payload, Mapping) or set(payload) != {"tool", "arguments"}:
        raise ToolCallError("tool call requires exactly tool and arguments")
    tool, arguments = payload["tool"], payload["arguments"]
    if tool not in {"calculator", "python"} or not isinstance(arguments, Mapping):
        raise ToolCallError("unsupported tool or arguments")
    expected = {"expression"} if tool == "calculator" else {"code", "language"}
    if set(arguments) != expected:
        raise ToolCallError(f"{tool} arguments must be exactly {sorted(expected)}")
    if tool == "python" and arguments["language"] != "python-restricted":
        raise ToolCallError("only python-restricted is supported")
    return {"tool": tool, "arguments": dict(arguments)}


class ToolController:
    def __init__(self, *, allow_code: bool = False, code_backend=None, malformed_retries: int = 1):
        self.allow_code = allow_code
        self.code_backend = code_backend or RestrictedPythonBackend()
        self.malformed_retries = max(0, int(malformed_retries))

    def execute(self, payload, malformed_attempt: int = 0) -> dict[str, Any]:
        try:
            call = parse_tool_call(payload)
        except ToolCallError as error:
            return {
                "ok": False,
                "error": "invalid_tool_call",
                "message": str(error),
                "retry_allowed": malformed_attempt < self.malformed_retries,
            }
        if call["tool"] == "calculator":
            try:
                return {"ok": True, "tool": "calculator", "result": calculate(call["arguments"]["expression"])}
            except ToolCallError as error:
                return {"ok": False, "tool": "calculator", "error": "invalid_expression", "message": str(error)}
        if not self.allow_code:
            return {"ok": False, "tool": "python", "error": "disabled"}
        try:
            result = self.code_backend.run(call["arguments"]["code"])
        except ToolCallError as error:
            return {
                "ok": False,
                "tool": "python",
                "error": "invalid_code",
                "message": str(error),
            }
        return {"tool": "python", **result}

    def execute_model_output(self, text: str, malformed_attempt: int = 0):
        """Execute a model-emitted ``<TOOL_CALL>`` JSON payload."""
        marker = "<TOOL_CALL>"
        if marker not in text:
            return None
        prefix, payload = text.split(marker, 1)
        result = self.execute(payload.strip(), malformed_attempt=malformed_attempt)
        return {"assistant_text": prefix.rstrip(), "result": result}

    def execute_with_repair_candidates(self, payloads):
        """Try at most one initial call plus the configured malformed retries."""
        last = None
        for attempt, payload in enumerate(payloads):
            if attempt > self.malformed_retries:
                break
            last = self.execute(payload, malformed_attempt=attempt)
            if last.get("error") != "invalid_tool_call":
                return last
        return last or {
            "ok": False,
            "error": "invalid_tool_call",
            "message": "no tool-call candidate was supplied",
            "retry_allowed": False,
        }
