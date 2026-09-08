"""Validated local tool calls for tightly restricted Python snippets."""

from __future__ import annotations

import ast
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Mapping


class ToolCallError(ValueError):
    pass


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
    if tool != "python" or not isinstance(arguments, Mapping):
        raise ToolCallError("unsupported tool or arguments")
    expected = {"code", "language"}
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
