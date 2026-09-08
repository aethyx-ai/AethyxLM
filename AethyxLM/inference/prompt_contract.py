"""Versioned prompt contracts shared by SFT, evaluation, and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence
import re


BASE_CONTRACT = "base-v1"
LEGACY_CHAT_CONTRACT = "legacy-chat-v1"
SFT_CHAT_CONTRACT = "aethyx-sft-v1"
PROMPT_CONTRACTS = (BASE_CONTRACT, LEGACY_CHAT_CONTRACT, SFT_CHAT_CONTRACT)

ROLE_TOKENS = {
    "system": "<SYSTEM>",
    "user": "<USER>",
    "assistant": "<ASSISTANT>",
}


@dataclass(frozen=True)
class PromptContract:
    name: str
    stop_strings: tuple[str, ...]

    def format_system(self, text: str) -> str:
        if not text.strip():
            return ""
        if self.name == SFT_CHAT_CONTRACT:
            return f"{ROLE_TOKENS['system']}\n{text.strip()}\n"
        if self.name == LEGACY_CHAT_CONTRACT:
            return f"System: {text.strip()}\n"
        raise ValueError("base completion mode does not use a system prompt")

    def format_user_turn(self, text: str, history: str = "") -> str:
        if self.name == BASE_CONTRACT:
            return text
        if self.name == LEGACY_CHAT_CONTRACT:
            return f"{history}User: {text}\nAethyx:"
        if self.name == SFT_CHAT_CONTRACT:
            return f"{history}{ROLE_TOKENS['user']}\n{text.strip()}\n{ROLE_TOKENS['assistant']}\n"
        raise ValueError(f"Unsupported prompt contract: {self.name}")

    def append_assistant_turn(self, prompt: str, response: str) -> str:
        if self.name == LEGACY_CHAT_CONTRACT:
            return f"{prompt} {response.strip()}\n"
        if self.name == SFT_CHAT_CONTRACT:
            return f"{prompt}{response.strip()}\n"
        return prompt + response

    def format_tool_result(self, tool: str, result: Mapping[str, object]) -> str:
        import json

        payload = json.dumps(result, ensure_ascii=False, sort_keys=True)
        if self.name == SFT_CHAT_CONTRACT:
            return f"<TOOL_RESULT name={json.dumps(tool)}>\n{payload}\n"
        return f"\nTool[{tool}] result: {payload}\n"

    def continue_after_tool(self, prompt: str, model_output: str, tool_result: str) -> str:
        if self.name == SFT_CHAT_CONTRACT:
            return f"{prompt}{model_output.strip()}\n{tool_result}{ROLE_TOKENS['assistant']}\n"
        if self.name == LEGACY_CHAT_CONTRACT:
            return f"{prompt} {model_output.strip()}\n{tool_result}Aethyx:"
        return f"{prompt}{model_output}\n{tool_result}"


def get_prompt_contract(name: str) -> PromptContract:
    if name == BASE_CONTRACT:
        return PromptContract(name, ())
    if name == LEGACY_CHAT_CONTRACT:
        return PromptContract(name, ("\nUser:", "User:", "\nAethyx:", "Aethyx:"))
    if name == SFT_CHAT_CONTRACT:
        return PromptContract(
            name,
            (ROLE_TOKENS["user"], ROLE_TOKENS["system"]),
        )
    raise ValueError(
        f"Unknown prompt contract {name!r}; choose one of {', '.join(PROMPT_CONTRACTS)}"
    )


def checkpoint_prompt_contract(checkpoint_config: Mapping[str, object]) -> str | None:
    inference = checkpoint_config.get("inference", {})
    if isinstance(inference, Mapping):
        value = inference.get("prompt_contract")
        if isinstance(value, str) and value:
            return value
    return None


def resolve_inference_mode(
    requested: str, checkpoint_config: Mapping[str, object] | None = None
) -> str:
    if requested in {"base", "chat"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported inference mode: {requested}")
    declared = checkpoint_prompt_contract(checkpoint_config or {})
    return "chat" if declared and declared != BASE_CONTRACT else "base"


def resolve_prompt_contract(
    mode: str,
    checkpoint_config: Mapping[str, object] | None = None,
    requested: str | None = None,
) -> PromptContract:
    """Resolve explicitly; missing checkpoint metadata never implies chat tuning."""
    if mode == "base":
        return get_prompt_contract(BASE_CONTRACT)
    if mode != "chat":
        raise ValueError(f"Unsupported inference mode: {mode}")
    name = requested or checkpoint_prompt_contract(checkpoint_config or {})
    if name in {None, BASE_CONTRACT}:
        raise ValueError(
            "This checkpoint does not declare a chat prompt contract. "
            "Use --mode base, or explicitly pass --prompt-contract legacy-chat-v1 "
            "or aethyx-sft-v1 only when you know how the checkpoint was tuned."
        )
    return get_prompt_contract(name)


def serialize_messages(messages: Sequence[Mapping[str, str]], tokenizer=None):
    """Serialize the canonical SFT contract, optionally returning labels."""
    text_parts: list[str] = []
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = message["role"]
        prefix = f"{ROLE_TOKENS[role]}\n"
        content = message["content"].strip() + "\n"
        text_parts.extend((prefix, content))
        if tokenizer is None:
            continue
        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        input_ids.extend(content_ids)
        labels.extend(content_ids if role == "assistant" else [-100] * len(content_ids))
        if role == "assistant" and tokenizer.eos_id is not None:
            input_ids.append(tokenizer.eos_id)
            labels.append(tokenizer.eos_id)
    return (input_ids, labels) if tokenizer is not None else "".join(text_parts)


def trim_serialized_history(
    history: str, tokenizer, max_tokens: int, contract_name: str
) -> tuple[str, int]:
    """Drop complete oldest turns and preserve an initial SFT system instruction."""
    original = tokenizer.encode(history)
    if len(original) <= max_tokens:
        return history, 0
    if contract_name == SFT_CHAT_CONTRACT:
        chunks = [part for part in re.split(r"(?=<(?:SYSTEM|USER)>\n)", history) if part]
        preserved = chunks[:1] if chunks and chunks[0].startswith("<SYSTEM>\n") else []
        candidates = chunks[len(preserved) :]
    elif contract_name == LEGACY_CHAT_CONTRACT:
        preserved = []
        candidates = [part for part in re.split(r"(?=User:)", history) if part]
    else:
        return "", len(original)
    kept = list(preserved)
    for chunk in reversed(candidates):
        candidate = "".join((*preserved, chunk, *kept[len(preserved) :]))
        if len(tokenizer.encode(candidate)) > max_tokens:
            continue
        kept.insert(len(preserved), chunk)
    rendered = "".join(kept)
    return rendered, len(original) - len(tokenizer.encode(rendered))


def critical_prefix_token_count(prompt: str, tokenizer, contract_name: str) -> int:
    """Return the initial system segment that context budgeting must retain."""
    if contract_name != SFT_CHAT_CONTRACT or not prompt.startswith("<SYSTEM>\n"):
        return 0
    boundary = prompt.find("<USER>\n")
    prefix = prompt if boundary < 0 else prompt[:boundary]
    return len(tokenizer.encode(prefix))
