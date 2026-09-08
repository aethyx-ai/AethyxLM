"""Supervised fine-tuning dataset with assistant-only loss masking."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from inference.prompt_contract import ROLE_TOKENS, serialize_messages


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    assistant_turns = 0
    previous_role = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("every message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in ROLE_TOKENS:
            raise ValueError(f"unsupported role: {role}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("message content must be non-empty text")
        if role == "system" and index != 0:
            raise ValueError("system role is only allowed as the first message")
        if role != "system" and previous_role not in {None, "system"} and role == previous_role:
            raise ValueError("user and assistant roles must alternate")
        if previous_role in {None, "system"} and role not in {"system", "user"}:
            raise ValueError("a conversation must begin with a user turn")
        if role != "system":
            previous_role = role
        assistant_turns += int(role == "assistant")
    if assistant_turns == 0:
        raise ValueError("conversation has no assistant response")
    if previous_role != "assistant":
        raise ValueError("conversation must end with an assistant response")


def encode_conversation(messages, tokenizer):
    """Encode role-tagged turns and supervise assistant content only."""
    validate_messages(messages)
    return serialize_messages(messages, tokenizer)


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, context_length: int = 512):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        if self.context_length < 8:
            raise ValueError("context_length is too small for role-formatted examples")
        self.examples = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    validate_messages(record["messages"])
                except (KeyError, ValueError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"Invalid SFT record at {self.path}:{line_number}"
                    ) from error
                self.examples.append((line_number, record["messages"]))
        if not self.examples:
            raise ValueError(f"No SFT examples found in {self.path}")
        if tokenizer.pad_id is None:
            raise ValueError("SFT requires the tokenizer's <PAD> token")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        line_number, messages = self.examples[index]
        input_ids, labels = encode_conversation(messages, self.tokenizer)
        required = self.context_length + 1
        if len(input_ids) > required:
            raise ValueError(
                f"SFT record at {self.path}:{line_number} has {len(input_ids)} tokens, "
                f"exceeding the {required}-token limit; rebuild the data bundle"
            )
        padding = required - len(input_ids)
        if padding:
            input_ids.extend([self.tokenizer.pad_id] * padding)
            labels.extend([-100] * padding)
        if not any(label != -100 for label in labels[1:]):
            raise ValueError(
                "Truncation removed every assistant target; shorten this example"
            )
        return (
            torch.tensor(input_ids[:-1], dtype=torch.long),
            torch.tensor(labels[1:], dtype=torch.long),
        )
