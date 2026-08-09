"""Synthetic long-context retrieval benchmark for decoder checkpoints."""

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class LongContextResult:
    context_length: int
    depth: float
    exact_matches: int
    trials: int

    @property
    def accuracy(self):
        return self.exact_matches / max(self.trials, 1)


def build_passkey_example(tokenizer, context_length, depth, passkey):
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between zero and one")
    needle = tokenizer.encode(f" The secret passkey is {passkey}. ")
    question = tokenizer.encode(" What is the secret passkey? The secret passkey is ")
    answer = tokenizer.encode(str(passkey))
    if context_length <= len(needle) + len(question):
        raise ValueError("context length is too short for the benchmark template")
    filler_pattern = tokenizer.encode(" Context filler information. ")
    filler_count = context_length - len(needle) - len(question)
    filler = (filler_pattern * (filler_count // len(filler_pattern) + 1))[:filler_count]
    insertion = int(len(filler) * depth)
    prompt = filler[:insertion] + needle + filler[insertion:] + question
    return prompt, answer


@torch.no_grad()
def evaluate_passkey_retrieval(
    model,
    tokenizer,
    context_lengths: Iterable[int],
    depths=(0.1, 0.5, 0.9),
    trials=5,
):
    """Greedy exact-match passkey retrieval across lengths and depths."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    results = []
    for context_length in context_lengths:
        for depth in depths:
            exact = 0
            for trial in range(trials):
                passkey = 100000 + 7919 * trial
                prompt, answer = build_passkey_example(
                    tokenizer, context_length, depth, passkey
                )
                sequence = torch.tensor([prompt], device=device)
                predicted = []
                for expected_token in answer:
                    logits = model(sequence[:, -model.context_length :])
                    next_token = int(logits[:, -1].argmax(-1).item())
                    predicted.append(next_token)
                    # Teacher forcing isolates retrieval from cascading generation errors.
                    expected = torch.tensor([[expected_token]], device=device)
                    sequence = torch.cat((sequence, expected), dim=1)
                exact += int(predicted == answer)
            results.append(LongContextResult(context_length, depth, exact, trials))
    if was_training:
        model.train()
    return results
