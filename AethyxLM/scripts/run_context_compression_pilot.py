"""Synthetic representation-density pilot for the AethyxLM context interface.

The task is exact key-value retrieval from a 64-record context. It deliberately
separates logical units from scalar storage and labels oracle decoders, avoiding
the false conclusion that fewer visual/graph "tokens" automatically mean less
information or compute.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.context_adapter import LatentContextAdapter


class LatentRetriever(nn.Module):
    def __init__(self, keys: int, values: int, embed_dim: int, latents: int):
        super().__init__()
        self.keys = keys
        self.values = values
        self.adapter = LatentContextAdapter(
            keys + values, embed_dim, num_latents=latents, num_heads=4, depth=2
        )
        self.query_embedding = nn.Embedding(keys, embed_dim)
        self.readout = nn.MultiheadAttention(
            embed_dim, 4, batch_first=True, bias=False
        )
        self.classifier = nn.Linear(embed_dim, values)
        self.register_buffer("key_features", torch.eye(keys), persistent=False)

    def forward(self, value_ids: torch.Tensor, query_ids: torch.Tensor):
        batch = value_ids.size(0)
        keys = self.key_features.unsqueeze(0).expand(batch, -1, -1)
        values = torch.nn.functional.one_hot(
            value_ids, num_classes=self.values
        ).float()
        features = torch.cat((keys, values), dim=-1)
        latents = self.adapter(features)
        query = self.query_embedding(query_ids).unsqueeze(1)
        retrieved, _ = self.readout(query, latents, latents, need_weights=False)
        return self.classifier(retrieved.squeeze(1))


def random_batch(batch_size: int, keys: int, values: int, device: str):
    records = torch.randint(values, (batch_size, keys), device=device)
    queries = torch.randint(keys, (batch_size,), device=device)
    answers = records.gather(1, queries[:, None]).squeeze(1)
    return records, queries, answers


@torch.no_grad()
def evaluate(model, args, batches=100):
    model.eval()
    correct = 0
    examples = 0
    for _ in range(batches):
        records, queries, answers = random_batch(
            args.batch_size, args.keys, args.values, "cuda"
        )
        predictions = model(records, queries).argmax(-1)
        correct += int((predictions == answers).sum())
        examples += len(answers)
    return correct / examples


def train_budget(latents: int, args):
    torch.manual_seed(args.seed + latents)
    model = LatentRetriever(args.keys, args.values, args.embed_dim, latents).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    final_loss = None
    for _ in range(args.steps):
        records, queries, answers = random_batch(
            args.batch_size, args.keys, args.values, "cuda"
        )
        loss = torch.nn.functional.cross_entropy(model(records, queries), answers)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    accuracy = evaluate(model, args, args.eval_batches)
    return {
        "representation": "learned latent resampler",
        "logical_units": latents,
        "unit_reduction": 1 - latents / args.keys,
        "stored_scalars": latents * args.embed_dim,
        "exact_retrieval_accuracy": accuracy,
        "accuracy_retention_vs_raw_oracle": accuracy,
        "final_training_loss": final_loss,
        "training_steps": args.steps,
        "evaluation_examples": args.eval_batches * args.batch_size,
    }


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the learned-latent pilot")
    budgets = [int(value) for value in args.latent_budgets.split(",")]
    raw_scalars = args.keys * (args.keys + args.values)
    indexed_scalars = args.keys * args.values
    patch_side = 2
    visual_units = args.keys // (patch_side**2)
    result = {
        "experiment": "synthetic exact key-value context retrieval",
        "warning": (
            "This isolates representation capacity on synthetic data. It does not "
            "validate natural-language, graph, image, privacy, or 70% product claims."
        ),
        "task": {"records": args.keys, "value_classes": args.values},
        "representations": {
            "raw_structured": {
                "decoder": "oracle lookup upper bound",
                "logical_units": args.keys,
                "unit_reduction": 0.0,
                "stored_scalars": raw_scalars,
                "exact_retrieval_accuracy": 1.0,
            },
            "indexed_graph": {
                "decoder": "oracle indexed-node lookup",
                "logical_units": args.keys,
                "unit_reduction": 0.0,
                "stored_scalars": indexed_scalars,
                "exact_retrieval_accuracy": 1.0,
                "note": "Fixed key order removes explicit key one-hots but not node count.",
            },
            "spatial_grid_patches": {
                "decoder": "oracle fixed-grid lookup; no vision encoder tested",
                "logical_units": visual_units,
                "unit_reduction": 1 - visual_units / args.keys,
                "stored_scalars": indexed_scalars,
                "exact_retrieval_accuracy": 1.0,
                "note": "Packing four cells per unit cuts unit count, not scalar information.",
            },
            "learned_latents": {},
        },
    }
    for budget in budgets:
        print(f"Training latent budget {budget}", flush=True)
        result["representations"]["learned_latents"][str(budget)] = train_budget(
            budget, args
        )
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result["representations"]["learned_latents"][str(budget)], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys", type=int, default=64)
    parser.add_argument("--values", type=int, default=16)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--latent-budgets", default="64,32,19,8")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="context_compression_pilot.json")
    arguments = parser.parse_args()
    if arguments.keys % 4:
        parser.error("--keys must be divisible by four for the grid-patch baseline")
    main(arguments)
