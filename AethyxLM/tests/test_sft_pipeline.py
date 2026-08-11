import json

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT
from scripts.prepare_sft_data import normalize_record, stable_key
from tokenizer.tokenizer import AethyxTokenizer
from training.sft_dataset import SFTDataset, encode_conversation
from training.trainer import Trainer


def test_sft_normalizes_sharegpt_and_masks_non_assistant_tokens(tmp_path):
    record = normalize_record(
        {
            "conversations": [
                {"from": "human", "value": "What is two plus two?"},
                {"from": "gpt", "value": "It is four."},
            ]
        }
    )
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert stable_key(record) == stable_key(record)

    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    tokenizer = AethyxTokenizer()
    dataset = SFTDataset(path, tokenizer, context_length=64)
    inputs, targets = dataset[0]

    assert inputs.shape == targets.shape == (64,)
    assert torch.any(targets == -100)
    assert torch.any(targets != -100)
    ids, labels = encode_conversation(record["messages"], tokenizer)
    assert len(ids) == len(labels)
    assert tokenizer.token_to_id("<USER>") is not None
    assert tokenizer.token_to_id("<ASSISTANT>") is not None


def test_sft_dataset_runs_one_assistant_masked_optimizer_step(tmp_path):
    tokenizer = AethyxTokenizer()
    records = [
        {
            "messages": [
                {"role": "user", "content": f"Question {index}?"},
                {"role": "assistant", "content": f"Answer {index}."},
            ]
        }
        for index in range(4)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    dataset = SFTDataset(path, tokenizer, context_length=32)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        config={
            "vocab_size": tokenizer.vocab_size,
            "context_length": 32,
            "embed_dim": 32,
            "num_heads": 4,
            "num_kv_heads": 2,
            "num_layers": 1,
            "ffn_dim": 64,
            "dropout": 0.0,
            "normalization": "rmsnorm",
            "position_encoding": "rope",
            "ffn_type": "swiglu",
            "fused_qkv": True,
            "use_sdpa": True,
        },
    )
    trainer = Trainer(
        model,
        loader,
        max_steps=1,
        warmup_steps=0,
        use_amp=False,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_interval=1,
        eval_interval=1,
        generate_interval=0,
        milestone_interval=1,
        milestone_dir=str(tmp_path / "milestones"),
    )

    trainer.train()

    assert trainer.step == 1
    assert (tmp_path / "checkpoints/checkpoint_step_1.pt").is_file()
    assert (tmp_path / "milestones/checkpoint_step_1.pt").is_file()
