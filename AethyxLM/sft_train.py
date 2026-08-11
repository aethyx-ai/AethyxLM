"""Supervised instruction fine-tuning entry point for AethyxLM."""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer
from training.sft_dataset import SFTDataset
from training.trainer import Trainer
from utils.seed import set_seed


def resolve(path):
    path = Path(path).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.json")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))
    tokenizer = AethyxTokenizer(resolve(config["tokenizer"]))
    base_checkpoint = resolve(args.base_checkpoint or config["base_checkpoint"])
    base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    expected_hash = (
        base.get("config", {}).get("tokenizer", {}).get("sha256")
        or base.get("config", {}).get("tokenizer_sha256")
    )
    if expected_hash and expected_hash != tokenizer.sha256:
        raise RuntimeError("Base checkpoint tokenizer does not match tokenizer v2")
    state = base["model_state_dict"]
    model_config = GPT._infer_checkpoint_config(
        state, base.get("config", {}).get("model", {})
    )
    model = GPT(vocab_size=tokenizer.vocab_size, config=model_config)
    model.load_compatible_state_dict(state, strict=True)
    del base

    data = config["data"]
    train_dataset = SFTDataset(
        resolve(data["train"]), tokenizer, context_length=data["context_length"]
    )
    validation_dataset = SFTDataset(
        resolve(data["validation"]), tokenizer, context_length=data["context_length"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data["batch_size"],
        shuffle=True,
        num_workers=data.get("num_workers", 0),
        drop_last=True,
        pin_memory=args.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=data["batch_size"],
        shuffle=False,
        num_workers=data.get("num_workers", 0),
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
    )

    training = config["training"]
    checkpoint = config["checkpoint"]
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=validation_loader,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        grad_clip=training["grad_clip"],
        warmup_steps=training["warmup_steps"],
        max_steps=training["max_steps"],
        min_lr_ratio=training["min_lr_ratio"],
        grad_accum_steps=training["grad_accum_steps"],
        use_amp=training["use_amp"],
        amp_dtype=training.get("amp_dtype", "auto"),
        fused_optimizer=training.get("fused_optimizer", False),
        tokenizer_sha256=tokenizer.sha256,
        tokenizer_path=str(tokenizer.path),
        eval_batches=training.get("eval_batches"),
        checkpoint_dir=str(resolve(checkpoint["checkpoint_dir"])),
        milestone_dir=str(resolve(checkpoint["milestone_dir"])),
        milestone_interval=checkpoint.get("milestone_interval", 0),
        metrics_file=str(resolve(checkpoint["metrics_file"])),
        log_dir=str(resolve(checkpoint["log_dir"])),
        tensorboard_dir=str(resolve(checkpoint["tensorboard_dir"])),
        log_interval=checkpoint["log_interval"],
        eval_interval=checkpoint["eval_interval"],
        save_interval=checkpoint["save_interval"],
        generate_interval=0,
        device=args.device,
    )
    if args.resume:
        trainer.load_checkpoint(str(resolve(args.resume)))
    trainer.train()


if __name__ == "__main__":
    main()
