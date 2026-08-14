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


def newest_sft_checkpoint(checkpoint_dir):
    """Return the newest resumable SFT checkpoint, if one exists."""
    checkpoint_dir = resolve(checkpoint_dir)
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.is_file():
        return latest
    numbered = []
    for path in checkpoint_dir.glob("checkpoint_step_*.pt"):
        try:
            numbered.append((int(path.stem.rsplit("_", 1)[1]), path))
        except ValueError:
            continue
    return max(numbered, default=(None, None))[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_config.json")
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--resume")
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume checkpoint_latest.pt when present; otherwise start from the base model",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild the streamed SFT data bundle before training",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))
    data = config["data"]
    train_path = resolve(data["train"])
    validation_path = resolve(data["validation"])
    if args.force_prepare or not (train_path.is_file() and validation_path.is_file()):
        from scripts.prepare_sft_bundle import prepare_from_config

        prepare_from_config(args.config, force=args.force_prepare)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot access a GPU")
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

    train_dataset = SFTDataset(
        train_path, tokenizer, context_length=data["context_length"]
    )
    validation_dataset = SFTDataset(
        validation_path, tokenizer, context_length=data["context_length"]
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
    resume_path = resolve(args.resume) if args.resume else None
    if args.auto_resume and resume_path is None:
        resume_path = newest_sft_checkpoint(checkpoint["checkpoint_dir"])
    if resume_path:
        print(f"[RESUME] Loading {resume_path}")
        trainer.load_checkpoint(str(resume_path))
    else:
        effective_batch = data["batch_size"] * training["grad_accum_steps"]
        print(
            f"[SFT] Starting from {base_checkpoint.name}; "
            f"effective batch={effective_batch}, context={data['context_length']}"
        )
    trainer.train()


if __name__ == "__main__":
    main()
