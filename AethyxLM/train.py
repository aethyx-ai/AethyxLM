"""
AethyxLM - Main Training Entry Point (Kaggle Compatible)
"""

import json
import argparse
import subprocess
import platform
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model.gpt import GPT
from model.config import VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS
from dataset.dataset import AethyxDataset
from training.trainer import Trainer
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from utils.seed import set_seed


def load_config(config_path: str) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def get_git_commit_hash() -> str:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def get_cuda_version() -> str:
    """Get CUDA version if available."""
    if torch.cuda.is_available():
        return torch.version.cuda
    return "N/A"


def get_pytorch_version() -> str:
    """Get PyTorch version."""
    return torch.__version__


def get_cudnn_version() -> str:
    """Get cuDNN version."""
    if torch.cuda.is_available():
        return str(torch.backends.cudnn.version())
    return "N/A"


def get_gpu_name() -> str:
    """Get GPU name."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def save_run_config(config: dict, log_dir: Path, git_hash: str):
    """Save complete run configuration for reproducibility."""
    run_config = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": git_hash,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "cudnn_version": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "N/A",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seed": config.get('seed', 42),
        "model_config": config.get('model', {}),
        "training_config": config.get('training', {}),
        "data_config": config.get('data', {}),
        "checkpoint_config": config.get('checkpoint', {}),
        "tokenizer_config": config.get('tokenizer', {}),
    }
    
    run_config_path = Path("logs/run_config.json")
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(run_config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    print(f"Run config saved to: {run_config_path}")
    return run_config_path


def download_tinystories(data_dir: Path):
    """Download TinyStories dataset from Hugging Face."""
    try:
        from datasets import load_dataset
        print("Downloading TinyStories dataset from Hugging Face...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = [item["text"] for item in dataset]
        
        # Split train/val
        split_idx = int(0.9 * len(texts))
        train_texts = texts[:split_idx]
        val_texts = texts[split_idx:]
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / "train.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(train_texts))
        with open(data_dir / "val.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(val_texts))
        
        print(f"[OK] Dataset saved: {len(train_texts)} train, {len(val_texts)} val stories")
        return True
    except Exception as e:
        print(f"[WARN] Failed to download TinyStories: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train AethyxLM on Kaggle")
    parser.add_argument('--config', type=str, default='configs/train_config_kaggle.json',
                        help='Path to training config JSON')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"Random seed: {seed}")
    
    model_config = config['model']
    train_config = config['training']
    data_config = config['data']
    checkpoint_config = config['checkpoint']
    
    # Get git hash for reproducibility
    git_hash = get_git_commit_hash()
    
    # Print config
    print("=" * 60)
    print("AethyxLM Training Configuration (Kaggle)")
    print("=" * 60)
    print(f"Model: {model_config['num_layers']} layers, {model_config['embed_dim']} dim, {model_config['num_heads']} heads")
    print(f"Context: {model_config['context_length']}, Vocab: {model_config['vocab_size']}")
    print(f"LR: {train_config['learning_rate']}, WD: {train_config['weight_decay']}")
    print(f"Warmup: {train_config['warmup_steps']}, Max Steps: {train_config['max_steps']}")
    print(f"Batch: {data_config['batch_size']}, Grad Accum: {train_config['grad_accum_steps']}")
    print(f"AMP: {train_config['use_amp']}, Device: {args.device or 'auto'}")
    print(f"Git commit: {git_hash}")
    print("=" * 60)
    
    # Save run config for reproducibility
    save_run_config(config, Path("logs"), git_hash)
    
    # Device - accept any CUDA GPU
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Verify CUDA if requested
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available! Enable GPU in Kaggle settings (Accelerator -> GPU T4 x2)")
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {device_name} ({vram_gb:.1f} GB VRAM)")
        # Accept any CUDA GPU
        known_gpus = ['T4', 'L4', 'A100', 'V100', 'P100', 'K80', 'A10G']
        if not any(g in device_name for g in known_gpus):
            print(f"Note: GPU '{device_name}' not in common Kaggle types (T4, L4, A100, V100, P100, K80, A10G). Proceeding anyway...")
    
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = GPT()
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Download/prepare dataset only if local files are missing
    print("Preparing dataset...")
    train_file = Path(data_config['train_file'])
    val_file = Path(data_config.get('val_file', 'data/val.txt'))

    if train_file.exists() and val_file.exists():
        print(f"[OK] Local dataset found: {train_file} ({train_file.stat().st_size // 1_000_000} MB), "
              f"{val_file} ({val_file.stat().st_size // 1_000_000} MB) — skipping HF Hub download.")
    else:
        print("Local dataset not found. Attempting to download from Hugging Face Hub...")
        if not download_tinystories(Path("data")):
            if not train_file.exists():
                raise FileNotFoundError(
                    f"Training data not found at '{train_file}'. "
                    "Either run the dataset-preparation cell first, or provide a data/train.txt file."
                )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = AethyxDataset(
        text_path=data_config['train_file'],
        context_length=data_config['context_length'],
    )
    
    val_dataset = AethyxDataset(
        text_path=data_config['val_file'] if Path(data_config['val_file']).exists() else data_config['train_file'],
        context_length=data_config['context_length'],
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=data_config.get('shuffle', True),
        num_workers=data_config.get('num_workers', 2),
        drop_last=True,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 2),
        drop_last=True,
        pin_memory=True,
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        betas=tuple(train_config['betas']),
        eps=train_config['eps'],
        grad_clip=train_config['grad_clip'],
        warmup_steps=train_config['warmup_steps'],
        max_steps=train_config['max_steps'],
        min_lr_ratio=train_config['min_lr_ratio'],
        grad_accum_steps=train_config['grad_accum_steps'],
        use_amp=train_config['use_amp'] and device == 'cuda',
        checkpoint_dir=checkpoint_config['checkpoint_dir'],
        log_interval=checkpoint_config['log_interval'],
        eval_interval=checkpoint_config['eval_interval'],
        save_interval=checkpoint_config['save_interval'],
        generate_interval=train_config.get('generate_interval', 1000),
        device=device,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()