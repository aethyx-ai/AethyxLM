"""
AethyxLM - Main Training Entry Point (Kaggle Compatible)
"""

import json
import argparse
import subprocess
import platform
import os
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path for package imports (needed by torchrun)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from model.gpt import GPT
from model.config import VOCAB_SIZE, CONTEXT_LENGTH, NUM_LAYERS
from tokenizer.tokenizer import AethyxTokenizer
from dataset.dataset import (
    AethyxDataset,
    DistributedStridedSampler,
    MixedAethyxDataset,
    worker_init_fn,
)
from training.trainer import Trainer
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path: str) -> Path:
    """Resolve configured relative paths from the repository root."""
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def load_config(config_path: str) -> dict:
    """Load training configuration from JSON file."""
    with resolve_project_path(config_path).open('r', encoding='utf-8') as f:
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
    
    run_config_path = log_dir / "run_config.json"
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(run_config_path, 'w') as f:
        json.dump(run_config, f, indent=2)
    
    print(f"Run config saved to: {run_config_path}")
    return run_config_path


def setup_ddp():
    """Initialize Distributed Data Parallel."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Default to single GPU if not set
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        print(f"DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}, device={device}")
        return rank, world_size, local_rank, device, True
    else:
        return rank, world_size, local_rank, 'cuda' if torch.cuda.is_available() else 'cpu', False


def cleanup_ddp():
    """Cleanup DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


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
        
        data_dir = Path(data_dir)
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
    parser.add_argument('--config', type=str, default='configs/train_config_modern.json',
                        help='Path to training config JSON')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to train on (cuda/cpu)')
    parser.add_argument('--ddp', action='store_true',
                        help='Use Distributed Data Parallel (DDP)')
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank, device, is_ddp = setup_ddp()
    if is_ddp:
        print(f"Running with DDP: rank={rank}, world_size={world_size}")
    
    # Only print on rank 0
    is_main_process = (rank == 0)
    
    # Load config
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(seed + rank)  # Different seed per rank
    if is_main_process:
        print(f"Random seed: {seed}")
    
    model_config = config['model']
    train_config = config['training']
    data_config = config['data']
    if 'datasets_file' in data_config:
        registry_path = resolve_project_path(data_config['datasets_file'])
        registry = load_config(registry_path)
        data_config['datasets'] = [
            {"name": name, **dataset_config}
            for name, dataset_config in registry.items()
        ]
    checkpoint_config = config['checkpoint']
    tokenizer_config = config.get('tokenizer', {})
    tokenizer_path = resolve_project_path(
        tokenizer_config.get('tokenizer_file', 'tokenizer/tokenizer.json')
    )
    
    # Get git hash for reproducibility
    git_hash = get_git_commit_hash()
    
    # Print config (only on main process)
    if is_main_process:
        print("=" * 60)
        print("AethyxLM Training Configuration (Kaggle)")
        print("=" * 60)
        print(f"Model: {model_config['num_layers']} layers, {model_config['embed_dim']} dim, {model_config['num_heads']} heads")
        print(f"Context: {model_config['context_length']}, Vocab: {model_config['vocab_size']}")
        print(f"LR: {train_config['learning_rate']}, WD: {train_config['weight_decay']}")
        print(f"Warmup: {train_config['warmup_steps']}, Max Steps: {train_config['max_steps']}")
        print(f"Batch: {data_config['batch_size']}, Grad Accum: {train_config['grad_accum_steps']}")
        print(f"AMP: {train_config['use_amp']}, Device: {device}")
        print(f"Git commit: {git_hash}")
        print("=" * 60)
    
    # Save run config for reproducibility (only on main process)
    if is_main_process:
        save_run_config(config, resolve_project_path("logs"), git_hash)
    
    # Load tokenizer first to get actual vocab size (only on rank 0)
    if is_main_process:
        print("Loading tokenizer...")
        tokenizer = AethyxTokenizer(tokenizer_path)
        actual_vocab_size = tokenizer.vocab_size
        print(f"Tokenizer vocab size: {actual_vocab_size}")
        configured_vocab = tokenizer_config.get('vocab_size')
        if configured_vocab is not None and configured_vocab != actual_vocab_size:
            raise ValueError(
                f"Tokenizer config declares {configured_vocab} tokens, but "
                f"{tokenizer_path} contains {actual_vocab_size}."
            )
    else:
        actual_vocab_size = None
    
    # Broadcast vocab size to all ranks
    if is_ddp:
        vocab_tensor = torch.tensor([actual_vocab_size] if is_main_process else [0], device=device)
        dist.broadcast(vocab_tensor, src=0)
        actual_vocab_size = vocab_tensor.item()
    
    # Create model with actual vocab size
    if is_main_process:
        print("Creating model...")
    model = GPT(vocab_size=actual_vocab_size, config=model_config)
    model.to(device)

    if train_config.get('torch_compile', False):
        if not hasattr(model, 'compile'):
            raise RuntimeError("torch.compile requires a newer PyTorch version")
        if is_main_process:
            print(f"Compiling model ({train_config.get('compile_mode', 'default')})...")
        model.compile(mode=train_config.get('compile_mode', 'default'))
    
    # Wrap with DDP if using multi-GPU
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Count parameters (only on main process)
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Download/prepare dataset only if local files are missing
    if is_main_process:
        print("Preparing dataset...")
    
    # Check if new mixed dataset format is used
    if 'datasets' in data_config:
        # New mixed dataset format
        datasets_config = data_config['datasets']
        
        # Check if all .bin files exist
        all_exist = True
        for ds_config in datasets_config:
            train_bin = resolve_project_path(ds_config['train'])
            val_bin = resolve_project_path(ds_config.get('val', ds_config['train']))
            if not train_bin.exists() or (val_bin.exists() and val_bin.stat().st_size == 0):
                all_exist = False
                break
            if not val_bin.exists():
                # val file is optional, but if it exists it should be valid
                pass
        
        if not all_exist and is_main_process:
            print("Some binary datasets are missing; preparing them on rank 0.")
            for ds_config in datasets_config:
                name = ds_config.get('name', 'unknown')
                train_bin = resolve_project_path(ds_config['train'])
                val_bin = resolve_project_path(ds_config.get('val', ds_config['train']))
                if name == 'tinystories' and (
                    not train_bin.exists() or not val_bin.exists()
                ):
                    print("Preparing TinyStories...")
                    data_dir = train_bin.parent
                    if not download_tinystories(data_dir):
                        raise RuntimeError("Failed to download TinyStories")
                    for binary_path in {train_bin, val_bin}:
                        raw_path = binary_path.with_suffix('.txt')
                        if not raw_path.exists():
                            raise FileNotFoundError(
                                f"TinyStories source was not created at {raw_path}"
                            )
                        AethyxDataset(
                            raw_path,
                            context_length=data_config['context_length'],
                            tokenizer_path=tokenizer_path,
                        )
                elif name == 'fineweb_edu' and not train_bin.exists():
                    raise FileNotFoundError(
                        f"FineWeb-Edu training data not found at '{train_bin}'. "
                        "Run: python scripts/prepare_fineweb.py --target-gb 10"
                    )
        
        if is_ddp:
            dist.barrier()

        # Every rank validates rank 0's completed preparation before mmap.
        for ds_config in datasets_config:
            for key in ('train', 'val'):
                if key not in ds_config:
                    continue
                binary_path = resolve_project_path(ds_config[key])
                if not binary_path.exists() or binary_path.stat().st_size == 0:
                    raise FileNotFoundError(f"Prepared dataset is missing or empty: {binary_path}")
        
        # Tokenize on rank 0 ONLY, then all ranks load
        if is_main_process:
            print("Checking/Tokenizing datasets (rank 0)...")
            for ds_config in datasets_config:
                for f_key in ['train', 'val']:
                    if f_key in ds_config:
                        f = ds_config[f_key]
                        bin_f = resolve_project_path(f)
                        if bin_f.exists() and bin_f.stat().st_size == 0:
                            print(f"  Removing empty .bin file: {bin_f}")
                            bin_f.unlink()
        
        if is_ddp:
            dist.barrier()
        
        # Create mixed dataset
        if is_main_process:
            print("Loading mixed datasets...")
        
        # Build datasets config for MixedAethyxDataset
        mixed_datasets_config = []
        for ds_config in datasets_config:
            mixed_datasets_config.append({
                'train': str(resolve_project_path(ds_config['train'])),
                'val': str(resolve_project_path(ds_config.get('val', ds_config['train']))),
                'weight': ds_config.get('weight', 1.0),
            })
        
        train_dataset = MixedAethyxDataset(
            mixed_datasets_config,
            context_length=data_config['context_length'],
            tokenizer_path=tokenizer_path,
        )
        
        validation_configs = [
            {
                'train': str(resolve_project_path(item.get('val', item['train']))),
                'weight': item.get('weight', 1.0),
            }
            for item in datasets_config
        ]
        val_dataset = MixedAethyxDataset(
            validation_configs,
            context_length=data_config['context_length'],
            seed=seed + 1,
            tokenizer_path=tokenizer_path,
        )
        
        if is_main_process:
            print(f"Train samples (mixed): {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
        
    else:
        # Legacy single dataset format
        train_file = resolve_project_path(data_config['train_file'])
        val_file = resolve_project_path(data_config.get('val_file', 'data/val.txt'))

        if train_file.exists() and val_file.exists():
            if is_main_process:
                print(f"[OK] Local dataset found: {train_file} ({train_file.stat().st_size // 1_000_000} MB), "
                      f"{val_file} ({val_file.stat().st_size // 1_000_000} MB) — skipping HF Hub download.")
        else:
            if is_main_process:
                print("Local dataset not found. Attempting to download from Hugging Face Hub...")
            if not download_tinystories(Path("data")):
                if not train_file.exists():
                    raise FileNotFoundError(
                        f"Training data not found at '{train_file}'. "
                        "Either run the dataset-preparation cell first, or provide a data/train.txt file."
                    )
        
        # Synchronize all processes after dataset preparation
        if is_ddp:
            dist.barrier()
        
        # Tokenize on rank 0 ONLY, then all ranks load
        if is_main_process:
            print("Tokenizing datasets (rank 0)...")
            # Remove any existing empty .bin files to force re-tokenization
            for f in [train_file, val_file if val_file.exists() else train_file]:
                bin_f = Path(f).with_suffix('.bin')
                if bin_f.exists() and bin_f.stat().st_size == 0:
                    print(f"  Removing empty .bin file: {bin_f}")
                    bin_f.unlink()
            
            train_ds = AethyxDataset(
                text_path=train_file,
                context_length=data_config['context_length'],
                tokenizer_path=tokenizer_path,
            )
            print(f"Train tokens: {len(train_ds._data):,}")
            val_ds = AethyxDataset(
                text_path=val_file if val_file.exists() else train_file,
                context_length=data_config['context_length'],
                tokenizer_path=tokenizer_path,
            )
            print(f"Val tokens: {len(val_ds._data):,}")
            # Verify .bin files exist and non-empty
            import os
            for f in [train_file, val_file if val_file.exists() else train_file]:
                bin_f = str(Path(f).with_suffix('.bin'))
                size = os.path.getsize(bin_f)
                print(f"  {bin_f}: {size:,} bytes")
                if size == 0:
                    raise RuntimeError(f"Empty .bin file after tokenization: {bin_f}")
        
        if is_ddp:
            dist.barrier()
        
        # Now create datasets on all ranks (will use existing .bin files)
        if is_main_process:
            print("Loading datasets...")
        train_dataset = AethyxDataset(
            text_path=train_file,
            context_length=data_config['context_length'],
            tokenizer_path=tokenizer_path,
        )
        
        val_dataset = AethyxDataset(
            text_path=val_file if val_file.exists() else train_file,
            context_length=data_config['context_length'],
            tokenizer_path=tokenizer_path,
        )
        
        if is_main_process:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
    
    # Create distributed samplers for DDP
    if is_ddp:
        train_sampler = DistributedStridedSampler(
            train_dataset, num_replicas=world_size, rank=rank
        )
        val_sampler = DistributedStridedSampler(
            val_dataset, num_replicas=world_size, rank=rank
        )
        train_shuffle = False  # Sampler handles shuffling
        val_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = data_config.get('shuffle', True)
        val_shuffle = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=data_config.get('num_workers', 2),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn if data_config.get('num_workers', 0) > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=val_shuffle,
        sampler=val_sampler,
        num_workers=data_config.get('num_workers', 2),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn if data_config.get('num_workers', 0) > 0 else None,
    )
    
    # Create trainer (only on main process for logging/checkpointing)
    if is_main_process:
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
        use_amp=train_config['use_amp'] and device.startswith('cuda'),
        amp_dtype=train_config.get('amp_dtype', 'auto'),
        fused_optimizer=train_config.get('fused_optimizer', False),
        z_loss_coefficient=train_config.get('z_loss_coefficient', 0.0),
        context_schedule=train_config.get('context_schedule'),
        tokenizer_sha256=(tokenizer.sha256 if is_main_process else None),
        eval_batches=train_config.get('eval_batches'),
        tokenizer_path=str(tokenizer_path),
        checkpoint_dir=str(resolve_project_path(checkpoint_config['checkpoint_dir'])),
        log_dir=str(resolve_project_path(checkpoint_config.get('log_dir', 'logs'))),
        tensorboard_dir=str(
            resolve_project_path(checkpoint_config.get('tensorboard_dir', 'logs/tensorboard'))
        ),
        log_interval=checkpoint_config['log_interval'],
        eval_interval=train_config['eval_interval'],
        save_interval=checkpoint_config['save_interval'],
        milestone_interval=checkpoint_config.get('milestone_interval', 0),
        milestone_dir=str(
            resolve_project_path(
                checkpoint_config.get('milestone_dir', 'checkpoints/milestones')
            )
        ),
        metrics_file=str(
            resolve_project_path(
                checkpoint_config.get('metrics_file', 'logs/metrics.jsonl')
            )
        ),
        run_id=checkpoint_config.get('run_id'),
        checkpoint_backup=checkpoint_config.get('backup'),
        generate_interval=train_config.get('generate_interval', 1000),
        device=device,
    )
    
    # Resume from checkpoint if provided
    if args.resume:
        if is_main_process:
            print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(str(resolve_project_path(args.resume)))
    
    # Train
    if is_main_process:
        print("Starting training...")
    trainer.train()
    if is_main_process:
        print("Training complete!")
    
    # Cleanup DDP
    cleanup_ddp()


if __name__ == "__main__":
    main()
