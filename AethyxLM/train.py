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
from dataset.dataset import AethyxDataset, MixedAethyxDataset, worker_init_fn
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
    
    run_config_path = Path("logs/run_config.json")
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
    checkpoint_config = config['checkpoint']
    
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
        save_run_config(config, Path("logs"), git_hash)
    
    # Load tokenizer first to get actual vocab size (only on rank 0)
    if is_main_process:
        print("Loading tokenizer...")
        tokenizer = AethyxTokenizer()
        actual_vocab_size = tokenizer.vocab_size
        print(f"Tokenizer vocab size: {actual_vocab_size}")
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
            train_bin = Path(ds_config['train'])
            val_bin = Path(ds_config.get('val', ds_config['train']))
            if not train_bin.exists() or (val_bin.exists() and val_bin.stat().st_size == 0):
                all_exist = False
                break
            if not val_bin.exists():
                # val file is optional, but if it exists it should be valid
                pass
        
        if not all_exist:
            if is_main_process:
                print("Some .bin files missing. Will need to prepare datasets.")
            # Try to download/prepare each dataset
            for ds_config in datasets_config:
                name = ds_config.get('name', 'unknown')
                if name == 'tinystories':
                    if is_main_process:
                        print("Preparing TinyStories...")
                    if not download_tinystories(Path("data")):
                        raise RuntimeError("Failed to download TinyStories")
                elif name == 'fineweb_edu':
                    if is_main_process:
                        print("FineWeb-Edu should be prepared with scripts/prepare_fineweb.py")
                    # Note: FineWeb-Edu preparation is done separately via scripts/prepare_fineweb.py
                    train_bin = Path(ds_config['train'])
                    if not train_bin.exists():
                        raise FileNotFoundError(
                            f"FineWeb-Edu training data not found at '{train_bin}'. "
                            "Run: python scripts/prepare_fineweb.py --target-gb 10"
                        )
        
        if is_ddp:
            dist.barrier()
        
        # Tokenize on rank 0 ONLY, then all ranks load
        if is_main_process:
            print("Checking/Tokenizing datasets (rank 0)...")
            for ds_config in datasets_config:
                for f_key in ['train', 'val']:
                    if f_key in ds_config:
                        f = ds_config[f_key]
                        bin_f = Path(f)
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
                'train': ds_config['train'],
                'val': ds_config.get('val', ds_config['train']),
                'weight': ds_config.get('weight', 1.0),
            })
        
        train_dataset = MixedAethyxDataset(
            mixed_datasets_config,
            context_length=data_config['context_length'],
        )
        
        # For validation, we can use a small mixed dataset or just the first dataset's val
        # For simplicity, use the first dataset's val file
        val_dataset = AethyxDataset(
            text_path=datasets_config[0].get('val', datasets_config[0]['train']),
            context_length=data_config['context_length'],
        )
        
        if is_main_process:
            print(f"Train samples (mixed): {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
        
    else:
        # Legacy single dataset format
        train_file = Path(data_config['train_file'])
        val_file = Path(data_config.get('val_file', 'data/val.txt'))

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
            for f in [data_config['train_file'], data_config['val_file'] if Path(data_config['val_file']).exists() else data_config['train_file']]:
                bin_f = Path(f).with_suffix('.bin')
                if bin_f.exists() and bin_f.stat().st_size == 0:
                    print(f"  Removing empty .bin file: {bin_f}")
                    bin_f.unlink()
            
            train_ds = AethyxDataset(
                text_path=data_config['train_file'],
                context_length=data_config['context_length'],
            )
            print(f"Train tokens: {len(train_ds._data):,}")
            val_ds = AethyxDataset(
                text_path=data_config['val_file'] if Path(data_config['val_file']).exists() else data_config['train_file'],
                context_length=data_config['context_length'],
            )
            print(f"Val tokens: {len(val_ds._data):,}")
            # Verify .bin files exist and non-empty
            import os
            for f in [data_config['train_file'], data_config['val_file'] if Path(data_config['val_file']).exists() else data_config['train_file']]:
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
            text_path=data_config['train_file'],
            context_length=data_config['context_length'],
        )
        
        val_dataset = AethyxDataset(
            text_path=data_config['val_file'] if Path(data_config['val_file']).exists() else data_config['train_file'],
            context_length=data_config['context_length'],
        )
        
        if is_main_process:
            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")
    
    # Create distributed samplers for DDP
    if is_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
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
        checkpoint_dir=str(resolve_project_path(checkpoint_config['checkpoint_dir'])),
        log_interval=checkpoint_config['log_interval'],
        eval_interval=train_config['eval_interval'],
        save_interval=checkpoint_config['save_interval'],
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
