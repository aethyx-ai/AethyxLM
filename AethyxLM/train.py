"""
AethyxLM - Main Training Entry Point
"""

import json
import argparse
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Train AethyxLM")
    parser.add_argument('--config', type=str, default='configs/train_config.json',
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
    
    # Print config
    print("=" * 60)
    print("AethyxLM Training Configuration")
    print("=" * 60)
    print(f"Model: {model_config['num_layers']} layers, {model_config['embed_dim']} dim, {model_config['num_heads']} heads")
    print(f"Context: {model_config['context_length']}, Vocab: {model_config['vocab_size']}")
    print(f"LR: {train_config['learning_rate']}, WD: {train_config['weight_decay']}")
    print(f"Warmup: {train_config['warmup_steps']}, Max Steps: {train_config['max_steps']}")
    print(f"Batch: {data_config['batch_size']}, Grad Accum: {train_config['grad_accum_steps']}")
    print(f"AMP: {train_config['use_amp']}, Device: {args.device or 'auto'}")
    print("=" * 60)
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        num_workers=data_config.get('num_workers', 0),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 0),
        drop_last=True,
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