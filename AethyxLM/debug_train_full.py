#!/usr/bin/env python3
"""
Debug with ACTUAL training config (8-10M params).
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gpt import GPT
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from torch.cuda.amp import GradScaler, autocast

def print_mem(label):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    print(f"  [{label}] Allocated: {alloc:.3f} GB | Reserved: {reserv:.3f} GB")

def main():
    device = "cuda"
    print("=" * 60)
    print("Debug with ACTUAL TRAINING CONFIG")
    print("=" * 60)
    
    # ACTUAL config from train_config_kaggle.json (before my quick-test changes)
    config = {
        'vocab_size': 32000,      # Config says 32000
        'context_length': 128,    # Config says 128
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 8,
        'ffn_dim': 1024,
        'dropout': 0.1,
        'use_bias': True,
        'layer_norm_eps': 1e-5,
    }
    
    print_mem("Before model creation")
    model = GPT(config=config)
    print_mem("After model creation")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"Model size (FP32): {total_params * 4 / 1e6:.1f} MB")
    print(f"Model size (FP16): {total_params * 2 / 1e6:.1f} MB")
    print()
    
    model.to(device)
    print_mem("After model.to(device)")
    
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    print_mem("After optimizer creation")
    
    scaler = GradScaler(enabled=True)
    criterion = LanguageModelLoss()
    
    batch_size = 32  # Actual training batch size
    seq_len = 128    # Actual training seq_len
    vocab_size = 32000
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Input shape: {input_ids.shape} ({input_ids.numel() * 2 / 1e6:.1f} MB FP16)")
    print_mem("Before forward")
    
    model.train()
    with autocast(enabled=True):
        logits = model(input_ids)
        print(f"Logits shape: {logits.shape} ({logits.numel() * 2 / 1e6:.1f} MB FP16)")
        print_mem("After forward")
        loss = criterion(logits, targets)
        print(f"Loss: {loss.item():.4f}")
        print_mem("After loss")
    
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    print_mem("After backward")
    
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Grad norm: {grad_norm:.4f}")
    scaler.step(optimizer)
    scaler.update()
    print_mem("After optimizer.step()")
    
    optimizer.zero_grad(set_to_none=True)
    print_mem("After zero_grad")

if __name__ == "__main__":
    main()