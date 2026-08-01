#!/usr/bin/env python3
"""
Time each component of training step.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.gpt import GPT
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from dataset.dataset import AethyxDataset
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

def print_mem(label):
    alloc = torch.cuda.memory_allocated() / 1e9
    reserv = torch.cuda.memory_reserved() / 1e9
    print(f"  [{label}] Allocated: {alloc:.3f} GB | Reserved: {reserv:.3f} GB")

def main():
    device = "cuda"
    torch.cuda.synchronize()
    print("=" * 60)
    print("Timing training components")
    print("=" * 60)
    
    # Use ACTUAL current config (after my quick-test changes)
    config = {
        'vocab_size': 1908,       # Actual tokenizer vocab
        'context_length': 64,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
        'ffn_dim': 512,
        'dropout': 0.1,
        'use_bias': True,
        'layer_norm_eps': 1e-5,
    }
    
    print("Creating model...")
    t0 = time.time()
    model = GPT(config=config).to(device)
    torch.cuda.synchronize()
    print(f"  Model creation: {time.time() - t0:.3f}s")
    print_mem("After model")
    
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    scaler = GradScaler(enabled=True)
    criterion = LanguageModelLoss()
    
    # Create dataset
    print("Creating dataset...")
    t0 = time.time()
    train_dataset = AethyxDataset(
        text_path="data/train.txt",
        context_length=64,
    )
    torch.cuda.synchronize()
    print(f"  Dataset creation: {time.time() - t0:.3f}s")
    print_mem("After dataset")
    
    # Create dataloader with num_workers=0 (Windows default)
    print("Creating dataloader...")
    t0 = time.time()
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    torch.cuda.synchronize()
    print(f"  Dataloader creation: {time.time() - t0:.3f}s")
    
    # Get first batch
    print("Loading first batch...")
    t0 = time.time()
    batch = next(iter(train_loader))
    torch.cuda.synchronize()
    print(f"  First batch load: {time.time() - t0:.3f}s")
    print(f"  Batch shapes: {batch[0].shape}, {batch[1].shape}")
    print_mem("After first batch")
    
    # Move to device
    t0 = time.time()
    input_ids = batch[0].to(device, non_blocking=True)
    targets = batch[1].to(device, non_blocking=True)
    torch.cuda.synchronize()
    print(f"  .to(device): {time.time() - t0:.3f}s")
    print_mem("After to(device)")
    
    # Forward pass
    model.train()
    t0 = time.time()
    with autocast(enabled=True):
        logits = model(input_ids)
    torch.cuda.synchronize()
    print(f"  Forward: {time.time() - t0:.3f}s")
    print_mem("After forward")
    
    t0 = time.time()
    loss = criterion(logits, targets)
    torch.cuda.synchronize()
    print(f"  Loss: {time.time() - t0:.3f}s | Value: {loss.item():.4f}")
    print_mem("After loss")
    
    # Backward
    t0 = time.time()
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    print(f"  Backward: {time.time() - t0:.3f}s")
    print_mem("After backward")
    
    # Optimizer step
    t0 = time.time()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print(f"  Optimizer step: {time.time() - t0:.3f}s")
    print_mem("After optimizer")
    
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"  Zero grad: {time.time() - t0:.3f}s")
    print_mem("After zero_grad")
    
    # Now time a full step loop (without logging overhead)
    print("\n--- Timing 5 steps ---")
    for step in range(5):
        batch = next(iter(train_loader))
        input_ids = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        
        t0 = time.time()
        with autocast(enabled=True):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_time = time.time() - t0
        tok_per_sec = (8 * 64) / step_time
        print(f"  Step {step}: {step_time:.3f}s ({tok_per_sec:.0f} tok/s)")

if __name__ == "__main__":
    main()