"""
Training Loop for AethyxLM.
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from model.gpt import GPT
from model.config import CONTEXT_LENGTH, VOCAB_SIZE, NUM_LAYERS
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup


class Trainer:
    """
    Training loop with:
    - Gradient accumulation
    - Gradient clipping
    - Mixed precision (AMP)
    - Checkpoint saving/loading
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model: GPT,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        min_lr_ratio: float = 0.1,
        grad_accum_steps: int = 1,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss
        self.criterion = LanguageModelLoss()
        
        # Optimizer
        self.optimizer = create_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        
        # AMP
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(self, batch) -> float:
        """Single training step with gradient accumulation."""
        input_ids, targets = batch
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        # Forward with AMP
        if self.use_amp:
            with autocast():
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps
        else:
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            loss = loss / self.grad_accum_steps
        
        # Backward
        self.scaler.scale(loss).backward()
        
        return loss.item() * self.grad_accum_steps
    
    def optimizer_step(self):
        """Optimizer step with gradient clipping."""
        # Unscale gradients
        self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_dataloader:
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            with autocast(device_type=self.device, enabled=self.use_amp):
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": {
                "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "num_layers": NUM_LAYERS,
            }
        }
        
        # Latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        # Step checkpoint
        if self.step % self.save_interval == 0:
            step_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
            torch.save(checkpoint, step_path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Steps: {self.max_steps}, Warmup: {self.warmup_steps}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        
        self.model.train()
        running_loss = 0.0
        step_start_time = time.time()
        
        while self.step < self.max_steps:
            self.epoch += 1
            
            for batch in self.train_dataloader:
                if self.step >= self.max_steps:
                    break
                
                loss = self.train_step(batch)
                running_loss += loss
                
                # Optimizer step (after accumulation)
                if (self.step + 1) % self.grad_accum_steps == 0:
                    self.optimizer_step()
                
                self.step += 1
                
                # Logging
                if self.step % self.log_interval == 0:
                    elapsed = time.time() - step_start_time
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(
                        f"Step {self.step}/{self.max_steps} | "
                        f"Loss: {running_loss / self.log_interval:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Time: {elapsed:.2f}s"
                    )
                    running_loss = 0.0
                    step_start_time = time.time()
                
                # Evaluation
                if self.val_dataloader and self.step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    print(f"Validation Loss: {val_loss:.4f}")
                    
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    self.save_checkpoint(is_best=is_best)
                
                # Periodic checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        print("Training complete!")