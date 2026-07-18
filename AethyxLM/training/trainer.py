"""
Training Loop for AethyxLM.
"""

import os
import time
import json
import signal
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

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
    - TensorBoard logging
    - Sample generation during training
    - Robust checkpoint synchronization
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
        generate_interval: int = 1000,
        device: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        log_dir: str = "logs",
        seed: int = 42,
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
        self.generate_interval = generate_interval
        
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
        self.seed = seed
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.tensorboard_dir = Path("logs/tensorboard")
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(Path("logs/tensorboard")))
        
        # Generated samples log
        self.samples_log = Path("logs/generated_samples.txt")
        
        # Sync control
        self._sync_in_progress = False
        
        # Register signal handlers for graceful shutdown
        self._register_signal_handlers()
    
    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, saving checkpoint...")
            self.sync_to_drive()
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
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
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Scheduler step
        self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        return grad_norm
    
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
            
            if self.use_amp:
                with autocast():
                    logits = self.model(input_ids)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / max(1, num_batches)
    
    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time", max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
        """Generate a text sample from the model."""
        self.model.eval()
        
        # Load tokenizer
        from tokenizer.tokenizer import AethyxTokenizer
        tokenizer = AethyxTokenizer()
        
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        
        for _ in range(max_new_tokens):
            logits = self.model(ids[:, -128:])  # Crop to context length
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        
        self.model.train()
        return tokenizer.decode(ids[0].tolist())
    
    def log_generated_sample(self, step: int, loss: float, prompt: str = "Once upon a time"):
        """Generate and log a sample to file and TensorBoard."""
        try:
            generated = self.generate_sample(prompt)
            
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"\n{'='*80}\n"
                f"Step: {step} | Loss: {loss:.4f} | Time: {timestamp}\n"
                f"Prompt: {prompt}\n"
                f"Generated: {generated}\n"
                f"{'='*80}\n"
            )
            
            # Write to file
            with open(self.samples_log, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_text("generated_samples", log_entry, step)
            
            print(f"\n[Sample @ Step {step}]\n{generated[:200]}...\n")
            
        except Exception as e:
            warnings.warn(f"Sample generation failed: {e}")
    
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
            "rng_state": {
                "python": __import__('random').getstate(),
                "numpy": __import__('numpy').random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
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
        
        # Sync to Drive (if on Colab)
        self.sync_to_drive()
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        
        # Restore RNG states
        if "rng_state" in checkpoint:
            import random
            import numpy as np
            random.setstate(checkpoint["rng_state"]["python"])
            np.random.set_state(checkpoint["rng_state"]["numpy"])
            torch.set_rng_state(torch.ByteTensor(checkpoint["rng_state"]["torch"]))
            if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
                cuda_states = [torch.ByteTensor(s) for s in checkpoint["rng_state"]["cuda"]]
                torch.cuda.set_rng_state_all(cuda_states)
        
        print(f"Loaded checkpoint from step {self.step}")
    
    def sync_to_drive(self):
        """Sync checkpoints, logs, and config to Google Drive."""
        if self._sync_in_progress:
            return
        
        self._sync_in_progress = True
        try:
            drive_root = Path("/content/drive/MyDrive/AethyxLM")
            if not drive_root.exists():
                return
            
            # Sync checkpoints
            if self.checkpoint_dir.exists():
                for f in self.checkpoint_dir.glob("*.pt"):
                    dst = Path("/content/drive/MyDrive/AethyxLM/checkpoints") / f.name
                    if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                        import shutil
                        shutil.copy2(f, dst)
            
            # Sync logs
            log_dir = Path("logs")
            if log_dir.exists():
                for f in log_dir.rglob("*"):
                    if f.is_file():
                        rel = f.relative_to(log_dir)
                        dst = Path("/content/drive/MyDrive/AethyxLM/logs") / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy2(f, dst)
            
            # Sync configs
            for cfg in ["configs/train_config.json", "configs/train_config_colab.json"]:
                if Path(cfg).exists():
                    import shutil
                    shutil.copy2(cfg, Path("/content/drive/MyDrive/AethyxLM/configs") / Path(cfg).name)
                    
        except Exception as e:
            warnings.warn(f"Drive sync failed: {e}")
        finally:
            self._sync_in_progress = False
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on {self.device}")
        print(f"Steps: {self.max_steps}, Warmup: {self.warmup_steps}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"TensorBoard: {self.tensorboard_dir}")
        
        self.model.train()
        running_loss = 0.0
        step_start_time = time.time()
        tokens_per_step = self.train_dataloader.batch_size * CONTEXT_LENGTH
        
        while self.step < self.max_steps:
            self.epoch += 1
            
            for batch in self.train_dataloader:
                if self.step >= self.max_steps:
                    break
                
                loss = self.train_step(batch)
                running_loss += loss
                
                # Optimizer step (after accumulation)
                if (self.step + 1) % self.grad_accum_steps == 0:
                    grad_norm = self.optimizer_step()
                
                self.step += 1
                
                # Logging
                if self.step % self.log_interval == 0:
                    elapsed = time.time() - step_start_time
                    lr = self.optimizer.param_groups[0]["lr"]
                    avg_loss = running_loss / self.log_interval
                    tokens_per_sec = (tokens_per_step * self.log_interval) / elapsed
                    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    
                    # Console output
                    print(
                        f"Step {self.step}/{self.max_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {grad_norm:.4f} | "
                        f"Tok/s: {tokens_per_sec:.0f} | "
                        f"GPU: {gpu_mem:.2f}GB | "
                        f"Time: {elapsed:.2f}s | "
                        f"ETA: {(self.max_steps - self.step) * elapsed / self.log_interval / 3600:.1f}h"
                    )
                    
                    # TensorBoard logging
                    if self.writer:
                        self.writer.add_scalar("train/loss", avg_loss, self.step)
                        self.writer.add_scalar("train/learning_rate", lr, self.step)
                        self.writer.add_scalar("train/grad_norm", grad_norm, self.step)
                        self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)
                        self.writer.add_scalar("train/gpu_memory_gb", gpu_mem, self.step)
                    
                    running_loss = 0.0
                    step_start_time = time.time()
                
                # Evaluation
                if self.val_dataloader and self.step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    perplexity = torch.exp(torch.tensor(val_loss)).item()
                    print(f"Validation Loss: {val_loss:.4f} | Perplexity: {perplexity:.2f}")
                    
                    if self.writer:
                        self.writer.add_scalar("val/loss", val_loss, self.step)
                        self.writer.add_scalar("val/perplexity", perplexity, self.step)
                    
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    self.save_checkpoint(is_best=is_best)
                
                # Sample generation
                if self.step % self.generate_interval == 0:
                    avg_loss = running_loss / self.log_interval if self.log_interval > 0 else 0
                    self.log_generated_sample(self.step, avg_loss)
                
                # Periodic checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint()
        
        # Final checkpoint
        self.save_checkpoint(is_best=False)
        print("Training complete!")
        
        if self.writer:
            self.writer.close()


# Backward compatibility - original interface
class TrainerLegacy:
    """Legacy Trainer interface for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        # Map old params to new
        self.trainer = Trainer(*args, **kwargs)
    
    def __getattr__(self, name):
        return getattr(self.trainer, name)


if __name__ == "__main__":
    # Quick test
    from model.gpt import GPT
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    model = GPT()
    train_data = TensorDataset(torch.randint(0, 32000, (100, 128)), torch.randint(0, 32000, (100, 128)))
    val_data = TensorDataset(torch.randint(0, 32000, (20, 128)), torch.randint(0, 32000, (20, 128)))
    
    train_loader = DataLoader(train_data, batch_size=2)
    val_loader = DataLoader(val_data, batch_size=2)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_steps=5,
        log_interval=2,
        eval_interval=3,
        save_interval=3,
        generate_interval=3,
        device="cpu",
        use_amp=False,
    )
    
    trainer.train()
    print("Trainer test passed!")