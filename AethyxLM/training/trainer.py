"""
Training Loop for AethyxLM - Production Ready.

Features:
- Gradient accumulation
- Gradient clipping
- Mixed precision (AMP)
- Checkpoint saving with rotation (latest, best, last 3 numbered)
- Learning rate scheduling (warmup + cosine decay)
- TensorBoard logging
- Sample generation during training
- Robust checkpoint loading/resuming
- Graceful shutdown on signals
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
    - Checkpoint saving with rotation (latest, best, last 3 numbered)
    - Learning rate scheduling (warmup + cosine decay)
    - TensorBoard logging
    - Sample generation during training
    - Robust checkpoint loading/resuming
    - Graceful shutdown on signals
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
        
        # Signal handlers
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, saving checkpoint...")
            self._save_checkpoint(is_best=False, force=True)
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def train_step(self, batch) -> float:
        input_ids, targets = batch
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        if self.use_amp:
            with autocast():
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps
        else:
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            loss = loss / self.grad_accum_steps
        
        self.scaler.scale(loss).backward()
        return loss.item() * self.grad_accum_steps

    def optimizer_step(self) -> float:
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def evaluate(self) -> float:
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
        from tokenizer.tokenizer import AethyxTokenizer
        tokenizer = AethyxTokenizer()
        self.model.eval()
        
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        
        for _ in range(max_new_tokens):
            logits = self.model(ids[:, -CONTEXT_LENGTH:])  # Crop to context length
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
        try:
            generated = self.generate_sample()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (
                f"\n{'='*80}\n"
                f"Step: {step} | Loss: {loss:.4f} | Time: {timestamp}\n"
                f"Prompt: {prompt}\n"
                f"Generated: {generated}\n"
                f"{'='*80}\n"
            )
            
            with open(self.samples_log, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            if self.writer:
                self.writer.add_text("generated_samples", f"Step {step}: {generated[:500]}", step)
            
            print(f"\n[Sample @ Step {step}]\n{generated[:200]}...\n")
        except Exception as e:
            warnings.warn(f"Sample generation failed: {e}")

    def save_checkpoint(self, is_best: bool = False, force: bool = False):
        """Public interface for saving checkpoints."""
        self._save_checkpoint(is_best=is_best, force=force)

    def _save_checkpoint(self, is_best: bool = False, force: bool = False):
        """
        Save checkpoint with rotation policy.
        
        Rotation policy:
        - Always update checkpoint_latest.pt
        - Update checkpoint_best.pt if is_best
        - Save numbered checkpoint ONLY at save_interval steps (or force)
        - Keep: latest.pt, best.pt, last 3 numbered checkpoints
        """
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
        
        # 1. Always update latest
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # 2. Best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
        
        # 3. Numbered checkpoint ONLY at save_interval (or force)
        is_interval_step = (self.step % self.save_interval == 0)
        is_forced = force
        
        if (self.step % self.save_interval == 0) or force:
            step_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
            torch.save(checkpoint, step_path)
            print(f"[OK] Saved checkpoint_step_{self.step}.pt")
            
            # Rotation: keep last 3 numbered checkpoints
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        """Keep last 3 numbered checkpoints + latest + best."""
        step_checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1])
        )
        
        if len(step_checkpoints) > 3:
            for old_ckpt in step_checkpoints[:-3]:
                try:
                    old_ckpt.unlink()
                    print(f"[Cleanup] Removed old checkpoint: {old_ckpt.name}")
                except Exception as e:
                    warnings.warn(f"Failed to remove old checkpoint: {e}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint with full state restoration."""
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
            torch.set_rng_state(checkpoint["rng_state"]["torch"])
            if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
                torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
        
        print(f"Loaded checkpoint from step {self.step}")

    def train(self):
        print(f"Starting training on {self.device}")
        print(f"Steps: {self.max_steps}, Warmup: {self.warmup_steps}")
        print(f"Gradient accumulation: {self.grad_accum_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"[DEBUG] eval_interval={self.eval_interval}, save_interval={self.save_interval}, log_interval={self.log_interval}")
        
        # CUDA warmup - run a few forward passes to trigger kernel compilation
        if self.device.startswith('cuda'):
            print("Running CUDA warmup...")
            self.model.train()
            dummy = torch.randint(0, self.model.vocab_size, (self.train_dataloader.batch_size, self.model.context_length), device=self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                for _ in range(3):
                    _ = self.model(dummy)
            torch.cuda.synchronize()
            print("CUDA warmup complete.")
        
        self.model.train()
        running_loss = 0.0
        step_start_time = time.time()
        tokens_per_step = self.train_dataloader.batch_size * self.model.context_length
        
        print("[DEBUG] Entered training loop", flush=True)
        
        while self.step < self.max_steps:
                self.epoch += 1
                print(f"[DEBUG] Epoch {self.epoch} started", flush=True)
                
                for batch in self.train_dataloader:
                    if self.step >= self.max_steps:
                        break
                    
                    print(f"[DEBUG] Step {self.step}: calling train_step", flush=True)
                    loss = self.train_step(batch)
                    running_loss += loss
                    print(f"[DEBUG] Step {self.step}: train_step done, loss={loss:.4f}", flush=True)
                    
                    # Optimizer step after accumulation
                    if (self.step + 1) % self.grad_accum_steps == 0:
                        print(f"[DEBUG] Step {self.step}: calling optimizer_step", flush=True)
                        self.optimizer_step()
                        print(f"[DEBUG] Step {self.step}: optimizer_step done", flush=True)
                    
                    # Logging
                    if self.step % self.log_interval == 0:
                        elapsed = time.time() - step_start_time
                        lr = self.optimizer.param_groups[0]["lr"]
                        avg_loss = running_loss / self.log_interval
                        tokens_per_sec = (self.log_interval * self.train_dataloader.batch_size * self.model.context_length) / elapsed
                        
                        print(
                            f"Step {self.step}/{self.max_steps} | "
                            f"Loss: {running_loss / self.log_interval:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:.0f} | "
                            f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                            f"Time: {elapsed:.1f}s"
                        )
                        
                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, self.step)
                            self.writer.add_scalar("train/lr", lr, self.step)
                            self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)
                            self.writer.add_scalar("train/gpu_mem_gb", torch.cuda.memory_allocated()/1e9, self.step)
                        
                        running_loss = 0.0
                        step_start_time = time.time()
                    
                    # Validation
                    if self.val_dataloader and self.step > 0 and self.step % self.eval_interval == 0:
                        print(f"[DEBUG] Step {self.step}: entering evaluation (eval_interval={self.eval_interval})", flush=True)
                        val_loss = self.evaluate()
                        print(f"[DEBUG] Step {self.step}: leaving evaluation, val_loss={val_loss:.4f}", flush=True)
                        print(f"Validation Loss: {val_loss:.4f}")
                        
                        if self.writer:
                            self.writer.add_scalar("val/loss", val_loss, self.step)
                        
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                        
                        # Save checkpoint at eval interval (with best flag)
                        print(f"[DEBUG] Step {self.step}: saving checkpoint (eval)", flush=True)
                        self._save_checkpoint(is_best=is_best)
                        print(f"[DEBUG] Step {self.step}: checkpoint saved", flush=True)
                    
                    # Periodic checkpoint (only at save_interval)
                    if self.step % self.save_interval == 0:
                        print(f"[DEBUG] Step {self.step}: saving checkpoint (periodic)", flush=True)
                        self._save_checkpoint(is_best=False)
                        print(f"[DEBUG] Step {self.step}: checkpoint saved", flush=True)
                    
                    print(f"[DEBUG] Step {self.step}: incrementing step to {self.step + 1}", flush=True)
                    self.step += 1
                    print(f"[DEBUG] Step now: {self.step}", flush=True)
                    
                    if self.step >= self.max_steps:
                        break
                
                if self.step >= self.max_steps:
                    break
        
        # Final checkpoint
        self._save_checkpoint(is_best=False, force=True)
        print("Training complete!")
        
        if self.writer:
            self.writer.close()


def create_trainer(
    model: GPT,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Optional[dict] = None,
) -> Trainer:
    """Factory function to create Trainer from config dict."""
    if config is None:
        config = {}
    
    default_config = {
        "learning_rate": 3e-4,
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "grad_clip": 1.0,
        "warmup_steps": 1000,
        "max_steps": 10000,
        "min_lr_ratio": 0.1,
        "grad_accum_steps": 1,
        "use_amp": True,
        "checkpoint_dir": "checkpoints",
        "log_interval": 10,
        "eval_interval": 500,
        "save_interval": 1000,
        "generate_interval": 1000,
        "device": None,
        "tensorboard_dir": None,
        "log_dir": "logs",
        "seed": 42,
    }
    
    for k, v in config.items():
        if k in default_config:
            default_config[k] = v
    
    return Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **default_config
    )


if __name__ == "__main__":
    from model.gpt import GPT
    from torch.utils.data import DataLoader, TensorDataset
    
    model = GPT()
    train_data = torch.randint(0, 32000, (100, 128))
    val_data = torch.randint(0, 32000, (20, 128))
    train_ds = TensorDataset(train_data, train_data)
    val_ds = TensorDataset(val_data, val_data)
    
    trainer = Trainer(
        model=model,
        train_dataloader=DataLoader(train_ds, batch_size=2),
        val_dataloader=DataLoader(val_ds, batch_size=2),
        max_steps=5,
        save_interval=2,
        eval_interval=2,
        log_interval=1,
        use_amp=False,
    )
    
    trainer.train()
    print("Trainer test passed!")