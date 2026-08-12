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
import signal
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from model.gpt import GPT
from model.config import CONTEXT_LENGTH
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from training.checkpoint_backup import create_checkpoint_backup
from tracking import JsonlExperimentTracker


def create_grad_scaler(enabled: bool):
    """Use the current AMP scaler API with compatibility for PyTorch 2.0/2.1."""
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


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
        amp_dtype: str = "auto",
        fused_optimizer: bool = False,
        z_loss_coefficient: float = 0.0,
        context_schedule: Optional[list] = None,
        tokenizer_sha256: Optional[str] = None,
        eval_batches: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        milestone_interval: int = 0,
        milestone_dir: Optional[str] = None,
        metrics_file: Optional[str] = None,
        run_id: Optional[str] = None,
        checkpoint_backup: Optional[dict] = None,
    ):
        self.model = model
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = self.rank == 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        if amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be auto, float16, or bfloat16")
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.amp_dtype = (
            torch.bfloat16
            if amp_dtype == "bfloat16" or (amp_dtype == "auto" and bf16_supported)
            else torch.float16
        )
        self.context_schedule = sorted(
            context_schedule or [], key=lambda item: int(item["step"])
        )
        self.tokenizer_sha256 = tokenizer_sha256
        self.eval_batches = eval_batches
        self.tokenizer_path = tokenizer_path
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.generate_interval = generate_interval
        self.milestone_interval = int(milestone_interval or 0)
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Loss
        self.criterion = LanguageModelLoss(z_loss_coefficient=z_loss_coefficient)
        
        # Optimizer
        self.optimizer = create_optimizer(
            model,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused_optimizer and self.device.startswith("cuda"),
        )
        
        # Scheduler
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            min_lr_ratio=min_lr_ratio,
        )
        
        # AMP
        self.scaler = create_grad_scaler(
            enabled=self.use_amp and self.amp_dtype == torch.float16
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.tokens_seen = 0
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_backup = (
            create_checkpoint_backup(checkpoint_backup, self.checkpoint_dir)
            if self.is_main_process
            else None
        )
        self.milestone_dir = Path(
            milestone_dir or self.checkpoint_dir / "milestones"
        ).expanduser().resolve()
        if self.milestone_interval > 0:
            self.milestone_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = (
            JsonlExperimentTracker(
                metrics_file,
                run_id=run_id,
                metadata={
                    "max_steps": self.max_steps,
                    "checkpoint_dir": str(self.checkpoint_dir),
                    "milestone_interval": self.milestone_interval,
                },
            )
            if metrics_file and self.is_main_process
            else None
        )
        
        # TensorBoard
        self.tensorboard_dir = Path(tensorboard_dir or self.log_dir / "tensorboard")
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = None
        if TENSORBOARD_AVAILABLE and self.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Generated samples log
        self.samples_log = self.log_dir / "generated_samples.txt"
        
        # Sync control
        self._sync_in_progress = False
        
        # Signal handlers
        self._register_signal_handlers()

    def _track(self, event: str, **values):
        tracker = getattr(self, "tracker", None)
        if tracker is not None:
            tracker.log(
                event,
                step=self.step,
                tokens_seen=getattr(self, "tokens_seen", 0),
                **values,
            )

    def _register_signal_handlers(self):
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, saving checkpoint...")
            self._save_checkpoint(is_best=False, force=True)
            raise KeyboardInterrupt
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def train_step(self, batch) -> float:
        input_ids, targets = batch
        active_context = self._active_context_length()
        input_ids = input_ids[:, :active_context]
        targets = targets[:, :active_context]
        self.last_batch_tokens = input_ids.numel()
        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
                loss = loss / self.grad_accum_steps
        else:
            logits = self.model(input_ids)
            loss = self.criterion(logits, targets)
            loss = loss / self.grad_accum_steps
        
        self.scaler.scale(loss).backward()
        return loss.item() * self.grad_accum_steps

    def _active_context_length(self) -> int:
        """Return the curriculum context length for the current step."""
        model = self._raw_model()
        length = model.context_length
        for stage in self.context_schedule:
            if self.step >= int(stage["step"]):
                length = min(int(stage["context_length"]), model.context_length)
            else:
                break
        return length

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
            if self.eval_batches is not None and num_batches >= self.eval_batches:
                break
            input_ids, targets = batch
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)
            
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    logits = self.model(input_ids)
                    loss = self.criterion(logits, targets)
            else:
                logits = self.model(input_ids)
                loss = self.criterion(logits, targets)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        if getattr(self, "is_distributed", False):
            totals = torch.tensor(
                [total_loss, float(num_batches)], device=self.device, dtype=torch.float64
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss, num_batches = totals.tolist()
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def generate_sample(self, prompt: str = "Once upon a time", max_new_tokens: int = 100, temperature: float = 0.8, top_k: int = 50) -> str:
        from tokenizer.tokenizer import AethyxTokenizer
        tokenizer = (
            AethyxTokenizer(self.tokenizer_path)
            if self.tokenizer_path
            else AethyxTokenizer()
        )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model.eval()
        
        model = self._raw_model()
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=self.device)
        ids = ids[:, -model.context_length:]
        logits, cache = model(ids, use_cache=True)
        
        for _ in range(max_new_tokens):
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
            if tokenizer.eos_id is not None and int(next_id.item()) == tokenizer.eos_id:
                break
            if cache[0][0].size(2) >= model.context_length:
                logits, cache = model(ids[:, -model.context_length:], use_cache=True)
            else:
                logits, cache = model(next_id, kv_cache=cache, use_cache=True)
        
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
        """Save on explicit request; best checkpoints retain alias-only semantics."""
        if not is_best and not force:
            is_interval_step = self.step > 0 and self.step % self.save_interval == 0
            force = not is_interval_step
        self._save_checkpoint(is_best=is_best, force=force)

    def _save_checkpoint(self, is_best: bool = False, force: bool = False):
        """
        Save checkpoint with rotation policy.
        
        Rotation policy:
        - Update checkpoint_best.pt only after a new best validation loss
        - Update checkpoint_latest.pt at save intervals and forced shutdown/final saves
        - Save numbered checkpoints only at positive save_interval steps
        - Keep checkpoint_best.pt, checkpoint_latest.pt, and the last 3 numbered files
        """
        if not getattr(self, "is_main_process", True):
            return
        is_interval_step = self.step > 0 and self.step % self.save_interval == 0
        if not is_best and not force and not is_interval_step:
            return

        model = self._raw_model()
        checkpoint = {
            "step": self.step,
            "epoch": self.epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "tokens_seen": getattr(self, "tokens_seen", 0),
            "rng_state": {
                "python": __import__('random').getstate(),
                "numpy": __import__('numpy').random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "config": {
                "tokenizer_sha256": getattr(self, "tokenizer_sha256", None),
                "tokenizer": {
                    "sha256": getattr(self, "tokenizer_sha256", None),
                    "file_name": (
                        Path(self.tokenizer_path).name
                        if getattr(self, "tokenizer_path", None)
                        else None
                    ),
                },
                "model": {
                    "vocab_size": model.vocab_size,
                    "context_length": model.context_length,
                    "embed_dim": model.embed_dim,
                    "num_heads": model.num_heads,
                    "num_kv_heads": model.num_kv_heads,
                    "num_layers": model.num_layers,
                    "ffn_dim": model.ffn_dim,
                    "dropout": model.dropout_rate,
                    "use_bias": model.use_bias,
                    "layer_norm_eps": model.layer_norm_eps,
                    "normalization": model.normalization,
                    "position_encoding": model.position_encoding,
                    "ffn_type": model.ffn_type,
                    "rope_base": model.rope_base,
                    "rope_max_seq_len": model.rope_max_seq_len,
                    "rope_scaling_factor": model.rope_scaling_factor,
                    "fused_qkv": model.fused_qkv,
                    "use_sdpa": model.use_sdpa,
                    "qk_norm": model.qk_norm,
                    "gradient_checkpointing": model.gradient_checkpointing,
                    "context_adapter": model.context_adapter_config,
                    "sliding_window": model.sliding_window,
                    "global_attention_interval": model.global_attention_interval,
                }
            }
        }

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            self._atomic_torch_save(checkpoint, best_path)
            print(f"[OK] Updated {best_path.name} at step {self.step}")
            if not force:
                return

        if force or is_interval_step:
            latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
            self._atomic_torch_save(checkpoint, latest_path)
            print(f"[OK] Updated {latest_path.name} at step {self.step}")

        if is_interval_step:
            step_path = self.checkpoint_dir / f"checkpoint_step_{self.step}.pt"
            self._atomic_torch_save(checkpoint, step_path)
            print(f"[OK] Saved checkpoint_step_{self.step}.pt")
            milestone_interval = getattr(self, "milestone_interval", 0)
            if milestone_interval > 0 and self.step % milestone_interval == 0:
                from training.milestones import archive_milestone

                milestone_path = archive_milestone(step_path, self.milestone_dir)
                print(f"[OK] Preserved milestone {milestone_path}")
                self._track("milestone_saved", path=str(milestone_path))
            self._track("checkpoint_saved", path=str(step_path))
            self._rotate_checkpoints()
            self._backup_checkpoint(step_path)
        elif force:
            self._backup_checkpoint(latest_path)

    def _backup_checkpoint(self, checkpoint_path: Path):
        backup = getattr(self, "checkpoint_backup", None)
        if backup is None:
            return
        uploaded = backup.upload(checkpoint_path, self.step)
        if uploaded:
            self._track(
                "checkpoint_backed_up",
                path=str(checkpoint_path),
                destination=backup.handle,
            )

    @staticmethod
    def _atomic_torch_save(checkpoint: dict, path: Path):
        """Write a checkpoint completely before replacing its public path."""
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

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
        # Stage the serialized state on system RAM first. Loading a complete
        # optimizer checkpoint directly onto CUDA creates an avoidable VRAM
        # spike, which is especially costly on 6 GB laptop GPUs.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_tokenizer = checkpoint.get("config", {}).get("tokenizer_sha256")
        if (
            saved_tokenizer
            and self.tokenizer_sha256
            and saved_tokenizer != self.tokenizer_sha256
        ):
            raise RuntimeError("checkpoint tokenizer fingerprint does not match this run")

        model = self._raw_model()
        model.load_compatible_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.tokens_seen = checkpoint.get("tokens_seen", 0)
        
        # Restore RNG states
        if "rng_state" in checkpoint:
            import random
            import numpy as np
            random.setstate(checkpoint["rng_state"]["python"])
            np.random.set_state(checkpoint["rng_state"]["numpy"])
            # map_location may move this CPU RNG tensor onto CUDA.
            torch.set_rng_state(checkpoint["rng_state"]["torch"].cpu())
            if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"]:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in checkpoint["rng_state"]["cuda"]]
                )
        
        print(f"Loaded checkpoint from step {self.step}")
        self._track("checkpoint_loaded", path=str(Path(path).expanduser().resolve()))

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
            raw_model = self._raw_model()
            dummy = torch.randint(
                0,
                raw_model.vocab_size,
                (self.train_dataloader.batch_size, raw_model.context_length),
                device=self.device,
            )
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                for _ in range(3):
                    _ = self.model(dummy)
            torch.cuda.synchronize()
            print("CUDA warmup complete.")
        
        self.model.train()
        running_loss = 0.0
        step_start_time = time.time()
        interval_tokens = 0
        interval_microbatches = 0
        microbatches_since_update = 0
        
        print("[DEBUG] Entered training loop", flush=True)
        
        while self.step < self.max_steps:
                self.epoch += 1
                print(f"[DEBUG] Epoch {self.epoch} started", flush=True)
                dataset = getattr(self.train_dataloader, "dataset", None)
                if hasattr(dataset, "set_epoch"):
                    dataset.set_epoch(self.epoch)
                sampler = getattr(self.train_dataloader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self.epoch)
                
                for batch in self.train_dataloader:
                    if self.step >= self.max_steps:
                        break

                    will_update = (
                        microbatches_since_update + 1 == self.grad_accum_steps
                    )
                    sync_context = (
                        self.model.no_sync()
                        if getattr(self, "is_distributed", False)
                        and hasattr(self.model, "no_sync")
                        and not will_update
                        else nullcontext()
                    )
                    with sync_context:
                        loss = self.train_step(batch)
                    running_loss += loss
                    batch_tokens = getattr(
                        self,
                        "last_batch_tokens",
                        self.train_dataloader.batch_size * self._raw_model().context_length,
                    )
                    batch_tokens *= (
                        dist.get_world_size()
                        if getattr(self, "is_distributed", False)
                        else 1
                    )
                    interval_tokens += batch_tokens
                    self.tokens_seen = getattr(self, "tokens_seen", 0) + batch_tokens
                    interval_microbatches += 1
                    microbatches_since_update += 1

                    if not will_update:
                        continue

                    self.optimizer_step()
                    microbatches_since_update = 0
                    # A step is one optimizer update, independent of accumulation.
                    self.step += 1

                    # Logging
                    if self.step % self.log_interval == 0:
                        elapsed = time.time() - step_start_time
                        lr = self.optimizer.param_groups[0]["lr"]
                        avg_loss = running_loss / max(interval_microbatches, 1)
                        tokens_per_sec = interval_tokens / elapsed
                        
                        print(
                            f"Step {self.step}/{self.max_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Tok/s: {tokens_per_sec:.0f} | "
                            f"GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB | "
                            f"Time: {elapsed:.1f}s"
                        )
                        
                        if self.writer:
                            self.writer.add_scalar("train/loss", avg_loss, self.step)
                            self.writer.add_scalar("train/lr", lr, self.step)
                            self.writer.add_scalar("train/tokens_per_sec", tokens_per_sec, self.step)
                            self.writer.add_scalar("train/tokens_seen", self.tokens_seen, self.step)
                            self.writer.add_scalar("train/gpu_mem_gb", torch.cuda.memory_allocated()/1e9, self.step)
                        self._track(
                            "train_metrics",
                            loss=avg_loss,
                            learning_rate=lr,
                            tokens_per_second=tokens_per_sec,
                            gpu_memory_gb=torch.cuda.memory_allocated() / 1e9,
                            context_length=self._active_context_length(),
                        )
                        
                        running_loss = 0.0
                        interval_tokens = 0
                        interval_microbatches = 0
                        step_start_time = time.time()

                    generate_interval = getattr(self, "generate_interval", 0)
                    if generate_interval > 0 and self.step % generate_interval == 0:
                        if getattr(self, "is_distributed", False):
                            dist.barrier()
                        if getattr(self, "is_main_process", True):
                            self.log_generated_sample(self.step, loss)
                        if getattr(self, "is_distributed", False):
                            dist.barrier()
                    
                    # Validation
                    if self.val_dataloader and self.step % self.eval_interval == 0:
                        val_loss = self.evaluate()
                        print(f"Validation Loss: {val_loss:.4f}")
                        
                        if self.writer:
                            self.writer.add_scalar("val/loss", val_loss, self.step)
                        self._track("validation", loss=val_loss)
                        
                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss
                        
                        if is_best:
                            self._save_checkpoint(is_best=True)

                    # Periodic checkpoint at completed save_interval steps.
                    if self.step % self.save_interval == 0:
                        self._save_checkpoint(is_best=False)

                    if self.step >= self.max_steps:
                        break
                
                if self.step >= self.max_steps:
                    break
        
        # The interval path already persisted this exact optimizer boundary.
        if self.step % self.save_interval != 0:
            self._save_checkpoint(is_best=False, force=True)
        print("Training complete!")
        self._track("run_completed")
        
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
