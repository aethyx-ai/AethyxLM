"""
Tests for training components.
"""

import os
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from model.gpt import GPT
from model.config import VOCAB_SIZE, CONTEXT_LENGTH, EMBED_DIM, NUM_LAYERS
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from training.trainer import Trainer


def test_loss():
    """Test LanguageModelLoss."""
    print("Testing LanguageModelLoss...")
    
    loss_fn = LanguageModelLoss()
    
    batch_size = 2
    seq_len = 16
    vocab_size = VOCAB_SIZE
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = loss_fn(logits, targets)
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert not torch.isinf(loss), "Loss should not be Inf"
    
    print(f"  Loss: {loss.item():.4f}")
    print("  [OK] LanguageModelLoss test passed")


def test_optimizer():
    """Test AdamW optimizer creation."""
    print("Testing AdamW optimizer...")
    
    model = GPT()
    optimizer = create_optimizer(model, learning_rate=3e-4)
    
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.1
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    
    # Test step
    input_ids = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_LENGTH))
    logits = model(input_ids)
    targets = torch.randint(0, VOCAB_SIZE, (2, CONTEXT_LENGTH))
    
    loss_fn = LanguageModelLoss()
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("  [OK] AdamW optimizer test passed")


def test_scheduler():
    """Test learning rate scheduler."""
    print("Testing LR scheduler...")
    
    model = GPT()
    optimizer = create_optimizer(model, learning_rate=3e-4)
    
    warmup_steps = 10
    max_steps = 100
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    
    # Check warmup
    for step in range(warmup_steps):
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        expected = 3e-4 * (step + 1) / warmup_steps
        assert abs(lr - expected) < 1e-10, f"Step {step}: lr={lr}, expected={expected}"
    
    # Check cosine decay
    for step in range(warmup_steps, max_steps):
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert lr <= 3e-4, f"LR should decay: {lr}"
    
    # Check minimum LR
    min_lr = 3e-4 * 0.1  # min_lr_ratio = 0.1
    final_lr = optimizer.param_groups[0]["lr"]
    assert abs(final_lr - min_lr) < 1e-6, f"Final LR: {final_lr}, min: {min_lr}"
    
    print("  [OK] Scheduler test passed")


def test_trainer():
    """Test Trainer initialization and single step."""
    print("Testing Trainer...")
    
    model = GPT()
    
    # Create dummy data
    batch_size = 4
    num_batches = 10
    data = torch.randint(0, VOCAB_SIZE, (num_batches * batch_size, CONTEXT_LENGTH))
    targets = torch.randint(0, VOCAB_SIZE, (num_batches * batch_size, CONTEXT_LENGTH))
    
    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=3e-4,
        max_steps=5,
        warmup_steps=2,
        grad_accum_steps=1,
        use_amp=False,  # Disable for CPU test
        checkpoint_dir="test_checkpoints",
        log_interval=2,
        eval_interval=3,
        save_interval=10,
    )
    
    # Test single epoch
    initial_step = trainer.step
    for batch in train_loader:
        if trainer.step >= trainer.max_steps:
            break
        loss = trainer.train_step(batch)
        assert loss > 0
        trainer.step += 1
        
        if (trainer.step) % trainer.grad_accum_steps == 0:
            trainer.optimizer_step()
    
    assert trainer.step > initial_step
    
    # Test evaluation
    val_loss = trainer.evaluate()
    assert val_loss > 0
    assert not torch.isnan(torch.tensor(val_loss))
    
    print("  [OK] Trainer test passed")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Testing checkpoint save/load...")
    
    model = GPT()
    
    batch_size = 2
    data = torch.randint(0, VOCAB_SIZE, (batch_size * 5, CONTEXT_LENGTH))
    targets = torch.randint(0, VOCAB_SIZE, (batch_size * 5, CONTEXT_LENGTH))
    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            max_steps=3,
            warmup_steps=1,
            checkpoint_dir=tmpdir,
            use_amp=False,
        )
        
        # Train a bit
        for batch in train_loader:
            if trainer.step >= trainer.max_steps:
                break
            trainer.train_step(batch)
            trainer.step += 1
            if trainer.step % trainer.grad_accum_steps == 0:
                trainer.optimizer_step()
        
        step_before = trainer.step
        
        # Save checkpoint
        trainer.save_checkpoint()
        
        # Create new trainer and load
        model2 = GPT()
        trainer2 = Trainer(
            model=model2,
            train_dataloader=train_loader,
            max_steps=10,
            checkpoint_dir=tmpdir,
            use_amp=False,
        )
        
        latest_path = os.path.join(tmpdir, "checkpoint_latest.pt")
        trainer2.load_checkpoint(latest_path)
        
        assert trainer2.step == step_before
        print("  [OK] Checkpoint save/load test passed")


def test_full_training():
    """Test a complete mini training run."""
    print("Testing full mini training run...")
    
    model = GPT()
    
    batch_size = 2
    data = torch.randint(0, VOCAB_SIZE, (20, CONTEXT_LENGTH))
    targets = torch.randint(0, VOCAB_SIZE, (20, CONTEXT_LENGTH))
    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=1e-3,
        max_steps=10,
        warmup_steps=3,
        grad_accum_steps=1,
        use_amp=False,
        checkpoint_dir="test_checkpoints_full",
        log_interval=5,
        eval_interval=5,
        save_interval=10,
    )
    
    trainer.train()
    
    assert trainer.step == 10
    print("  [OK] Full training run test passed")


if __name__ == "__main__":
    test_loss()
    test_optimizer()
    test_scheduler()
    test_trainer()
    test_checkpoint_save_load()
    test_full_training()
    print("\n[OK] All training tests passed!")