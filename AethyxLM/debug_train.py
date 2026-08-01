"""
Systematic debugging audit for AethyxLM training pipeline.
Runs through all 15 checks to identify root cause of OOM/slowness.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.gpt import GPT
from model.config import (
    VOCAB_SIZE, CONTEXT_LENGTH, EMBED_DIM, NUM_HEADS,
    NUM_LAYERS, FFN_DIM, DROPOUT, USE_BIAS, LAYER_NORM_EPS
)
from training.trainer import Trainer
from training.loss import LanguageModelLoss
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from dataset.dataset import AethyxDataset

def print_mem(stage: str):
    """Print GPU memory at each stage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{stage}] Allocated: {allocated:.3f} GB | Reserved: {reserved:.3f} GB")

def check_model_size():
    """CHECK 1: Model size verification."""
    print("\n" + "="*60)
    print("CHECK 1: MODEL SIZE")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Estimated FP32 memory: {total_params * 4 / 1e9:.3f} GB")
    print(f"Estimated FP16 memory: {total_params * 2 / 1e9:.3f} GB")
    print(f"Optimizer states (AdamW, 2x params): {trainable_params * 2 * 4 / 1e9:.3f} GB (FP32)")
    
    return model

def check_gpu_memory_stages():
    """CHECK 2: GPU memory at each stage."""
    print("\n" + "="*60)
    print("CHECK 2: GPU MEMORY AT EACH STAGE")
    print("="*60)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print_mem("INITIAL (empty cache)")
    
    # Model creation
    model = GPT(vocab_size=VOCAB_SIZE)
    model.to('cuda')
    print_mem("AFTER model.to('cuda')")
    
    # Optimizer creation
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    print_mem("AFTER optimizer creation")
    
    # Dummy batch
    batch_size = 8
    seq_len = CONTEXT_LENGTH
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device='cuda')
    print_mem("AFTER batch creation on CUDA")
    
    # Forward pass
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
        print_mem("AFTER forward (inside autocast)")
    
    # Loss
    criterion = LanguageModelLoss()
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item():.4f}")
    print_mem("AFTER loss computation")
    
    # Backward
    loss.backward()
    print_mem("AFTER backward")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print_mem("AFTER optimizer.step() + zero_grad")
    
    return model, optimizer

def check_amp_active():
    """CHECK 3: Verify AMP is actually active."""
    print("\n" + "="*60)
    print("CHECK 3: AMP VERIFICATION")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    
    print(f"autocast enabled: {torch.is_autocast_enabled()}")
    
    with torch.amp.autocast('cuda', enabled=True):
        print(f"Inside autocast - is_autocast_enabled: {torch.is_autocast_enabled()}")
        print(f"Inside autocast - get_autocast_gpu_dtype: {torch.get_autocast_gpu_dtype()}")
        logits = model(input_ids)
        print(f"Logits dtype: {logits.dtype}")
        print(f"Model param dtype: {next(model.parameters()).dtype}")
    
    print(f"Outside autocast - is_autocast_enabled: {torch.is_autocast_enabled()}")
    
    # Check GradScaler
    print(f"GradScaler enabled: {scaler.is_enabled()}")
    print(f"GradScaler scale: {scaler.get_scale()}")

def check_device_placement():
    """CHECK 4: Verify all tensors on correct device."""
    print("\n" + "="*60)
    print("CHECK 4: DEVICE PLACEMENT")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    
    print(f"token_embedding: {model.token_embedding.weight.device}")
    print(f"position_embedding: {model.position_embedding.weight.device}")
    # final_norm may not have weight/bias if LayerNorm is custom
    if hasattr(model.final_norm, 'weight'):
        print(f"final_norm: {model.final_norm.weight.device}")
    else:
        print(f"final_norm: (no weight attr)")
    print(f"lm_head: {model.lm_head.weight.device}")
    
    for i, layer in enumerate(model.layers):
        print(f"  Layer {i}:")
        if hasattr(layer.norm1, 'weight'):
            print(f"    norm1: {layer.norm1.weight.device}")
        if hasattr(layer.attention.q_proj, 'weight'):
            print(f"    attention.q_proj: {layer.attention.q_proj.weight.device}")
        if hasattr(layer.attention.k_proj, 'weight'):
            print(f"    attention.k_proj: {layer.attention.k_proj.weight.device}")
        if hasattr(layer.attention.v_proj, 'weight'):
            print(f"    attention.v_proj: {layer.attention.v_proj.weight.device}")
        if hasattr(layer.attention.out_proj, 'weight'):
            print(f"    attention.out_proj: {layer.attention.out_proj.weight.device}")
        if hasattr(layer.norm2, 'weight'):
            print(f"    norm2: {layer.norm2.weight.device}")
        if hasattr(layer.feed_forward.fc1, 'weight'):
            print(f"    feed_forward.fc1: {layer.feed_forward.fc1.weight.device}")
        if hasattr(layer.feed_forward.fc2, 'weight'):
            print(f"    feed_forward.fc2: {layer.feed_forward.fc2.weight.device}")
    
    # Check causal mask
    print(f"Causal mask device: {model.layers[0].attention.causal_mask.device}")
    
    # Input tensors
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    print(f"input_ids device: {input_ids.device}")
    print(f"targets device: {targets.device}")
    
    # Optimizer state
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state.get(p, {})
            if 'exp_avg' in state:
                print(f"Optimizer exp_avg device: {state['exp_avg'].device}")
                break

def check_batch_size_and_seq_len():
    """CHECK 5 & 6: Verify actual batch size and sequence length."""
    print("\n" + "="*60)
    print("CHECK 5 & 6: BATCH SIZE & SEQUENCE LENGTH")
    print("="*60)
    
    # Test with actual dataloader
    train_dataset = AethyxDataset(
        text_path='data/train.txt',
        context_length=CONTEXT_LENGTH,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    
    for i, batch in enumerate(train_loader):
        input_ids, targets = batch
        print(f"Batch {i}: input_ids.shape = {input_ids.shape}, targets.shape = {targets.shape}")
        print(f"  Batch size: {input_ids.shape[0]}")
        print(f"  Sequence length: {input_ids.shape[1]}")
        print(f"  Dtype: {input_ids.dtype}")
        if i >= 2:
            break

def check_dataloader_config():
    """CHECK 7: DataLoader configuration."""
    print("\n" + "="*60)
    print("CHECK 7: DATALOADER CONFIG")
    print("="*60)
    
    train_dataset = AethyxDataset(
        text_path='data/train.txt',
        context_length=CONTEXT_LENGTH,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    
    print(f"batch_size: {train_loader.batch_size}")
    print(f"num_workers: {train_loader.num_workers}")
    print(f"pin_memory: {train_loader.pin_memory}")
    print(f"drop_last: {train_loader.drop_last}")
    print(f"sampler: {train_loader.sampler}")
    print(f"persistent_workers: {getattr(train_loader, 'persistent_workers', 'N/A')}")

def check_loss_finite():
    """CHECK 8: Loss finiteness."""
    print("\n" + "="*60)
    print("CHECK 8: LOSS FINITENESS")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    criterion = LanguageModelLoss()
    
    for i in range(5):
        input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
        targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
        
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        
        print(f"Iter {i}: loss = {loss.item():.4f} | isnan={torch.isnan(loss).item()} | isinf={torch.isinf(loss).item()}")

def check_backward_oom():
    """CHECK 9: Determine exactly where OOM occurs."""
    print("\n" + "="*60)
    print("CHECK 9: BACKWARD PASS OOM LOCATION")
    print("="*60)
    
    torch.cuda.empty_cache()
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    criterion = LanguageModelLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    
    try:
        print_mem("Before forward")
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(input_ids)
        print_mem("After forward")
        
        loss = criterion(logits, targets)
        print_mem("After loss")
        
        scaled_loss = scaler.scale(loss)
        print_mem("After scaler.scale")
        
        scaled_loss.backward()
        print_mem("After backward")
        
        scaler.unscale_(optimizer)
        print_mem("After unscale")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print_mem("After clip_grad_norm")
        
        scaler.step(optimizer)
        print_mem("After scaler.step")
        
        scaler.update()
        print_mem("After scaler.update")
        
        optimizer.zero_grad(set_to_none=True)
        print_mem("After zero_grad")
        
        print("SUCCESS: Full step completed without OOM")
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM ERROR at this stage!")
        print_mem("At OOM")
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        raise

def check_gradients():
    """CHECK 10: Gradient handling."""
    print("\n" + "="*60)
    print("CHECK 10: GRADIENT HANDLING")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    criterion = LanguageModelLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    
    # Test zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
        loss = criterion(logits, targets)
    
    scaler.scale(loss).backward()
    
    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    print(f"Gradients exist after backward: {has_grad}")
    
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"Grad norm: {grad_norm:.4f}")
    
    scaler.step(optimizer)
    scaler.update()
    
    # Check zero_grad behavior
    optimizer.zero_grad(set_to_none=True)
    has_grad_after = any(p.grad is not None for p in model.parameters())
    print(f"Gradients exist after zero_grad(set_to_none=True): {has_grad_after}")
    
    # Test without set_to_none
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
        loss = criterion(logits, targets)
    scaler.scale(loss).backward()
    optimizer.zero_grad()  # default set_to_none=False
    has_grad_default = any(p.grad is not None for p in model.parameters())
    print(f"Gradients exist after zero_grad() [default]: {has_grad_default}")

def check_computation_graph():
    """CHECK 11: Computation graph retention."""
    print("\n" + "="*60)
    print("CHECK 11: COMPUTATION GRAPH RETENTION")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    criterion = LanguageModelLoss()
    
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    
    # Check if loss retains graph
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
        loss = criterion(logits, targets)
    
    print(f"loss.requires_grad: {loss.requires_grad}")
    print(f"loss.grad_fn: {loss.grad_fn}")
    print(f"logits.requires_grad: {logits.requires_grad}")
    
    # Check for accidental retention
    losses = []
    for i in range(3):
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        losses.append(loss.item())  # Correct: .item()
    
    print(f"Losses list (with .item()): {losses}")
    
    # What if we don't use .item()?
    losses_bad = []
    for i in range(3):
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        losses_bad.append(loss)  # WRONG: retains graph
    
    print(f"Losses list (WITHOUT .item()): {[l.item() for l in losses_bad]}")
    print(f"  Graphs retained: {len([l for l in losses_bad if l.grad_fn is not None])}")

def check_optimizer_states():
    """CHECK 12: Optimizer state creation."""
    print("\n" + "="*60)
    print("CHECK 12: OPTIMIZER STATES")
    print("="*60)
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    
    # Create optimizer ONCE
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    
    print("Optimizer created once. Param groups:")
    for i, group in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, params={len(group['params'])}")
    
    # Check state dict size
    state_dict = optimizer.state_dict()
    print(f"Optimizer state dict keys: {list(state_dict.keys())}")
    print(f"State dict 'state' entries: {len(state_dict['state'])}")
    
    # Run a few steps, verify state accumulates correctly
    criterion = LanguageModelLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
    for step in range(3):
        input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
        targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
        
        with torch.amp.autocast('cuda', enabled=True):
            logits = model(input_ids)
            loss = criterion(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Check optimizer state growth
        state = optimizer.state_dict()['state']
        total_state_params = sum(
            sum(v.numel() for v in s.values() if isinstance(v, torch.Tensor))
            for s in state.values()
        )
        print(f"Step {step+1}: Optimizer state tensors: {total_state_params:,} elements")

def check_checkpoints():
    """CHECK 13: Checkpoint saving."""
    print("\n" + "="*60)
    print("CHECK 13: CHECKPOINT SAVING")
    print("="*60)
    
    import tempfile
    import os
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = {
            "step": 100,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        path = os.path.join(tmpdir, "test_checkpoint.pt")
        torch.save(checkpoint, path)
        
        size_mb = os.path.getsize(path) / 1e6
        print(f"Checkpoint size: {size_mb:.2f} MB")
        
        # Load and verify
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        print(f"Loaded step: {loaded['step']}")
        print(f"Model state dict keys: {len(loaded['model_state_dict'])}")
        print(f"Optimizer state dict keys: {list(loaded['optimizer_state_dict'].keys())}")
        
        # Verify no GPU tensors in checkpoint
        for k, v in loaded['model_state_dict'].items():
            if v.device.type == 'cuda':
                print(f"WARNING: GPU tensor in checkpoint: {k}")
        for k, v in loaded['optimizer_state_dict']['state'].items():
            for sk, sv in v.items():
                if isinstance(sv, torch.Tensor) and sv.device.type == 'cuda':
                    print(f"WARNING: GPU tensor in optimizer state: {k}.{sk}")

def check_cuda_cache():
    """CHECK 14: CUDA cache summary."""
    print("\n" + "="*60)
    print("CHECK 14: CUDA MEMORY SUMMARY")
    print("="*60)
    
    torch.cuda.empty_cache()
    print_mem("Before model")
    
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    print_mem("After model")
    
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    print_mem("After optimizer")
    
    # Run one step
    criterion = LanguageModelLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    input_ids = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH), device='cuda')
    
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
        loss = criterion(logits, targets)
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    print_mem("After one complete step")
    
    print("\n--- Full CUDA Memory Summary ---")
    print(torch.cuda.memory_summary())

def minimal_reproduction():
    """CHECK 15: Minimal reproduction script."""
    print("\n" + "="*60)
    print("CHECK 15: MINIMAL REPRODUCTION")
    print("="*60)
    
    print("Building model...")
    model = GPT(vocab_size=VOCAB_SIZE).to('cuda')
    
    print("Creating optimizer...")
    optimizer = create_optimizer(model, learning_rate=3e-4, weight_decay=0.1)
    
    print("Generating random input...")
    batch_size = 8
    seq_len = CONTEXT_LENGTH
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device='cuda')
    targets = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device='cuda')
    
    print("Running forward pass...")
    with torch.amp.autocast('cuda', enabled=True):
        logits = model(input_ids)
    print(f"  Logits shape: {logits.shape}, dtype: {logits.dtype}")
    
    print("Computing loss...")
    criterion = LanguageModelLoss()
    loss = criterion(logits, targets)
    print(f"  Loss: {loss.item():.4f}")
    
    print("Running backward pass...")
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scaler.scale(loss).backward()
    
    print("Optimizer step...")
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    
    print("SUCCESS: Minimal reproduction completed without OOM!")

def main():
    print("="*60)
    print("AETHYXLM SYSTEMATIC DEBUGGING AUDIT")
    print("="*60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Run all checks
    check_model_size()
    check_gpu_memory_stages()
    check_amp_active()
    check_device_placement()
    check_batch_size_and_seq_len()
    check_dataloader_config()
    check_loss_finite()
    check_backward_oom()
    check_gradients()
    check_computation_graph()
    check_optimizer_states()
    check_checkpoints()
    check_cuda_cache()
    minimal_reproduction()
    
    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()