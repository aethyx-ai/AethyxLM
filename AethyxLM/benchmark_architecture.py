"""
Benchmark script to compare GPT-2 style vs Modern architecture.
"""

import sys
sys.path.insert(0, 'D:/CODING/AETHYXLabs/AethyxLM')

import json
import time
import torch
import torch.nn as nn
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def benchmark_model(config, name, device, num_steps=50, warmup=5):
    """Benchmark a model configuration."""
    model = GPT(config=config).to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Create dummy input - use actual vocab size from tokenizer and correct seq_len
    tokenizer = AethyxTokenizer()
    actual_vocab = tokenizer.vocab_size
    batch_size = 8
    seq_len = 64  # Match model's context_length
    x = torch.randint(0, actual_vocab, (batch_size, seq_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    start = time.time()
    with torch.no_grad():
        for _ in range(num_steps):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    forward_time = (time.time() - start) / num_steps * 1000  # ms
    
    # Benchmark forward + backward
    model.train()
    tokenizer = AethyxTokenizer()
    actual_vocab = tokenizer.vocab_size
    seq_len = 64  # Match model's context_length
    x_train = torch.randint(0, actual_vocab, (8, seq_len), device=device)
    targets = torch.randint(0, actual_vocab, (8, seq_len), device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Warmup
    for _ in range(3):
        optimizer.zero_grad()
        out = model(torch.randint(0, actual_vocab, (8, seq_len), device=device))
        loss = nn.functional.cross_entropy(out.view(-1, actual_vocab), targets.view(-1))
        loss.backward()
        optimizer.step()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(num_steps):
        optimizer.zero_grad()
        out = model(torch.randint(0, actual_vocab, (8, seq_len), device=device))
        loss = nn.functional.cross_entropy(out.view(-1, actual_vocab), targets.view(-1))
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = (time.time() - start) / num_steps * 1000  # ms
    
    # Memory
    if device.type == 'cuda':
        max_mem = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    else:
        max_mem = 0
    
    return {
        "name": name,
        "params": total_params,
        "trainable_params": trainable_params,
        "forward_ms": forward_time,
        "train_ms": train_time,
        "peak_mem_mb": max_mem,
        "throughput_tokens_per_sec": (8 * seq_len * 1000) / train_time,
    }


def main():
    print("=" * 80)
    print("AethyxLM Architecture Benchmark")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load configs
    with open('D:/CODING/AETHYXLabs/AethyxLM/configs/train_config_kaggle.json', 'r') as f:
        gpt2_config = json.load(f)
    
    with open('D:/CODING/AETHYXLabs/AethyxLM/configs/train_config_modern.json', 'r') as f:
        modern_config = json.load(f)
    
    gpt2_config = gpt2_config['model']
    modern_config = modern_config['model']
    
    # Override vocab_size to match actual tokenizer (1908)
    gpt2_config['vocab_size'] = 1908
    modern_config['vocab_size'] = 1908
    
    # GPT-2 style (current)
    print("\n" + "=" * 80)
    print("Benchmarking GPT-2 Style (LayerNorm + Learned PosEmb + GELU)")
    print("=" * 80)
    gpt2_results = benchmark_model(gpt2_config, "GPT-2 Style", torch.device('cpu'))
    
    # Modern (RMSNorm + RoPE + SwiGLU)
    print("\n" + "=" * 80)
    print("Benchmarking Modern (RMSNorm + RoPE + SwiGLU)")
    print("=" * 80)
    modern_results = benchmark_model(modern_config, "Modern", torch.device('cpu'))
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<30} {'GPT-2 Style':<20} {'Modern':<20} {'Speedup':<15}")
    print("-" * 80)
    print(f"{'Parameters':<30} {gpt2_results['params']:,<20} {modern_results['params']:,<20} {modern_results['params']/gpt2_results['params']:.2f}x")
    print(f"{'Trainable Params':<30} {gpt2_results['trainable_params']:,<20} {modern_results['trainable_params']:,<20} {modern_results['trainable_params']/gpt2_results['trainable_params']:.2f}x")
    print(f"{'Forward Time (ms)':<30} {gpt2_results['forward_ms']:.2f}{'':<14} {modern_results['forward_ms']:.2f}{'':<14} {gpt2_results['forward_ms']/modern_results['forward_ms']:.2f}x")
    print(f"{'Train Step (ms)':<30} {gpt2_results['train_ms']:.2f}{'':<14} {modern_results['train_ms']:.2f}{'':<14} {gpt2_results['train_ms']/modern_results['train_ms']:.2f}x")
    print(f"{'Throughput (tok/s)':<30} {gpt2_results['throughput_tokens_per_sec']:,.0f}{'':<14} {modern_results['throughput_tokens_per_sec']:,.0f}{'':<14} {modern_results['throughput_tokens_per_sec']/gpt2_results['throughput_tokens_per_sec']:.2f}x")
    if torch.cuda.is_available():
        print(f"{'Peak Memory (MB)':<30} {gpt2_results['peak_mem_mb']:.0f}{'':<14} {modern_results['peak_mem_mb']:.0f}{'':<14} {modern_results['peak_mem_mb']/gpt2_results['peak_mem_mb']:.2f}x")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    speedup = gpt2_results['train_ms'] / modern_results['train_ms']
    if speedup > 1:
        print(f"Modern architecture is {speedup:.2f}x FASTER in training")
    else:
        print(f"Modern architecture is {1/speedup:.2f}x SLOWER in training")


if __name__ == "__main__":
    main()