#!/usr/bin/env python3
"""
AethyxLM - Interactive Chat Interface
Loads a checkpoint and provides an interactive chat loop.
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def find_valid_checkpoint(ckpt_dir: Path) -> Path:
    """Find the latest valid checkpoint (>10MB)."""
    ckpts = list(ckpt_dir.glob("checkpoint_step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    for ckpt in ckpts:
        if ckpt.stat().st_size > 10_000_000:
            return ckpt
    raise RuntimeError("No valid checkpoints found (>10MB)")


def load_checkpoint(model: GPT, checkpoint_path: Path, device: str) -> dict:
    """Load checkpoint and return metadata."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[OK] Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    return checkpoint


@torch.no_grad()
def generate(model: GPT, tok: AethyxTokenizer, prompt: str, 
             max_new_tokens: int = 200, temperature: float = 0.8, 
             top_k: int = 50, device: str = "cpu") -> str:
    """Generate text from prompt using autoregressive sampling."""
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Crop to context length
        logits = model(ids[:, -128:])
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    
    return tok.decode(ids[0].tolist())


def safe_decode(text: str) -> str:
    """Safely decode text, replacing problematic characters."""
    try:
        return text
    except UnicodeEncodeError:
        return text.encode('ascii', errors='replace').decode('ascii')


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Find valid checkpoint
    ckpt_dir = Path("checkpoints")
    try:
        ckpt_path = find_valid_checkpoint(Path("checkpoints"))
    except Exception as e:
        print(f"[Error] {e}")
        return

    # Load model
    model = GPT().to(device)
    load_checkpoint(model, ckpt_path, device)

    # Load tokenizer (once, at startup)
    tok = AethyxTokenizer()
    print(f"[OK] Tokenizer vocab size: {tok.vocab_size}")

    # Generation parameters
    temperature = 0.8
    top_k = 50
    max_new = 200

    # Context for conversation
    context = ""

    print("\n" + "="*60)
    print("AethyxLM Chat - Type 'exit' to quit")
    print("Commands: /temp <val> | /topk <val> | /max <val> | /clear")
    print("="*60)

    context = ""
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            break
        if prompt.lower() == "clear":
            context = ""
            print("[Context cleared]")
            continue

        # Handle commands
        if prompt.startswith('/'):
            parts = prompt.split()
            cmd = parts[0].lower()
            if cmd == '/temp' and len(parts) > 1:
                temperature = float(parts[1])
                print(f"[OK] Temperature = {temperature}")
                continue
            elif cmd == '/topk' and len(parts) > 1:
                top_k = int(parts[1])
                print(f"[OK] Top-k = {top_k}")
                continue
            elif cmd == '/max' and len(parts) > 1:
                max_new = int(parts[1])
                print(f"[OK] Max new tokens = {max_new}")
                continue
            elif cmd == '/help':
                print("Commands: /temp <val> | /topk <val> | /max <val> | /clear")
                continue
            else:
                print("Unknown command. Use /help")
                continue

        # Build prompt with context
        full_prompt = context + "\nUser: " + prompt + "\nAethyx:"

        # Generate response
        try:
            full_response = generate(model, tok, full_prompt,
                                   max_new_tokens=max_new,
                                   temperature=temperature,
                                   top_k=top_k,
                                   device=device)
            
            # Extract just the new response
            response = full_response[len(full_prompt):].strip()
            safe_response = safe_decode(response)
            print(f"\nAethyx: {safe_response}")
            
            # Update context
            context = full_prompt + " " + response
            if len(context) > 2000:
                context = context[-2000:]
                
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()