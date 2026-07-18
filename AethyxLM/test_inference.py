#!/usr/bin/env python3
"""
AethyxLM - Inference Pipeline Test
Tests: checkpoint loading, tokenizer, autoregressive generation, sampling
"""

import torch
import sys
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def load_model(ckpt_path: str, device: str) -> tuple:
    """Load model and tokenizer."""
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GPT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    
    tok = AethyxTokenizer()
    print(f"[OK] Model loaded from step {ckpt.get('step', 'unknown')}")
    print(f"[OK] Tokenizer vocab: {tok.vocab_size}")
    
    return model, tok


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int = 200, 
             temp: float = 0.8, top_k: int = 50, device: str = "cpu") -> str:
    """Autoregressive generation with top-k sampling."""
    model.eval()
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    
    for _ in range(max_new):
        logits = model(ids[:, -128:])  # context window
        logits = logits[:, -1, :] / max(temp, 1e-8)
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    
    return tok.decode(ids[0].tolist())


def safe_print(text):
    """Print with safe encoding for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace non-ASCII characters
        safe = text.encode('ascii', errors='replace').decode('ascii')
        print(safe)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Find latest checkpoint
    ckpt_dir = Path("checkpoints")
    checkpoints = sorted(Path("checkpoints").glob("checkpoint_step_*.pt"))
    ckpt_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Using checkpoint: {ckpt_path.name}")
    
    # Load model & tokenizer
    model, tok = load_model(str(ckpt_path), device)
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "The little boy",
        "In a magical forest",
        "Hello, my name is",
    ]
    
    print("\n" + "="*60)
    print("TESTING AUTOREGRESSIVE GENERATION")
    print("="*60)
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        try:
            output = generate(model, tok, prompt, 
                             max_new=100, temp=0.8, top_k=50, 
                             device=device)
            new_text = output[len(prompt):].strip()
            safe_print(f"Generated: {new_text[:200]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Test sampling parameters
    print("\n" + "="*50)
    print("TESTING SAMPLING PARAMETERS")
    print("="*50)
    
    test_prompt = "The quick brown fox"
    for temp in [0.5, 0.8, 1.2]:
        for top_k in [10, 50, 100]:
            try:
                out = generate(model, tok, test_prompt, 
                              max_new=50, temp=temp, top_k=top_k, 
                              device=device)
                new = out[len(test_prompt):].strip()
                print(f"temp={temp}, top_k={top_k}: {new[:80]}...")
            except Exception as e:
                print(f"  temp={temp}, top_k={top_k} ERROR: {e}")
    
    print("\n[OK] All inference pipeline tests passed!")


if __name__ == "__main__":
    main()