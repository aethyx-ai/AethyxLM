#!/usr/bin/env python3
"""
AethyxLM - Local Chat Interface

Interactive chat with a trained AethyxLM model.
Commands: /temp <val>, /topk <val>, /max <val>, /clear, /quit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def find_best_checkpoint(ckpt_dir="checkpoints"):
    """Find the best checkpoint: latest step checkpoint, or best if newer than latest."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    # Find latest step checkpoint
    step_checkpoints = []
    for f in ckpt_dir.glob("checkpoint_step_*.pt"):
        try:
            if f.stat().st_size > 1_000_000:
                step = int(f.stem.split("_")[-1])
                torch.load(f, map_location="cpu", weights_only=False)
                step_checkpoints.append((step, f))
        except Exception:
            continue
    
    if step_checkpoints:
        latest_step, latest_path = max(step_checkpoints, key=lambda x: x[0])
        print(f"Using latest step checkpoint: step {latest_step}")
        return latest_path
    
    # Fall back to best checkpoint
    best_path = ckpt_dir / "checkpoint_best.pt"
    if best_path.exists() and best_path.stat().st_size > 1_000_000:
        try:
            torch.load(best_path, map_location="cpu", weights_only=False)
            print("Using best checkpoint (checkpoint_best.pt)")
            return best_path
        except Exception:
            pass
    
    # Fall back to latest checkpoint
    latest_path = ckpt_dir / "checkpoint_latest.pt"
    if latest_path.exists() and latest_path.stat().st_size > 1_000_000:
        try:
            torch.load(latest_path, map_location="cpu", weights_only=False)
            print("Using latest checkpoint (checkpoint_latest.pt)")
            return latest_path
        except Exception:
            pass
    
    raise FileNotFoundError("No valid checkpoints found!")


def safe_print(text):
    """Print with safe encoding for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode("ascii", "replace").decode("ascii")
        print(safe)


@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
    """Generate text from a prompt."""
    model.eval()
    context_length = model.context_length if hasattr(model, 'context_length') else 128
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    
    for _ in range(max_new):
        logits = model(ids[:, -context_length:])  # Crop to context length
        logits = logits[:, -1, :] / temp
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    
    return tok.decode(ids[0].tolist())


def main():
    print("=" * 60)
    print("AethyxLM Chat")
    print("Commands: /temp <val>, /topk <val>, /max <val>, /clear, /quit")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Find and load checkpoint (prefers latest step)
    ckpt_path = find_best_checkpoint()
    print(f"Loading: {ckpt_path.name}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Infer vocab size from actual checkpoint embeddings (not config)
    model_state = ckpt["model_state_dict"]
    vocab_size = model_state["token_embedding.weight"].shape[0]
    embed_dim = model_state["token_embedding.weight"].shape[1]
    
    # Build model config from checkpoint
    model_config = ckpt.get("config", {})
    model_config["vocab_size"] = vocab_size
    model_config["embed_dim"] = embed_dim
    
    print(f"Checkpoint config: vocab={vocab_size}, embed={embed_dim}, ctx={model_config.get('context_length')}, layers={model_config.get('num_layers')}")
    
    model = GPT(vocab_size=vocab_size, config=model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    tok = AethyxTokenizer()
    print(f"Loaded step: {ckpt.get('step', 'unknown')}")
    print(f"Tokenizer vocab: {tok.vocab_size}")
    print(f"Model vocab: {vocab_size}")
    print(f"Context: {model.context_length if hasattr(model, 'context_length') else 'unknown'}")
    
    # Test generation
    test = generate(model, tok, "Once upon a time", 50)
    safe_print(f"Test: {test[:100]}...")
    
    print("\n" + "=" * 60)
    print("Chat started. Type your message below.")
    print("=" * 60)
    
    history = ""
    temperature, top_k, max_new = 0.8, 50, 200
    
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not prompt:
            continue
        
        # Handle commands
        if prompt.lower() in ("quit", "exit", "q"):
            break
        
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            
            if cmd == "/temp" and len(parts) > 1:
                temperature = float(parts[1])
                print(f"[Temp = {temperature}]")
                continue
            elif cmd == "/topk" and len(parts) > 1:
                top_k = int(parts[1])
                print(f"[Top-k = {top_k}]")
                continue
            elif cmd == "/max" and len(parts) > 1:
                max_new = int(parts[1])
                print(f"[Max new = {max_new}]")
                continue
            elif cmd in ("/clear", "/reset"):
                history = ""
                print("[Context cleared]")
                continue
            elif cmd == "/help":
                print("Commands: /temp, /topk, /max, /clear, /quit")
                continue
            else:
                print("Unknown command. Type /help for options.")
                continue
        
        # Build prompt with history
        full_prompt = f"{history}\nUser: {prompt}\nAethyx:" if history else f"User: {prompt}\nAethyx:"
        
        try:
            full = generate(model, tok, full_prompt, max_new, temperature, top_k)
            parts = full.split("Aethyx:")
            response = parts[-1].strip() if len(parts) > 1 else full.strip()
            safe_print(f"\nAethyx: {response}")
            history = f"{full_prompt} {response}"[-2000:]  # Keep last 2000 chars
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
