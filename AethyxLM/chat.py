#!/usr/bin/env python3
"""
AethyxLM - Local Chat Interface
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer


def find_latest_checkpoint(ckpt_dir="checkpoints"):
    """Find the latest valid checkpoint."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    valid = []
    for f in ckpt_dir.glob("checkpoint_step_*.pt"):
        try:
            if f.stat().st_size > 1_000_000:
                step = int(f.stem.split("_")[-1])
                torch.load(f, map_location="cpu", weights_only=False)
                valid.append((step, f))
        except:
            continue
    
    if not valid:
        raise FileNotFoundError("No valid checkpoints found!")
    
    return max(valid, key=lambda x: x[0])[1]


def safe_print(text):
    """Print with safe encoding for Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode('ascii', 'replace').decode('ascii')
        print(safe)


@torch.no_grad()
def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    for _ in range(max_new):
        logits = model(ids[:, -128:])
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())


def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode('ascii', 'replace').decode('ascii')
        print(safe)


def main():
    import torch
    from model.gpt import GPT
    from tokenizer.tokenizer import AethyxTokenizer
    from pathlib import Path
    import torch

    def find_latest_checkpoint(ckpt_dir="checkpoints"):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        valid = []
        for f in ckpt_dir.glob("checkpoint_step_*.pt"):
            try:
                if f.stat().st_size > 1_000_000:
                    step = int(f.stem.split("_")[-1])
                    torch.load(f, map_location="cpu", weights_only=False)
                    valid.append((step, f))
            except:
                continue
        
        if not valid:
            raise FileNotFoundError("No valid checkpoints!")
        
        return max(valid, key=lambda x: x[0])[1]

    print("=" * 60)
    print("AethyxLM Chat - Step ~3000 (partial training)")
    print("Commands: /temp <val>, /topk <val>, /max <val>, /clear, /quit")
    print("=" * 60)

    # Find checkpoint
    ckpt_path = find_latest_checkpoint()
    print(f"Loading: {ckpt_path.name}")

    # Load model
    ckpt = torch.load(find_latest_checkpoint(), map_location="cpu", weights_only=False)
    model = GPT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from tokenizer.tokenizer import AethyxTokenizer
    tok = AethyxTokenizer()
    print(f"Loaded step: {ckpt.get('step', 'unknown')}")
    print(f"Vocab: {tok.vocab_size}")

    @torch.no_grad()
    def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        for _ in range(max_new):
            logits = model(ids[:, -128:])
            logits = logits[:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return tok.decode(ids[0].tolist())

    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            safe = text.encode('ascii', 'replace').decode('ascii')
            print(safe)

    # Quick test
    model = GPT()
    ckpt = torch.load(find_latest_checkpoint(), map_location="cpu", weights_only=False)
    model = GPT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from tokenizer.tokenizer import AethyxTokenizer
    tok = AethyxTokenizer()

    @torch.no_grad()
    def gen(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        for _ in range(max_new):
            logits = model(ids[:, -128:])
            logits = logits[:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return tok.decode(ids[0].tolist())

    # Quick test
    test = gen(model, AethyxTokenizer(), "Once upon a time", 50)
    safe_print(f"Test: {test[:100]}...")

    print("\n" + "="*60)
    print("AethyxLM Chat - Commands: /temp, /topk, /max, /clear, /quit")
    print("="*60)

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
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() == "clear":
            history = ""
            print("[Context cleared]")
            continue
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
            elif cmd == "/help":
                print("Commands: /temp, /topk, /max, /clear, /quit")
                continue
            elif cmd == "/clear":
                history = ""
                print("[Context cleared]")
                continue
            else:
                print("Commands: /temp, /topk, /max, /clear, /quit")
                continue

        full_prompt = f"{history}\nUser: {prompt}\nAethyx:" if history else f"User: {prompt}\nAethyx:"

        try:
            full = gen(model, AethyxTokenizer(), full_prompt, max_new, temp, top_k)
            parts = full.split("Aethyx:")
            response = parts[-1].strip() if len(parts) > 1 else full.strip()
            safe_print(f"\nAethyx: {response}")
            history = f"{full_prompt} {response}"[-2000:]
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    import torch
    from model.gpt import GPT
    from tokenizer.tokenizer import AethyxTokenizer
    from pathlib import Path
    import torch

    def find_latest_checkpoint(ckpt_dir="checkpoints"):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        
        valid = []
        for f in ckpt_dir.glob("checkpoint_step_*.pt"):
            try:
                if f.stat().st_size > 1_000_000:
                    step = int(f.stem.split("_")[-1])
                    torch.load(f, map_location="cpu", weights_only=False)
                    valid.append((step, f))
            except:
                continue
        
        if not valid:
            raise FileNotFoundError("No valid checkpoints!")
        
        return max(valid, key=lambda x: x[0])[1]

    print("=" * 60)
    print("AethyxLM Chat - Step ~3000 (partial training)")
    print("Commands: /temp, /topk, /max, /clear, /quit")
    print("=" * 60)

    # Find checkpoint
    ckpt_path = find_latest_checkpoint()
    print(f"Loading: {ckpt_path.name}")

    # Load model
    ckpt = torch.load(find_latest_checkpoint(), map_location="cpu", weights_only=False)
    model = GPT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from tokenizer.tokenizer import AethyxTokenizer
    tok = AethyxTokenizer()
    print(f"Loaded step: {ckpt.get('step', 'unknown')}")
    print(f"Vocab: {tok.vocab_size}")

    @torch.no_grad()
    def generate(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        for _ in range(200):
            logits = model(ids[:, -128:])
            logits = logits[:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return tok.decode(ids[0].tolist())

    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            safe = text.encode('ascii', 'replace').decode('ascii')
            print(safe)

    # Quick test
    model = GPT()
    ckpt = torch.load(find_latest_checkpoint(), map_location="cpu", weights_only=False)
    model = GPT()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    from tokenizer.tokenizer import AethyxTokenizer
    tok = AethyxTokenizer()

    @torch.no_grad()
    def gen(model, tok, prompt, max_new=200, temp=0.8, top_k=50):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
        for _ in range(max_new):
            logits = model(ids[:, -128:])
            logits = logits[:, -1, :] / temp
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
        return tok.decode(ids[0].tolist())

    test = gen(model, AethyxTokenizer(), "Once upon a time", 50)
    safe_print(f"Test: {test[:100]}...")

    print("\n" + "="*60)
    print("AethyxLM Chat - Commands: /temp, /topk, /max, /clear, /quit")
    print("="*60)

    history = ""
    temp, top_k, max_new = 0.8, 50, 200

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() == "clear":
            history = ""
            print("[Context cleared]")
            continue
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
            elif cmd == "/help":
                print("Commands: /temp, /topk, /max, /clear, /quit")
                continue
            elif cmd == "/clear":
                history = ""
                print("[Context cleared]")
                continue
            else:
                print("Commands: /temp, /topk, /max, /clear, /quit")
                continue

        full_prompt = f"{history}\nUser: {prompt}\nAethyx:" if history else f"User: {prompt}\nAethyx:"

        try:
            full = gen(model, AethyxTokenizer(), f"{history}\nUser: {prompt}\nAethyx:", max_new, temp, top_k)
            parts = full.split("Aethyx:")
            response = parts[-1].strip() if len(parts) > 1 else full.strip()
            safe_print(f"\nAethyx: {response}")
            history = f"{full_prompt} {response}"[-2000:]
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()