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


def find_best_checkpoint(ckpt_dir=None):
    """Find the best checkpoint: latest step checkpoint, or best if newer than latest."""
    if ckpt_dir is None:
        ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
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
    if temp <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    context_length = model.context_length if hasattr(model, 'context_length') else 128
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    ids = ids[:, -context_length:]
    prompt_len = len(ids[0])  # Track where generation starts
    logits, cache = model(ids, use_cache=True)
    
    for _ in range(max_new):
        logits = logits[:, -1, :] / temp
        
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
        if tok.eos_id is not None and int(next_id.item()) == tok.eos_id:
            break
        if cache[0][0].size(2) >= context_length:
            logits, cache = model(ids[:, -context_length:], use_cache=True)
        else:
            logits, cache = model(next_id, kv_cache=cache, use_cache=True)
        
        # Check stop conditions on generated portion ONLY
        gen_ids = ids[0][prompt_len:]
        decoded = tok.decode(gen_ids.tolist())
        normalized = decoded.replace("Ġ", " ").replace("Ċ", "\n")
        norm_lower = normalized.lower()
        
        # Stop if model generates another conversation turn
        if "user:" in norm_lower:
            break
        # Stop if model generates "aethyx" marker again (but not at position 0)
        if "aethyx" in norm_lower:
            break
        # Stop if excessive colon repetition
        if normalized.count(":") > 8:
            break
        # Stop if excessive repetition
        if len(normalized) > 50 and normalized[-15:].count(normalized[-15:][:4]) > 3:
            break
        
        # Hard limit
        if len(normalized) > 500:
            break
    
    full_text = tok.decode(ids[0].tolist())
    # Fix ByteLevel BPE artifacts: Ġ (U+0120) = space, Ċ (U+010A) = newline
    full_text = full_text.replace("\u0120", " ").replace("\u010a", "\n")
    # Remove mojibake artifacts
    mojibake_chars = ["âĤ¬", "Åĵ", "âĦ¢", "â", "Ă", "ĵ", "Ĥ", "¬", "Å", "Ĥ", "¢"]
    for ch in mojibake_chars:
        full_text = full_text.replace(ch, "")
    # Fix ByteLevel BPE subword spacing: "us er" -> "user", "he ll o" -> "hello", "a et h y x" -> "aethyx"
    import re
    def fix_bytelevel_spacing(text):
        # Fix common subword patterns from ByteLevel BPE
        replacements = {
            'us er': 'user',
            'he ll o': 'hello',
            'a et h y x': 'aethyx',
            'a et h y x:': 'aethyx:',
            'u s e r': 'user',
            'h e l l o': 'hello',
            'o n c e': 'once',
            'u p o n': 'upon',
            't h e': 'the',
            'a n d': 'and',
            't o': 'to',
            'i n': 'in',
            'i t': 'it',
            'i s': 'is',
            'w a s': 'was',
            'w e r e': 'were',
            'b u t': 'but',
            'o r': 'or',
            'f o r': 'for',
            'w i t h': 'with',
            'a s': 'as',
            'h a s': 'has',
            'h a d': 'had',
            'd o': 'do',
            'w i l l': 'will',
            'c o u l d': 'could',
            'w o u l d': 'would',
            's h o u l d': 'should',
            't h i s': 'this',
            't h a t': 'that',
        }
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        return text
    
    full_text = fix_bytelevel_spacing(full_text)
    # General fix: remove spaces between single letters
    full_text = re.sub(r'(?<=[a-zA-Z]) (?=[a-zA-Z])', '', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    return full_text.strip()

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
    checkpoint_config = ckpt.get("config", {})
    model_config = checkpoint_config.get("model", checkpoint_config).copy()
    model_config["vocab_size"] = vocab_size
    model_config["embed_dim"] = embed_dim
    
    print(f"Checkpoint config: vocab={vocab_size}, embed={embed_dim}, ctx={model_config.get('context_length')}, layers={model_config.get('num_layers')}")
    
    model = GPT(vocab_size=vocab_size, config=model_config)
    model.load_compatible_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    
    tokenizer_info = ckpt.get("config", {}).get("tokenizer", {})
    tokenizer_name = tokenizer_info.get("file_name") or "tokenizer.json"
    tokenizer_path = Path(__file__).resolve().parent / "tokenizer" / tokenizer_name
    tok = AethyxTokenizer(tokenizer_path)
    expected_tokenizer_hash = (
        tokenizer_info.get("sha256")
        or ckpt.get("config", {}).get("tokenizer_sha256")
    )
    if expected_tokenizer_hash and tok.sha256 != expected_tokenizer_hash:
        raise RuntimeError(
            f"Checkpoint expects tokenizer SHA256 {expected_tokenizer_hash}, "
            f"but {tokenizer_path} has {tok.sha256}."
        )
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
            # Extract only the response after the last "Aethyx:" 
            if "Aethyx:" in full:
                response = full.split("Aethyx:")[-1]
            elif "a et h y x :" in full:
                response = full.split("a et h y x :")[-1]
            else:
                response = full
            # Remove any "User:" prefix that might have been generated
            for prefix in ["User:", "U s e r :"]:
                if prefix in response:
                    response = response.split(prefix)[0]
            response = response.strip()
            safe_print(f"\nAethyx: {response}")
            history = f"{full_prompt} {response}"[-2000:]  # Keep last 2000 chars
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
