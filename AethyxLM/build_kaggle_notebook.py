#!/usr/bin/env python3
"""
Build Kaggle training notebook programmatically using nbformat.
"""

import nbformat as nbf
from pathlib import Path


def build_notebook():
    nb = nbf.v4.new_notebook()

    # Set notebook metadata
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10'
        }
    }

    # Cell 0: Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell("""# AethyxLM - Production Kaggle Training (T4 GPU x2)

**Architecture:** 14M params, 8L, 256D, 8H, 128ctx, 32k vocab
**Dataset:** TinyStories (auto-download from Hugging Face)
**Storage:** GitHub = code, Kaggle Working = checkpoints/logs, Kaggle GPU = compute

---"""))

    # Cell 1: Setup
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 1: SETUP PROJECT (Kaggle)
# ============================================================
import os, sys, subprocess, shutil, json, time, glob, signal
from pathlib import Path

# Kaggle working directory (persists across restarts)
WORK_DIR = '/kaggle/working'
os.chdir('/kaggle/working')

# Project root
LOCAL_ROOT = '/kaggle/working/AethyxLM'

# Persistent directories on Kaggle working dir (survives restarts)
CKPT_DIR = '/kaggle/working/checkpoints'
LOGS_DIR = '/kaggle/working/logs'
TOK_DIR = '/kaggle/working/tokenizer'
DATA_DIR = '/kaggle/working/dataset'
CONFIG_DIR = '/kaggle/working/configs'

for d in [CKPT_DIR, LOGS_DIR, TOK_DIR, DATA_DIR, CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'[OK] Working dir: {WORK_DIR}')
print(f'[OK] Checkpoints: {CKPT_DIR}')
print(f'[OK] Logs: {LOGS_DIR}')
print(f'[OK] Configs: {CONFIG_DIR}')

# ============================================================
# CLONE/PULL FROM GITHUB (code lives in Git)
# ============================================================
REPO_URL = 'https://github.com/aethyx-ai/AethyxLM.git'
LOCAL_ROOT = '/kaggle/working/AethyxLM'

if os.path.exists(os.path.join(LOCAL_ROOT, '.git')):
    print('Updating existing repo...')
    subprocess.run(['git', '-C', LOCAL_ROOT, 'pull'], check=True)
else:
    print('Cloning repo...')
    subprocess.run(['git', 'clone', REPO_URL, LOCAL_ROOT], check=True)

# Fix nested directory from git clone
nested = os.path.join(LOCAL_ROOT, 'AethyxLM')
if os.path.exists(nested):
    for item in os.listdir(nested):
        src = os.path.join(nested, item)
        dst = os.path.join(LOCAL_ROOT, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
            shutil.move(src, LOCAL_ROOT)
        os.rmdir(nested)

os.chdir(LOCAL_ROOT)
sys.path.insert(0, LOCAL_ROOT)

print(f'[OK] Project: {LOCAL_ROOT}')
print(f'[OK] Config: {os.path.exists("configs/train_config.json")}')
print(f'[OK] Corpus: {os.path.exists("tokenizer/data/corpus.txt")}')

# Install deps
!pip install tokenizers datasets tensorboard -q"""))

    # Cell 2: Verify CUDA
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 2: VERIFY CUDA (accepts any CUDA GPU)
# ============================================================
import torch

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU not available! Enable GPU in Kaggle settings (Accelerator -> GPU T4 x2)')

device_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {device_name} ({vram_gb:.1f} GB)')

# Accept any CUDA GPU - just warn if unexpected
known_gpus = ['T4', 'L4', 'A100', 'V100', 'P100']
if not any(g in device_name for g in known_gpus):
    print(f"Warning: GPU '{device_name}' not in common Kaggle types. Proceeding anyway...")"""))

    # Cell 3: Prepare Data
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 3: PREPARE DATA (TinyStories from Hugging Face)
# ============================================================
import random
import os
import shutil

from datasets import load_dataset

print('Loading TinyStories from Hugging Face...')
ds = load_dataset('roneneldan/TinyStories', split='train')

# Use subset for faster training (adjust as needed)
NUM_STORIES = 50000  # Increase for full training
texts = ds['text'][:NUM_STORIES]

random.seed(42)
random.shuffle(texts)
split = int(0.95 * len(texts))
train_texts = texts[:split]
val_texts = texts[split:]

os.makedirs('data', exist_ok=True)
with open('data/train.txt', 'w', encoding='utf-8') as f:
    f.write('\\n\\n'.join(train_texts))
with open('data/val.txt', 'w', encoding='utf-8') as f:
    f.write('\\n\\n'.join(val_texts))

print(f'Train: {len(train_texts)} stories')
print(f'Val: {len(val_texts)} stories')

# Also copy to persistent storage for resume
shutil.copy('data/train.txt', os.path.join(WORK_DIR, 'train.txt'))
shutil.copy('data/val.txt', os.path.join(WORK_DIR, 'val.txt'))

print(f'Train: {len(train_texts)} stories')
print(f'Val: {len(val_texts)} stories)"""))

    # Cell 4: Train Tokenizer
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 4: TRAIN TOKENIZER (BPE, 32k vocab)
# ============================================================
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# Combine train + val for tokenizer training
with open('data/train.txt', 'r', encoding='utf-8') as f:
    train_data = f.read()
with open('data/val.txt', 'r', encoding='utf-8') as f:
    val_data = f.read()

combined_path = 'data/combined.txt'
with open(combined_path, 'w', encoding='utf-8') as f:
    f.write(train_data + '\\n\\n' + val_data)

print(f'Training BPE tokenizer ({VOCAB_SIZE} vocab)...')

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,
    show_progress=True
)
tokenizer.train([combined_path], trainer)
tokenizer.decoder = ByteLevelDecoder()

# Save tokenizer
os.makedirs('tokenizer', exist_ok=True)
tokenizer.save('tokenizer/tokenizer.json')

# Also save to persistent storage
shutil.copy('tokenizer/tokenizer.json', os.path.join(TOK_DIR, 'tokenizer.json'))
shutil.copy('data/combined.txt', os.path.join(TOK_DIR, 'corpus.txt'))

print(f'[OK] Tokenizer saved: tokenizer/tokenizer.json')
print(f'[OK] Vocab size: {tokenizer.get_vocab_size()}')"""))

    # Cell 5: Load Config
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 5: LOAD TRAINING CONFIG
# ============================================================
import json

with open('configs/train_config.json', 'r') as f:
    config = json.load(f)

print(json.dumps(config, indent=2))"""))

    # Cell 6: Build Model
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 6: BUILD MODEL
# ============================================================
import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.c_attn = nn.Linear(config['n_embd'], 3 * config['n_embd'])
        self.c_proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']
        self.n_embd = config['n_embd']
        self.register_buffer('bias', torch.tril(torch.ones(config['block_size'], config['block_size']))
                                   .view(1, 1, config['block_size'], config['block_size']))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config['n_embd'], 4 * config['n_embd'])
        self.c_proj = nn.Linear(4 * config['n_embd'], config['n_embd'])
    
    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config['n_embd'])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config['vocab_size'], config['n_embd']),
            wpe = nn.Embedding(config['block_size'], config['n_embd']),
            h = nn.ModuleList([Block(config) for _ in range(config['n_layer'])]),
            ln_f = nn.LayerNorm(config['n_embd']),
        ))
        self.lm_head = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config['block_size'], f"Sequence length {T} > block size {self.config['block_size']}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# Load tokenizer to get vocab size
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer/tokenizer.json')
config['vocab_size'] = tok.get_vocab_size()

device = 'cuda'
model = GPT(config).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f'Model: {n_params/1e6:.1f}M params')
print(f'Config: {config}')"""))

    # Cell 7: Data Loader
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 7: DATA LOADER
# ============================================================
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        self.data = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + 1 + self.block_size]
        return x, y

tok = Tokenizer.from_file('tokenizer/tokenizer.json')
block_size = config['block_size']
batch_size = config['batch_size']

train_ds = TextDataset('data/train.txt', tok, block_size)
val_ds = TextDataset('data/val.txt', tok, block_size)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

print(f'Train batches: {len(train_loader)}')
print(f'Val batches: {len(val_loader)}')"""))

    # Cell 8: Training Loop
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 8: TRAINING LOOP
# ============================================================
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import os
import time

device = 'cuda'
model = model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
scaler = GradScaler()
writer = SummaryWriter(os.path.join(LOGS_DIR, f'run_{int(time.time())}'))

# Resume from checkpoint if exists
start_step = 0
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'ckpt_step_*.pt')))
if ckpt_files:
    latest = ckpt_files[-1]
    print(f'Resuming from {latest}')
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scaler.load_state_dict(ckpt['scaler'])
    start_step = ckpt['step']
    print(f'Resumed at step {start_step}')

max_steps = config['max_steps']
eval_interval = config.get('eval_interval', 500)
save_interval = config.get('save_interval', 1000)
grad_accum = config.get('grad_accum_steps', 1)

print(f'Training: {max_steps} steps, eval every {eval_interval}, save every {save_interval}')
print(f'Grad accumulation: {grad_accum}')

step = start_step
epoch = 0
while step < max_steps:
    epoch += 1
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        with autocast():
            logits, loss = model(x, y)
            loss = loss / grad_accum
        
        scaler.scale(loss).backward()
        
        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        if step % 10 == 0:
            writer.add_scalar('train/loss', loss.item() * grad_accum, step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], step)
            if step % 50 == 0:
                print(f'Step {step}/{max_steps} | Loss: {loss.item() * grad_accum:.4f} | LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Evaluation
        if step % eval_interval == 0 and step > 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_loader:
                    vx, vy = vx.to(device), vy.to(device)
                    with autocast():
                        _, vloss = model(vx, vy)
                    val_losses.append(vloss.item())
            avg_val = sum(val_losses) / len(val_losses)
            writer.add_scalar('val/loss', avg_val, step)
            print(f'  >> Val loss: {avg_val:.4f}')
            model.train()
        
        # Save checkpoint
        if step % save_interval == 0 and step > 0:
            ckpt_path = os.path.join(CKPT_DIR, f'ckpt_step_{step}.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'step': step,
                'config': config,
            }, ckpt_path)
            print(f'  >> Saved checkpoint: {ckpt_path}')
        
        step += 1
        if step >= max_steps:
            break

# Final save
final_path = os.path.join(CKPT_DIR, f'ckpt_step_{step}.pt')
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scaler': scaler.state_dict(),
    'step': step,
    'config': config,
}, final_path)
print(f'Training complete. Final checkpoint: {final_path}')
writer.close()"""))

    # Cell 9: Generate Samples
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 9: GENERATE SAMPLES
# ============================================================
import torch
from tokenizers import Tokenizer

device = 'cuda'
model.eval()
tok = Tokenizer.from_file('tokenizer/tokenizer.json')

def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        idx_cond = ids[:, -config['block_size']:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

print(generate('Once upon a time', max_new_tokens=200))
print('---')
print(generate('The little boy', max_new_tokens=200))
print('---')
print(generate('In a magical forest', max_new_tokens=200))"""))

    # Clean up cell metadata
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.metadata = {}
            cell.outputs = []
            cell.execution_count = None

    return nb


def build_and_save():
    """Build and save the notebook."""
    nb = build_notebook()
    
    # Validate
    from nbformat import validate
    errors = validate(nb)
    if errors:
        print(f"Validation errors: {errors}")
        return False
    
    print("Notebook validation passed!")
    
    # Write
    output_path = Path(__file__).parent / 'kaggle_train_production.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook saved to: {output_path}")
    return True


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    success = build_and_save()
    sys.exit(0 if success else 1)