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
    
    # Helper to add code cells
    def add_code(source):
        cell = nbf.v4.new_code_cell(source)
        cell.metadata = {}
        cell.outputs = []
        cell.execution_count = None
        return cell
    
    # Cell 0: Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell("""# AethyxLM - Production Kaggle Training (T4 GPU x2)

**Architecture:** 14M params, 8L, 256D, 8H, 128ctx, 32k vocab
**Dataset:** TinyStories (auto-download from Hugging Face)
**Storage:** GitHub = code, Kaggle Working = checkpoints/logs, Kaggle GPU = compute

---"""))
    
    # Cell 1: Setup
    cell1 = """# ============================================================
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
!pip install tokenizers datasets tensorboard -q"""
    
    # Cell 2: Verify CUDA
    cell2 = """# ============================================================
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
    print(f"Warning: GPU '{device_name}' not in common Kaggle types. Proceeding anyway...")"""
    
    # Cell 3: Prepare Data
    cell3 = """# ============================================================
# CELL 3: PREPARE DATA (TinyStories from Hugging Face)
# ============================================================
import random

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
shutil.copy(os.path.join(DATA_DIR, 'val.txt'), os.path.join(WORK_DIR, 'val.txt'))

print(f'Train: {len(train_texts)} stories')
print(f'Val: {len(val_texts)} stories')"""
    
    # Build notebook
    nb = nbf.v4.new_notebook()
    
    # Set metadata
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
!pip install tokenizers datasets tensorboard -q""")
    
    # Build notebook
    nb = nbf.v4.new_notebook()
    
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
!pip install tokenizers datasets tensorboard -q""")
    
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
    print(f"Warning: GPU '{device_name}' not in common Kaggle types. Proceeding anyway...")""")
    
    # Cell 3: Prepare Data
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 3: PREPARE DATA (TinyStories from Hugging Face)
# ============================================================
import random

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
shutil.copy(os.path.join(DATA_DIR, 'val.txt'), os.path.join(WORK_DIR, 'val.txt'))

print(f'Train: {len(train_texts)} stories')
print(f'Val: {len(val_texts)} stories')""")
    
    # Cell 4: Train BPE Tokenizer
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 4: TRAIN BPE TOKENIZER (32k vocab) + SAVE TO PERSISTENT
# ============================================================
import subprocess

print('Training tokenizer...')
result = subprocess.run(
    [sys.executable, '-m', 'tokenizer.train_tokenizer'],
    cwd=LOCAL_ROOT, capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr)
    raise RuntimeError('Tokenizer training failed')

# Verify
sys.path.insert(0, LOCAL_ROOT)
from tokenizer.tokenizer import AethyxTokenizer
tok = AethyxTokenizer()
print(f'[OK] Vocab size: {tok.vocab_size}')
ids = tok.encode('Hello world')
print(f'[OK] Encode: {ids}')
print(f'[OK] Decode: {tok.decode(ids)}')

# Copy tokenizer to persistent storage
shutil.copy2('tokenizer/tokenizer.json', os.path.join(WORK_DIR, 'tokenizer.json'))
shutil.copy2('tokenizer/metadata.json', os.path.join(WORK_DIR, 'metadata.json'))
print('[OK] Tokenizer saved to persistent storage')""")
    
    # Cell 5: Config
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 5: CONFIG FOR KAGGLE T4 x2
# ============================================================
import json

with open('configs/train_config.json') as f:
    cfg = json.load(f)

# T4 x2 optimized settings
cfg['training'].update({
    'max_steps': 100000,
    'warmup_steps': 10000,
    'batch_size': 64,
    'grad_accum_steps': 1,
    'use_amp': True,
    'eval_interval': 1000,
    'save_interval': 1000,
    'log_interval': 100,
    'learning_rate': 6e-4,
    'grad_clip': 1.0,
    'weight_decay': 0.1,
    'min_lr_ratio': 0.1,
    'generate_interval': 1000,
})

with open('configs/train_config_kaggle.json', 'w') as f:
    json.dump(cfg, f, indent=2)

# Backup both to persistent storage
shutil.copy2('configs/train_config.json', '/kaggle/working/configs/train_config.json')
shutil.copy2('configs/train_config_kaggle.json', '/kaggle/working/configs/train_config_kaggle.json')

print('[OK] Config written to:')
for k, v in cfg['training'].items():
    print(f'  {k}: {v}')""")
    
    # Cell 6: Auto-resume
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 6: AUTO-RESUME FROM PERSISTENT CHECKPOINT
# ============================================================
import glob

def find_latest_checkpoint():
    '''Find latest valid checkpoint in persistent storage or local.'''
    candidates = [
        '/kaggle/working/checkpoints/checkpoint_latest.pt',
        'checkpoints/checkpoint_latest.pt',
    ]
    for base in ['/kaggle/working/checkpoints', 'checkpoints']:
        if os.path.exists(base):
            steps = sorted(glob.glob(os.path.join(base, 'checkpoint_step_*.pt')))
            if steps:
                candidates.append(steps[-1])

    for c in candidates:
        if os.path.exists(c):
            return c
    return None

resume_path = find_latest_checkpoint()
if resume_path:
    print(f'[OK] Found checkpoint: {resume_path}')
    RESUME_ARGS = ['--resume', resume_path]
else:
    print('[OK] No checkpoint found, starting fresh')
    RESUME_ARGS = []""")
    
    # Cell 7: Sync functions
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 7: SYNC CHECKPOINTS + LOGS + CONFIG (LOCAL <-> PERSISTENT)
# ============================================================
def sync_to_persistent():
    \"\"\"Copy local checkpoints, logs, config to persistent WORK_DIR.\"\"\"
    # Checkpoints
    if os.path.exists('checkpoints'):
        for f in os.listdir('checkpoints'):
            if f.endswith('.pt'):
                try:
                    shutil.copy2(os.path.join('checkpoints', f),
                               os.path.join(CKPT_DIR, f))
                except Exception as e:
                    print(f'  Sync failed for {f}: {e}')

    # Logs
    if os.path.exists('logs'):
        for f in os.listdir('logs'):
            try:
                shutil.copy2(os.path.join('logs', f),
                           os.path.join(LOGS_DIR, f))
            except Exception as e:
                print(f'  Log sync failed for {f}: {e}')

    # Config (kaggle version)
    colab_cfg = 'configs/train_config_kaggle.json'
    if os.path.exists(colab_cfg):
        try:
            shutil.copy2(colab_cfg, os.path.join(CONFIG_DIR, 'train_config_kaggle.json'))
        except Exception as e:
            print(f'  Config sync failed: {e}')

def sync_from_persistent():
    \"\"\"Copy persistent checkpoints to local before training.\"\"\"
    if not os.path.exists(CKPT_DIR):
        return
    os.makedirs('checkpoints', exist_ok=True)
    for f in os.listdir(CKPT_DIR):
        if f.endswith('.pt'):
            src = os.path.join(CKPT_DIR, f)
            dst = os.path.join('checkpoints', f)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                try:
                    shutil.copy2(src, dst)
                    print(f'  Synced from persistent: {f}')
                except Exception as e:
                    print(f'  Sync failed for {f}: {e}')

# Initial sync from Drive
sync_from_persistent()
print('[OK] Sync ready (checkpoints + logs + config)')""")
    
    # Cell 8: Training wrapper
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 8: TRAINING WRAPPER WITH TRY/FINALLY + AUTO-SYNC
# ============================================================
import torch
import threading, time

print('Starting training on', torch.cuda.get_device_name(0))
print('=' * 60)

cmd = [sys.executable, 'train.py',
       '--config', 'configs/train_config_kaggle.json',
       '--device', 'cuda']

if RESUME_ARGS:
    cmd.extend(RESUME_ARGS)

print(f'Command: {" ".join(cmd)}')
print('-' * 60)

stop_sync = False
sync_lock = threading.Lock()

def periodic_sync():
    while not stop_sync:
        time.sleep(300)  # Every 5 minutes
        if not stop_sync:
            with sync_lock:
                sync_to_persistent()
                print(f'[{time.strftime("%H:%M:%S")}] Synced checkpoints + logs + config to persistent')

sync_thread = threading.Thread(target=periodic_sync, daemon=True)
sync_thread.start()

start = time.time()
try:
    result = subprocess.run(cmd, cwd=LOCAL_ROOT)
finally:
    # ALWAYS sync on exit (crash, interrupt, success, disconnect)
    stop_sync = True
    sync_thread.join(timeout=10)
    sync_to_persistent()
    elapsed = time.time() - start
    print('=' * 60)
    print(f'Training finished in {elapsed/3600:.1f}h')
    print(f'Exit code: {result.returncode}')

    if result.returncode == 0:
        print('[OK] Training completed successfully!')
    else:
        print(f'[FAIL] Training failed with code {result.returncode}')
        print('[INFO] You can resume from last checkpoint on next session')""")
    
    # Cell 9: Download
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 9: DOWNLOAD FINAL CHECKPOINTS + LOGS
# ============================================================
from IPython.display import FileLink, display
import os

ckpt_best = 'checkpoints/checkpoint_best.pt'
ckpt_latest = 'checkpoints/checkpoint_latest.pt'
ckpt_steps = sorted([f for f in os.listdir('checkpoints') 
                   if f.startswith('checkpoint_step_')])

for f in [ckpt_best, ckpt_latest] + ckpt_steps:
    if os.path.exists(f):
        print(f'Downloading: {f}')
        display(FileLink(f))
    else:
        print(f'Not found: {f}')

# Also download logs if they exist
if os.path.exists('logs'):
    for f in os.listdir('logs'):
        display(FileLink(os.path.join('logs', f)))""")
    
    # Cell 10: Inference
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 10: QUICK INFERENCE TEST
# ============================================================
import torch
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer

device = 'cuda'
model = GPT().to(device)
tok = AethyxTokenizer()

ckpt_path = 'checkpoints/checkpoint_best.pt'
if not os.path.exists(ckpt_path):
    ckpt_path = 'checkpoints/checkpoint_latest.pt'

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

@torch.no_grad()
def generate(prompt, max_new=200, temp=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    for _ in range(max_new):
        logits = model(ids[:, -128:])
        logits = logits[:, -1, :] / temp
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return tok.decode(ids[0].tolist())

print(generate('Once upon a time'))
print('---')
print(generate('The little boy'))
print('---')
print(generate('In a magical forest'))""")
    
    # Set metadata
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
    
    # Clean up cell metadata
    for cell in nb.cells:
        if cell.cell_type == 'code':
            cell.metadata = {}
            cell.outputs = []
            cell.execution_count = None
    
    return nb


def build_notebook():
    """Build and save the notebook."""
    nb = build_notebook()
    
    # Validate
    from nbformat.validator import validate
    
    print("Validating notebook...")
    errors = nbf.validator.validate(nb)
    if errors:
        print(f"Validation errors: {errors}")
        return False
    
    print("Notebook validation passed!")
    
    # Write
    output_path = Path(__file__).parent / 'kaggle_train_production.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook saved to: kaggle_train_production.ipynb")
    return True


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    success = build_notebook()
    sys.exit(0 if success else 1)