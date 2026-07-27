#!/usr/bin/env python3
"""
Build Kaggle training notebook programmatically using nbformat.
Optimized for VSCode connected to Kaggle Jupyter Server.
"""

import nbformat as nbf
from pathlib import Path


def build_notebook():
    nb = nbf.v4.new_notebook()

    # Set explicit Python 3 notebook metadata
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3 (ipykernel)',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'codemirror_mode': {
                'name': 'ipython',
                'version': 3
            },
            'file_extension': '.py',
            'mimetype': 'text/x-python',
            'name': 'python',
            'nbconvert_exporter': 'python',
            'pygments_lexer': 'ipython3',
            'version': '3.10.12'
        }
    }

    # Cell 0: Markdown - Title
    nb.cells.append(nbf.v4.new_markdown_cell("""# AethyxLM - Production Kaggle Training (VSCode + Kaggle Jupyter Server)

**Architecture:** Decoder-only GPT (14M params, 8L, 256D, 8H, 128ctx, 32k vocab)
**Dataset:** TinyStories (auto-download from Hugging Face)
**Environment:** VSCode connected to Kaggle Remote Jupyter Server
**Storage:** Git = code, `/kaggle/working` = persistent checkpoints/logs/data/configs

---"""))

    # Cell 1: Setup
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 1: SETUP PROJECT (Kaggle / VSCode Remote Jupyter)
# ============================================================
import os, sys, subprocess, shutil, json, time, glob, signal
from pathlib import Path

WORK_DIR = '/kaggle/working'
if os.path.exists(WORK_DIR):
    os.chdir(WORK_DIR)
else:
    WORK_DIR = os.getcwd()

# Project root detection
if os.path.exists(os.path.join(WORK_DIR, 'AethyxLM')):
    LOCAL_ROOT = os.path.join(WORK_DIR, 'AethyxLM')
elif os.path.exists('model/gpt.py'):
    LOCAL_ROOT = os.getcwd()
else:
    LOCAL_ROOT = os.path.join(WORK_DIR, 'AethyxLM')

# Persistent directories on Kaggle working dir
CKPT_DIR = os.path.join(WORK_DIR, 'checkpoints')
LOGS_DIR = os.path.join(WORK_DIR, 'logs')
TOK_DIR = os.path.join(WORK_DIR, 'tokenizer')
DATA_DIR = os.path.join(WORK_DIR, 'data')
CONFIG_DIR = os.path.join(WORK_DIR, 'configs')

for d in [CKPT_DIR, LOGS_DIR, TOK_DIR, DATA_DIR, CONFIG_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'[OK] Working dir: {WORK_DIR}')
print(f'[OK] Checkpoints: {CKPT_DIR}')
print(f'[OK] Logs: {LOGS_DIR}')
print(f'[OK] Configs: {CONFIG_DIR}')

# Clone or update Git repository
REPO_URL = 'https://github.com/aethyx-ai/AethyxLM.git'

if os.path.exists(os.path.join(LOCAL_ROOT, '.git')):
    print('Existing repository found. Syncing updates safely...')
    subprocess.run(['git', '-C', LOCAL_ROOT, 'pull'], check=False)
elif not os.path.exists(os.path.join(LOCAL_ROOT, 'model', 'gpt.py')):
    print('Cloning repository...')
    subprocess.run(['git', 'clone', REPO_URL, LOCAL_ROOT], check=True)

# Fix nested directory if created during clone
nested = os.path.join(LOCAL_ROOT, 'AethyxLM')
if os.path.exists(nested) and os.path.isdir(nested):
    for item in os.listdir(nested):
        src = os.path.join(nested, item)
        dst = os.path.join(LOCAL_ROOT, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, LOCAL_ROOT)
    try:
        os.rmdir(nested)
    except Exception:
        pass

if os.path.exists(LOCAL_ROOT):
    os.chdir(LOCAL_ROOT)
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

print(f'[OK] Project root: {os.getcwd()}')

# Install required dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', 'tokenizers', 'datasets', 'tensorboard', '-q'], check=True)
print('[OK] Dependencies installed.')"""))

    # Cell 2: Verify CUDA
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 2: VERIFY CUDA ACCELERATOR
# ============================================================
import torch

print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    raise RuntimeError('CUDA GPU not available! Enable GPU accelerator in Kaggle settings (Settings -> Accelerator -> GPU T4 x2 or P100)')

device_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'GPU: {device_name} ({vram_gb:.1f} GB VRAM)')

known_gpus = ['T4', 'L4', 'A100', 'V100', 'P100', 'K80', 'A10G']
if not any(g in device_name for g in known_gpus):
    print(f"Notice: GPU '{device_name}' detected. Proceeding with training...")"""))

    # Cell 3: Prepare Data
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 3: PREPARE DATA (TinyStories from Hugging Face)
# ============================================================
import os, sys, random, shutil
from pathlib import Path
from datasets import load_dataset

def safe_copy(src, dst):
    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)

os.makedirs('data', exist_ok=True)
os.makedirs('tokenizer/data', exist_ok=True)

if os.path.exists('data/train.txt') and os.path.getsize('data/train.txt') > 1000:
    print('[OK] Local dataset data/train.txt already exists. Skipping download.')
elif os.path.exists(os.path.join(DATA_DIR, 'train.txt')) and os.path.getsize(os.path.join(DATA_DIR, 'train.txt')) > 1000:
    print('[OK] Restoring dataset from persistent storage...')
    safe_copy(os.path.join(DATA_DIR, 'train.txt'), 'data/train.txt')
    safe_copy(os.path.join(DATA_DIR, 'val.txt'), 'data/val.txt')
    safe_copy(os.path.join(TOK_DIR, 'corpus.txt'), 'tokenizer/data/corpus.txt')
else:
    print('Downloading TinyStories dataset from Hugging Face...')
    ds = load_dataset('roneneldan/TinyStories', split='train')

    NUM_STORIES = 50000  # Adjust as needed
    texts = [item['text'] for item in ds.select(range(min(NUM_STORIES, len(ds))))]

    random.seed(42)
    random.shuffle(texts)
    split = int(0.95 * len(texts))
    train_texts = texts[:split]
    val_texts = texts[split:]

    train_content = '\\n\\n'.join(train_texts)
    val_content = '\\n\\n'.join(val_texts)

    with open('data/train.txt', 'w', encoding='utf-8') as f:
        f.write(train_content)
    with open('data/val.txt', 'w', encoding='utf-8') as f:
        f.write(val_content)

    with open('tokenizer/data/corpus.txt', 'w', encoding='utf-8') as f:
        f.write(train_content)

    print(f'Train: {len(train_texts)} stories saved to data/train.txt')
    print(f'Val: {len(val_texts)} stories saved to data/val.txt')

    # Persistent storage backup
    safe_copy('data/train.txt', os.path.join(DATA_DIR, 'train.txt'))
    safe_copy('data/val.txt', os.path.join(DATA_DIR, 'val.txt'))
    safe_copy('tokenizer/data/corpus.txt', os.path.join(TOK_DIR, 'corpus.txt'))

print('[OK] Dataset ready.')"""))

    # Cell 4: Train Tokenizer
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 4: TRAIN TOKENIZER (BPE, 32k vocab)
# ============================================================
import os, sys, subprocess, shutil

def safe_copy2(src, dst):
    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)

if os.path.exists('tokenizer/tokenizer.json') and os.path.getsize('tokenizer/tokenizer.json') > 1000:
    print('[OK] Local tokenizer.json already exists. Skipping training.')
elif os.path.exists(os.path.join(TOK_DIR, 'tokenizer.json')) and os.path.getsize(os.path.join(TOK_DIR, 'tokenizer.json')) > 1000:
    print('[OK] Restoring tokenizer from persistent storage...')
    os.makedirs('tokenizer', exist_ok=True)
    safe_copy2(os.path.join(TOK_DIR, 'tokenizer.json'), 'tokenizer/tokenizer.json')
    if os.path.exists(os.path.join(TOK_DIR, 'metadata.json')):
        safe_copy2(os.path.join(TOK_DIR, 'metadata.json'), 'tokenizer/metadata.json')
else:
    print('Training AethyxTokenizer via tokenizer.train_tokenizer...')
    result = subprocess.run(
        [sys.executable, '-m', 'tokenizer.train_tokenizer'],
        cwd=os.getcwd(),
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr)
        raise RuntimeError('Tokenizer training failed!')

    safe_copy2('tokenizer/tokenizer.json', os.path.join(TOK_DIR, 'tokenizer.json'))
    safe_copy2('tokenizer/metadata.json', os.path.join(TOK_DIR, 'metadata.json'))

from tokenizer.tokenizer import AethyxTokenizer
tok = AethyxTokenizer()
print(f'[OK] Tokenizer loaded successfully. Vocab size: {tok.vocab_size}')"""))

    # Cell 5: Load Config
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 5: LOAD & CREATE KAGGLE TRAINING CONFIG
# ============================================================
import os, sys, json, shutil

def safe_copy2(src, dst):
    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)

base_config_path = 'configs/train_config.json'
if not os.path.exists(base_config_path):
    base_config_path = 'configs/train_config_kaggle.json'

with open(base_config_path, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# Optimize hyper-parameters for Kaggle GPU session
cfg['training'].update({
    'learning_rate': 3e-4,
    'weight_decay': 0.1,
    'warmup_steps': 1000,
    'max_steps': 20000,
    'batch_size': 32,
    'grad_accum_steps': 1,
    'use_amp': True,
    'log_interval': 50,
    'eval_interval': 500,
    'save_interval': 1000,
    'generate_interval': 500
})
cfg['data'].update({
    'batch_size': 32,
    'num_workers': 2,
    'train_file': 'data/train.txt',
    'val_file': 'data/val.txt'
})

os.makedirs('configs', exist_ok=True)
kaggle_config_path = 'configs/train_config_kaggle.json'
with open(kaggle_config_path, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)

safe_copy2(kaggle_config_path, os.path.join(CONFIG_DIR, 'train_config_kaggle.json'))

print('[OK] Kaggle config written to configs/train_config_kaggle.json:')
print(json.dumps(cfg, indent=2))"""))

    # Cell 6: Auto-Resume Checkpoint Detection
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 6: AUTO-RESUME CHECKPOINT DETECTION
# ============================================================
import os, glob

def find_latest_checkpoint():
    \"\"\"Find latest valid checkpoint by modification time.\"\"\"
    candidates = []
    
    for base in [CKPT_DIR, 'checkpoints']:
        if os.path.exists(base):
            for f in os.listdir(base):
                if f.endswith('.pt'):
                    path = os.path.join(base, f)
                    if os.path.getsize(path) > 1_000_000:
                        candidates.append((os.path.getmtime(path), path))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None

resume_path = find_latest_checkpoint()
if resume_path:
    print(f'[OK] Found existing checkpoint for auto-resume: {resume_path}')
    RESUME_ARGS = ['--resume', resume_path]
else:
    print('[OK] No previous checkpoint found. Starting fresh training run.')
    RESUME_ARGS = []"""))

    # Cell 7: Persistent Sync Manager
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 7: SYNC CHECKPOINTS + LOGS + CONFIG (LOCAL <-> PERSISTENT)
# ============================================================
import os, shutil

def safe_copy2(src, dst):
    if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(src, dst)

def sync_to_persistent():
    \"\"\"Copy local checkpoints, logs, and config to persistent Kaggle working dir.\"\"\"
    if os.path.exists('checkpoints'):
        for f in os.listdir('checkpoints'):
            if f.endswith('.pt'):
                try:
                    safe_copy2(os.path.join('checkpoints', f), os.path.join(CKPT_DIR, f))
                except Exception as e:
                    print(f'  Checkpoint sync failed for {f}: {e}')

    if os.path.exists('logs'):
        for root, dirs, files in os.walk('logs'):
            rel_path = os.path.relpath(root, 'logs')
            target_dir = os.path.join(LOGS_DIR, rel_path) if rel_path != '.' else LOGS_DIR
            os.makedirs(target_dir, exist_ok=True)
            for f in files:
                try:
                    safe_copy2(os.path.join(root, f), os.path.join(target_dir, f))
                except Exception as e:
                    print(f'  Log sync failed for {f}: {e}')

    kaggle_cfg = 'configs/train_config_kaggle.json'
    if os.path.exists(kaggle_cfg):
        try:
            safe_copy2(kaggle_cfg, os.path.join(CONFIG_DIR, 'train_config_kaggle.json'))
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
            if os.path.abspath(src) != os.path.abspath(dst):
                if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                    try:
                        shutil.copy2(src, dst)
                        print(f'  Synced from persistent: {f}')
                    except Exception as e:
                        print(f'  Sync failed for {f}: {e}')

# Perform initial sync from persistent storage
sync_from_persistent()
print('[OK] Sync manager ready.')"""))

    # Cell 8: Real-Time Streaming Training Execution
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 8: TRAINING WRAPPER WITH REAL-TIME VSCODE STREAMING
# ============================================================
import torch, subprocess, threading, time, sys, os

print('Starting training on GPU:', torch.cuda.get_device_name(0))
print('=' * 60)

cmd = [
    sys.executable, 'train.py',
    '--config', 'configs/train_config_kaggle.json',
    '--device', 'cuda'
]

if RESUME_ARGS:
    cmd.extend(RESUME_ARGS)

print(f'Command: {" ".join(cmd)}')
print('-' * 60)

stop_sync = False
sync_lock = threading.Lock()

def periodic_sync():
    while not stop_sync:
        time.sleep(300)  # Sync every 5 minutes
        if not stop_sync:
            with sync_lock:
                sync_to_persistent()

sync_thread = threading.Thread(target=periodic_sync, daemon=True)
sync_thread.start()

start_time = time.time()
try:
    process = subprocess.Popen(
        cmd,
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
        
    process.wait()
    retcode = process.returncode
finally:
    stop_sync = True
    sync_thread.join(timeout=10)
    sync_to_persistent()
    elapsed = time.time() - start_time
    print('=' * 60)
    print(f'Training run completed in {elapsed/3600:.2f} hours.')

if retcode == 0:
    print('[OK] Training completed successfully!')
else:
    print(f'[FAIL] Process exited with code {retcode}')"""))

    # Cell 9: Download Links
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 9: DOWNLOAD CHECKPOINTS & LOGS
# ============================================================
import os

try:
    from IPython.display import FileLink, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

print("Checkpoints & Log Files available:")
ckpt_list = []
if os.path.exists('checkpoints'):
    for f in sorted(os.listdir('checkpoints')):
        if f.endswith('.pt'):
            ckpt_list.append(os.path.join('checkpoints', f))

for f in ckpt_list:
    size_mb = os.path.getsize(f) / (1024 * 1024)
    print(f'File: {f} ({size_mb:.1f} MB)')
    if HAS_IPYTHON:
        display(FileLink(f))

if os.path.exists('logs'):
    for root, _, files in os.walk('logs'):
        for f in files:
            path = os.path.join(root, f)
            print(f'Log: {path}')
            if HAS_IPYTHON:
                display(FileLink(path))"""))

    # Cell 10: Inference Test
    nb.cells.append(nbf.v4.new_code_cell("""# ============================================================
# CELL 10: INFERENCE & TEXT GENERATION TEST
# ============================================================
import os, torch
from model.gpt import GPT
from tokenizer.tokenizer import AethyxTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        safe = text.encode('ascii', 'replace').decode('ascii')
        print(safe)

ckpt_path = find_latest_checkpoint()

if not ckpt_path:
    print("No valid checkpoint found for generation test.")
else:
    print(f"Loading checkpoint for generation: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model = GPT().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    tok = AethyxTokenizer()
    
    @torch.no_grad()
    def generate(prompt, max_new=150, temp=0.8, top_k=50):
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
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

    prompts = [
        "Once upon a time",
        "The little girl",
        "In a small kingdom"
    ]
    
    for p in prompts:
        safe_print(f"\\nPrompt: {p}")
        print("-" * 40)
        output = generate(p, max_new=100)
        safe_print(output)
        print("=" * 60)"""))

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
    
    # Synchronize to kaggle_train.ipynb
    kaggle_train_path = Path(__file__).parent / 'kaggle_train.ipynb'
    with open(kaggle_train_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Notebook synchronized to: {kaggle_train_path}")

    return True


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    success = build_and_save()
    sys.exit(0 if success else 1)