#!/usr/bin/env python3
"""Generate the Kaggle data-preparation and dual-T4 training notebooks."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def notebook(title: str):
    title = title.replace("\u2014", "-").replace("\u00d7", "x")
    result = nbf.v4.new_notebook()
    result.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    result.cells.append(nbf.v4.new_markdown_cell(title))
    return result


def code(value: str):
    return nbf.v4.new_code_cell(value.strip() + "\n")


def build_prepare():
    nb = notebook("""# AethyxLM v3 — prepare the 8B-token corpus on Kaggle

Run this notebook with a **CPU runtime and Internet enabled**. It streams raw datasets, applies the committed filters, tokenizes with the frozen 48K tokenizer, and writes exactly 8B `uint16` tokens (~14.9 GiB). The preparation is durable per source and resumes from its state files.

After an interruption or completion, run the upload cell. On a later session, attach that private Kaggle Dataset and rerun from the top.""")
    nb.cells.extend([
        code("""
from pathlib import Path
import json, os, shutil, subprocess, sys

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle.')
REPO_URL = 'https://github.com/aethyx-ai/AethyxLM.git'
REPO_ROOT = Path('/kaggle/working/aethyxlm-v3-repo')
PROJECT_ROOT = REPO_ROOT / 'AethyxLM'
DATA_ROOT = Path('/kaggle/working/aethyxlm-v3-data')
DATASET_HANDLE = 'aethyx/aethyxlm-v3-8b-tokenized'
DATA_ROOT.mkdir(parents=True, exist_ok=True)

if (REPO_ROOT / '.git').is_dir():
    subprocess.run(['git', '-C', str(REPO_ROOT), 'pull', '--ff-only'], check=True)
elif REPO_ROOT.exists():
    raise RuntimeError(f'{REPO_ROOT} exists but is not a Git checkout.')
else:
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)
os.chdir(PROJECT_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'tokenizers>=0.13', 'datasets>=2.14', 'kagglehub>=0.3',
                'tensorboard>=2.14', 'tqdm>=4.65', 'pyyaml>=6'], check=True)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUTF8'] = '1'
print('[OK] Project:', PROJECT_ROOT)
print('[OK] Data output:', DATA_ROOT)
"""),
        code("""
# Restore partial/completed preparation from any attached Kaggle Dataset.
manifest = json.loads((PROJECT_ROOT / 'configs/pretrain_8b_sources.json').read_text())
bundle = manifest['aethyxlm_v3_48k_8b']
expected_names = set()
for source in bundle['sources']:
    prefix = source['name']
    expected_names.update({
        f'{prefix}_train.bin', f'{prefix}_val.bin',
        f'{prefix}_train.bin.meta.json', f'{prefix}_val.bin.meta.json',
        f'{prefix}_metadata.json', f'{prefix}_state.json',
    })
input_root = Path('/kaggle/input')
restored = 0
for name in sorted(expected_names):
    candidates = [p for p in input_root.rglob(name) if p.is_file()]
    if not candidates:
        continue
    source = max(candidates, key=lambda p: p.stat().st_mtime)
    destination = DATA_ROOT / name
    if not destination.exists() or destination.stat().st_size != source.stat().st_size:
        shutil.copy2(source, destination)
        restored += 1
print(f'[OK] Restored {restored} preparation artifacts from /kaggle/input.')
"""),
        code("""
# Validate the frozen tokenizer before spending hours on corpus preparation.
from tokenizer.tokenizer import AethyxTokenizer
tokenizer_path = PROJECT_ROOT / 'tokenizer/tokenizer_v3_48k.json'
tokenizer = AethyxTokenizer(tokenizer_path)
if tokenizer.vocab_size != 48000:
    raise RuntimeError(f'Expected 48,000 vocabulary entries, found {tokenizer.vocab_size}.')
print('[OK] Tokenizer v3:', tokenizer_path)
print('[OK] SHA-256:', tokenizer.sha256)
"""),
        code("""
# Prepare/resume all sources. Interrupting is safe after the current document flushes.
command = [
    sys.executable, 'scripts/prepare_dataset_bundle.py',
    '--manifest', 'configs/pretrain_8b_sources.json',
    '--bundle', 'aethyxlm_v3_48k_8b',
    '--output-dir', str(DATA_ROOT),
    '--buffer-tokens', '500000',
    '--progress-seconds', '30',
    '--registry-output', str(DATA_ROOT / 'datasets_v3_8b.json'),
]
print('Running:', ' '.join(command))
subprocess.run(command, cwd=PROJECT_ROOT, check=True)
"""),
        code("""
# Persist the current state as a private Kaggle Dataset version.
import kagglehub
notes = 'AethyxLM v3 8B corpus preparation state'
kagglehub.dataset_upload(DATASET_HANDLE, str(DATA_ROOT), version_notes=notes)
print('[OK] Uploaded:', DATASET_HANDLE)
"""),
        code("""
# Completion audit: exactly 8B tokens and every source complete.
total_tokens = 0
incomplete = []
for source in bundle['sources']:
    prefix = source['name']
    metadata_path = DATA_ROOT / f'{prefix}_metadata.json'
    if not metadata_path.is_file():
        incomplete.append(prefix)
        continue
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('status') != 'complete':
        incomplete.append(prefix)
        continue
    total_tokens += int(metadata['train_tokens']) + int(metadata['validation_tokens'])
print(f'Tokens: {total_tokens:,} / 8,000,000,000')
if incomplete:
    raise RuntimeError(f'Incomplete sources ({len(incomplete)}): {incomplete}')
if total_tokens != 8_000_000_000:
    raise RuntimeError('Prepared token total is not exactly 8B.')
print('[READY] Attach this private Dataset to kaggle_train_production.ipynb.')
"""),
    ])
    return nb


def build_train():
    nb = notebook("""# AethyxLM v3 — production pretraining on Kaggle 2×T4

This starts or resumes the **48K-vocabulary, ~138M-parameter** AethyxLM v3 run. The context curriculum and global batch are calculated to process 8,000,012,288 tokens (the nearest optimizer boundary above 8B). Checkpoints are saved every 1,000 optimizer steps and uploaded to a private Kaggle Dataset.

Before running, select **GPU T4 ×2** and attach the completed private `aethyxlm-v3-8b-tokenized` Dataset plus any checkpoint Dataset you want to resume.""")
    nb.cells.extend([
        code("""
from pathlib import Path
import gc, json, os, re, signal, subprocess, sys, time

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle.')
REPO_URL = 'https://github.com/aethyx-ai/AethyxLM.git'
REPO_ROOT = Path('/kaggle/working/aethyxlm-v3-repo')
PROJECT_ROOT = REPO_ROOT / 'AethyxLM'
OUTPUT_ROOT = Path('/kaggle/working/aethyxlm-v3-output')
CHECKPOINT_ROOT = OUTPUT_ROOT / 'checkpoints'
LOG_ROOT = OUTPUT_ROOT / 'logs'
CONFIG_ROOT = OUTPUT_ROOT / 'configs'
for path in (CHECKPOINT_ROOT, LOG_ROOT, CONFIG_ROOT, CHECKPOINT_ROOT / 'milestones'):
    path.mkdir(parents=True, exist_ok=True)

if (REPO_ROOT / '.git').is_dir():
    subprocess.run(['git', '-C', str(REPO_ROOT), 'pull', '--ff-only'], check=True)
elif REPO_ROOT.exists():
    raise RuntimeError(f'{REPO_ROOT} exists but is not a Git checkout.')
else:
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)
os.chdir(PROJECT_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'tokenizers>=0.13', 'datasets>=2.14', 'kagglehub>=0.3',
                'tensorboard>=2.14', 'tqdm>=4.65', 'pyyaml>=6'], check=True)
os.environ.update({
    'CUDA_VISIBLE_DEVICES': '0,1', 'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2', 'NCCL_DEBUG': 'WARN',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
})
print('[OK] Project:', PROJECT_ROOT)
"""),
        code("""
import torch
print(f'PyTorch {torch.__version__}; CUDA {torch.version.cuda}')
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f'Expected Kaggle GPU T4 x2; CUDA devices={torch.cuda.device_count()}')
if not torch.distributed.is_nccl_available():
    raise RuntimeError('NCCL is unavailable.')
for index in range(2):
    props = torch.cuda.get_device_properties(index)
    print(f'GPU {index}: {props.name}, {props.total_memory / 2**30:.1f} GiB')
"""),
        code("""
# Resolve all 72 binary splits and sidecars from attached Kaggle Datasets.
expected_registry = json.loads((PROJECT_ROOT / 'configs/datasets_v3_8b.json').read_text())
input_root = Path('/kaggle/input')

def attached(name):
    matches = [p for p in input_root.rglob(name) if p.is_file() and p.stat().st_size > 0]
    return max(matches, key=lambda p: p.stat().st_size) if matches else None

runtime_registry = {}
missing = []
for dataset_name, entry in expected_registry.items():
    runtime_entry = {'weight': entry['weight']}
    for split in ('train', 'val'):
        filename = Path(entry[split]).name
        binary = attached(filename)
        sidecar = attached(filename + '.meta.json')
        if binary is None or sidecar is None or sidecar.parent != binary.parent:
            missing.append(filename)
        else:
            runtime_entry[split] = str(binary)
    runtime_registry[dataset_name] = runtime_entry
if missing:
    raise FileNotFoundError(f'Missing binaries/sidecars ({len(missing)}): {sorted(set(missing))}')
runtime_registry_path = CONFIG_ROOT / 'datasets_v3_8b_kaggle.json'
runtime_registry_path.write_text(json.dumps(runtime_registry, indent=2) + '\\n')
print(f'[OK] Resolved {len(runtime_registry)} dataset sources.')
"""),
        code("""
# Build the runtime config. Batch 1 + accumulation 16 is an OOM fallback with the same token plan.
PER_GPU_BATCH = 2
GRAD_ACCUM_STEPS = 8
if PER_GPU_BATCH * GRAD_ACCUM_STEPS != 16:
    raise ValueError('Keep PER_GPU_BATCH * GRAD_ACCUM_STEPS == 16 to preserve the 8B schedule.')
config = json.loads((PROJECT_ROOT / 'configs/train_config_v3_2xt4.json').read_text())
config['training'].update({'batch_size': PER_GPU_BATCH, 'grad_accum_steps': GRAD_ACCUM_STEPS})
config['data'].update({
    'datasets_file': str(runtime_registry_path), 'batch_size': PER_GPU_BATCH,
    'num_workers': 2, 'shuffle': False,
})
config['checkpoint'].update({
    'checkpoint_dir': str(CHECKPOINT_ROOT),
    'milestone_dir': str(CHECKPOINT_ROOT / 'milestones'),
    'metrics_file': str(LOG_ROOT / 'metrics.jsonl'),
    'log_dir': str(LOG_ROOT), 'tensorboard_dir': str(LOG_ROOT / 'tensorboard'),
    'save_interval': 1000,
    'backup': {
        'enabled': True, 'provider': 'kaggle_dataset',
        'handle': 'aethyx/aethyxlm-v3-live-checkpoints',
        'required': True, 'retries': 3,
    },
})
runtime_config = CONFIG_ROOT / 'train_config_v3_2xt4_runtime.json'
runtime_config.write_text(json.dumps(config, indent=2) + '\\n')
print('[OK] Runtime config:', runtime_config)
"""),
        code(r"""
# Select the highest numbered compatible checkpoint, or start v3 from scratch.
RESUME_CHECKPOINT = None
roots = [CHECKPOINT_ROOT, Path('/kaggle/input')]
candidates = []
for root in roots:
    if root.exists():
        candidates.extend(root.rglob('checkpoint_step_*.pt'))
numbered = []
for path in candidates:
    match = re.fullmatch(r'checkpoint_step_(\d+)\.pt', path.name)
    if match and path.stat().st_size > 10 * 2**20:
        numbered.append((int(match.group(1)), path))
resume_path = Path(RESUME_CHECKPOINT) if RESUME_CHECKPOINT else (
    max(numbered, key=lambda item: item[0])[1] if numbered else None
)
resume_args = [] if resume_path is None else ['--resume', str(resume_path)]
print('[OK] Starting fresh v3 run.' if resume_path is None else f'[OK] Resuming: {resume_path}')

subprocess.run([sys.executable, 'scripts/check_training_readiness.py',
                '--config', str(runtime_config)], cwd=PROJECT_ROOT, check=True)
"""),
        code("""
# Launch true two-process DDP.
command = [
    sys.executable, '-m', 'torch.distributed.run', '--standalone',
    '--nproc_per_node=2', 'train.py', '--config', str(runtime_config), '--ddp',
] + resume_args
print('Running:', ' '.join(command))
started = time.time()
process = subprocess.Popen(
    command, cwd=PROJECT_ROOT, env=os.environ.copy(), start_new_session=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
try:
    for line in process.stdout:
        print(line, end='', flush=True)
    return_code = process.wait()
except KeyboardInterrupt:
    os.killpg(process.pid, signal.SIGINT)
    return_code = process.wait(timeout=180)
print(f'Exit={return_code}; elapsed={(time.time() - started) / 3600:.2f}h')
if return_code not in (0, 130, -signal.SIGINT):
    raise RuntimeError(f'Training failed with exit code {return_code}.')
"""),
        code("""
checkpoints = sorted(CHECKPOINT_ROOT.glob('*.pt'), key=lambda p: p.stat().st_mtime)
for path in checkpoints:
    print(f'{path.name:36s} {path.stat().st_size / 2**20:8.1f} MiB')
print('Metrics:', LOG_ROOT / 'metrics.jsonl')
print('Persistent backup: aethyx/aethyxlm-v3-live-checkpoints')
"""),
    ])
    return nb


def save(nb, path: Path):
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    save(build_prepare(), ROOT / "kaggle_prepare_8b.ipynb")
    training = build_train()
    save(training, ROOT / "kaggle_train_production.ipynb")
    save(training, ROOT / "kaggle_train_v3_2xt4.ipynb")
