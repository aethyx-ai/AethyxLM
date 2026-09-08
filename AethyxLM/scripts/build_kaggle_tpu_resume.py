"""Build the self-contained Kaggle TPU resume notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "kaggle_train_production.ipynb"
OUTPUT_NOTEBOOK = ROOT / "kaggle_train_tpu_resume.ipynb"


def cell_source(text: str) -> list[str]:
    return [line + "\n" for line in text.rstrip().splitlines()]


def embedded_sources() -> dict[str, str]:
    return {
        name: base64.b64encode((ROOT / name).read_bytes()).decode("ascii")
        for name in (
            "train.py",
            "train_xla.py",
            "training/trainer.py",
            "model/attention.py",
        )
    }


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    payload = json.dumps(embedded_sources(), separators=(",", ":"))

    notebook["cells"][0]["source"] = cell_source(
        """# AethyxLM v3 — production resume on Kaggle TPU

This notebook resumes the **48K-vocabulary, ~138M-parameter** v3 run on every
core of a Kaggle TPU VM. It keeps the original global batch of 32, restores the
CUDA-produced checkpoint on CPU before moving optimizer state to XLA, saves
portable CPU checkpoints every 1,000 steps, and backs them up persistently."""
    )

    notebook["cells"][1]["source"] = cell_source(
        f"""from pathlib import Path
import base64, gc, json, os, re, signal, subprocess, sys, time

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle.')
os.environ.update({{
    'PJRT_DEVICE': 'TPU', 'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '1',
}})
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
    raise RuntimeError(f'{{REPO_ROOT}} exists but is not a Git checkout.')
else:
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_ROOT)], check=True)

# Install the tested TPU implementation into this private runtime without
# requiring an intermediate public repository commit.
embedded = json.loads({payload!r})
for relative, encoded in embedded.items():
    destination = PROJECT_ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(encoded))

os.chdir(PROJECT_ROOT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'tokenizers>=0.13', 'datasets>=2.14', 'kagglehub>=0.3',
                'tensorboard>=2.14', 'tqdm>=4.65', 'pyyaml>=6'], check=True)
print('[OK] Project:', PROJECT_ROOT)"""
    )

    notebook["cells"][2]["source"] = cell_source(
        """import importlib.util
if importlib.util.find_spec('torch_xla') is None:
    raise RuntimeError(
        'torch_xla is unavailable. This Kaggle run was not launched with a TPU accelerator.'
    )

# Do not import torch_xla or query TPU devices here. PyTorch/XLA requires the
# training launcher to be the first process that initializes the TPU runtime.
print('torch_xla package found; TPU initialization deferred to train_xla.py')"""
    )

    notebook["cells"][4]["source"] = cell_source(
        """# Preserve the CUDA run's effective global batch: 1 * 8 * 4 = 32.
PER_CORE_BATCH = 1
GRAD_ACCUM_STEPS = 4
EXPECTED_TPU_CORES = 8
if PER_CORE_BATCH * EXPECTED_TPU_CORES * GRAD_ACCUM_STEPS != 32:
    raise ValueError('The effective global batch must remain 32.')
config = json.loads((PROJECT_ROOT / 'configs/train_config_v3_2xt4.json').read_text())
config['model']['gradient_checkpointing'] = False
config['training'].update({
    'batch_size': PER_CORE_BATCH, 'grad_accum_steps': GRAD_ACCUM_STEPS,
    'planned_world_size': EXPECTED_TPU_CORES,
    'amp_dtype': 'bfloat16', 'torch_compile': False, 'log_interval': 10,
})
config['data'].update({
    'datasets_file': str(runtime_registry_path), 'batch_size': PER_CORE_BATCH,
    'num_workers': 0, 'shuffle': False,
})
config['checkpoint'].update({
    'checkpoint_dir': str(CHECKPOINT_ROOT),
    'milestone_dir': str(CHECKPOINT_ROOT / 'milestones'),
    'metrics_file': str(LOG_ROOT / 'metrics.jsonl'),
    'log_dir': str(LOG_ROOT), 'tensorboard_dir': str(LOG_ROOT / 'tensorboard'),
    'save_interval': 1000, 'log_interval': 10,
    'backup': {
        'enabled': True, 'provider': 'kaggle_dataset',
        'handle': 'aethyx/aethyxlm-v3-live-checkpoints',
        'required': True, 'retries': 3,
    },
})
runtime_config = CONFIG_ROOT / 'train_config_v3_tpu_runtime.json'
runtime_config.write_text(json.dumps(config, indent=2) + '\\n')
print('[OK] Runtime config:', runtime_config)"""
    )

    notebook["cells"][6]["source"] = cell_source(
        """command = [
    sys.executable, 'train_xla.py', '--config', str(runtime_config),
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
    return_code = process.wait(timeout=300)
print(f'Exit={return_code}; elapsed={(time.time() - started) / 3600:.2f}h')
if return_code not in (0, 130, -signal.SIGINT):
    raise RuntimeError(f'Training failed with exit code {return_code}.')"""
    )

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    OUTPUT_NOTEBOOK.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT_NOTEBOOK)


if __name__ == "__main__":
    main()
