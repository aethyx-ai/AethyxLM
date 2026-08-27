"""Generate the single-T4 Colab notebook used to resume AethyxLM v3."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "colab_train_v3_resume.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


notebook = nbf.v4.new_notebook()
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        "# AethyxLM v3 — resume pretraining on Colab T4\n\n"
        "Downloads the private 8B tokenized corpus and latest persisted Kaggle "
        "checkpoint, preserves the original 32-sequence global batch using one T4, "
        "and backs up each 1,000-step checkpoint to the private Kaggle Dataset."
    ),
    code(
        """
        import subprocess, sys
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "kagglehub>=0.3", "tokenizers>=0.13", "datasets>=2.14",
            "tensorboard>=2.14", "tqdm>=4.65", "pyyaml>=6",
        ], check=True)
        import kagglehub
        kagglehub.login()
        print("[OK] Kaggle authentication ready")
        """
    ),
    code(
        """
        from pathlib import Path
        import json, os, re, signal, subprocess, sys, time
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError(f"Expected one Colab GPU; CUDA devices={torch.cuda.device_count()}")
        props = torch.cuda.get_device_properties(0)
        print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}")
        print(f"GPU: {props.name}, {props.total_memory / 2**30:.1f} GiB")

        WORK_ROOT = Path("/content/aethyxlm-v3")
        REPO_ROOT = WORK_ROOT / "repo"
        PROJECT_ROOT = REPO_ROOT / "AethyxLM"
        OUTPUT_ROOT = WORK_ROOT / "output"
        CHECKPOINT_ROOT = OUTPUT_ROOT / "checkpoints"
        LOG_ROOT = OUTPUT_ROOT / "logs"
        CONFIG_ROOT = OUTPUT_ROOT / "configs"
        for path in (CHECKPOINT_ROOT, LOG_ROOT, CONFIG_ROOT, CHECKPOINT_ROOT / "milestones"):
            path.mkdir(parents=True, exist_ok=True)

        REPO_URL = "https://github.com/aethyx-ai/AethyxLM.git"
        if (REPO_ROOT / ".git").is_dir():
            subprocess.run(["git", "-C", str(REPO_ROOT), "pull", "--ff-only"], check=True)
        else:
            subprocess.run(["git", "clone", REPO_URL, str(REPO_ROOT)], check=True)
        os.chdir(PROJECT_ROOT)
        # The latest public branch may predate cross-world-size RNG restoration.
        # Apply the same narrow compatibility fix used in the local workspace.
        trainer_path = PROJECT_ROOT / "training/trainer.py"
        trainer_lines = trainer_path.read_text(encoding="utf-8").splitlines()
        old_rng_line = next(
            (index for index, line in enumerate(trainer_lines)
             if "torch.cuda.set_rng_state_all(" in line),
            None,
        )
        if old_rng_line is not None:
            indent = trainer_lines[old_rng_line][
                : len(trainer_lines[old_rng_line])
                - len(trainer_lines[old_rng_line].lstrip())
            ]
            trainer_lines[old_rng_line : old_rng_line + 3] = [
                indent + 'saved_cuda_states = checkpoint["rng_state"]["cuda"]',
                indent + "for device_index, state in enumerate(",
                indent + "    saved_cuda_states[: torch.cuda.device_count()]",
                indent + "):",
                indent + "    torch.cuda.set_rng_state(state.cpu(), device=device_index)",
            ]
            trainer_path.write_text(
                "\\n".join(trainer_lines) + "\\n", encoding="utf-8"
            )
        os.environ.update({
            "CUDA_VISIBLE_DEVICES": "0", "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "2",
        })
        print("[OK] Project:", PROJECT_ROOT)
        """
    ),
    code(
        """
        # Download latest private data/checkpoint Dataset versions.
        DATASET_ROOT = Path(kagglehub.dataset_download("aethyx/aethyxlm-v3-8b-tokenized"))
        REMOTE_CHECKPOINT_ROOT = Path(kagglehub.dataset_download("aethyx/aethyxlm-v3-live-checkpoints"))
        print("[OK] Tokenized data:", DATASET_ROOT)
        print("[OK] Remote checkpoints:", REMOTE_CHECKPOINT_ROOT)
        """
    ),
    code(
        """
        expected_registry = json.loads((PROJECT_ROOT / "configs/datasets_v3_8b.json").read_text())

        def attached(root, name):
            matches = [p for p in root.rglob(name) if p.is_file() and p.stat().st_size > 0]
            return max(matches, key=lambda p: p.stat().st_size) if matches else None

        runtime_registry = {}
        missing = []
        for dataset_name, entry in expected_registry.items():
            runtime_entry = {"weight": entry["weight"]}
            for split in ("train", "val"):
                filename = Path(str(entry[split]).replace(chr(92), "/")).name
                binary = attached(DATASET_ROOT, filename)
                sidecar = attached(DATASET_ROOT, filename + ".meta.json")
                if binary is None or sidecar is None or sidecar.parent != binary.parent:
                    missing.append(filename)
                else:
                    runtime_entry[split] = str(binary)
            runtime_registry[dataset_name] = runtime_entry
        if missing:
            raise FileNotFoundError(f"Missing binaries/sidecars ({len(missing)}): {sorted(set(missing))}")

        runtime_registry_path = CONFIG_ROOT / "datasets_v3_8b_colab.json"
        runtime_registry_path.write_text(json.dumps(runtime_registry, indent=2) + "\\n")
        print(f"[OK] Resolved {len(runtime_registry)} dataset sources")
        """
    ),
    code(
        """
        # One T4 x batch 2 x accumulation 16 = original 32 sequences/update.
        config = json.loads((PROJECT_ROOT / "configs/train_config_v3_2xt4.json").read_text())
        config["training"].update({
            "batch_size": 2, "grad_accum_steps": 16, "planned_world_size": 1,
        })
        config["data"].update({
            "datasets_file": str(runtime_registry_path), "batch_size": 2,
            "num_workers": 2, "shuffle": False,
        })
        config["checkpoint"].update({
            "checkpoint_dir": str(CHECKPOINT_ROOT),
            "milestone_dir": str(CHECKPOINT_ROOT / "milestones"),
            "metrics_file": str(LOG_ROOT / "metrics.jsonl"),
            "log_dir": str(LOG_ROOT),
            "tensorboard_dir": str(LOG_ROOT / "tensorboard"),
            "save_interval": 1000,
            "backup": {
                "enabled": True, "provider": "kaggle_dataset",
                "handle": "aethyx/aethyxlm-v3-live-checkpoints",
                "required": True, "retries": 3,
            },
        })
        runtime_config = CONFIG_ROOT / "train_config_v3_colab_runtime.json"
        runtime_config.write_text(json.dumps(config, indent=2) + "\\n")

        candidates = []
        for path in REMOTE_CHECKPOINT_ROOT.rglob("checkpoint_step_*.pt"):
            match = re.fullmatch(r"checkpoint_step_(\d+)\.pt", path.name)
            if match and path.stat().st_size > 10 * 2**20:
                candidates.append((int(match.group(1)), path))
        if not candidates:
            raise FileNotFoundError("No numbered checkpoint found in the live checkpoint Dataset")
        resume_step, resume_path = max(candidates, key=lambda item: item[0])
        print(f"[OK] Resuming checkpoint step {resume_step:,}: {resume_path}")

        subprocess.run([
            sys.executable, "scripts/check_training_readiness.py", "--config", str(runtime_config)
        ], cwd=PROJECT_ROOT, check=True)
        """
    ),
    code(
        """
        command = [
            sys.executable, "train.py", "--config", str(runtime_config),
            "--device", "cuda", "--resume", str(resume_path),
        ]
        print("Running:", " ".join(command))
        started = time.time()
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, env=os.environ.copy(), start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
            return_code = process.wait()
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGINT)
            return_code = process.wait(timeout=180)
        print(f"Exit={return_code}; elapsed={(time.time() - started) / 3600:.2f}h")
        if return_code not in (0, 130, -signal.SIGINT):
            raise RuntimeError(f"Training failed with exit code {return_code}")
        """
    ),
]
notebook["metadata"] = {
    "accelerator": "GPU",
    "colab": {"gpuType": "T4", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
}
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
