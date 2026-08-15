import json
from pathlib import Path

import nbformat

from scripts.prepare_dataset_bundle import (
    remaining_output_bytes,
    source_target_tokens,
    validate_bundle,
    write_dataset_registry,
)
from scripts.source_filters import is_high_quality_code, normalize_text
from tokenizer.tokenizer import AethyxTokenizer


ROOT = Path(__file__).resolve().parents[1]


def load_bundle():
    manifest = json.loads((ROOT / "configs/pretrain_8b_sources.json").read_text())
    return manifest["aethyxlm_v3_48k_8b"]


def test_v3_manifest_is_exactly_eight_billion_tokens():
    bundle = load_bundle()
    validate_bundle(bundle)
    assert sum(source_target_tokens(source) for source in bundle["sources"]) == 8_000_000_000
    assert abs(sum(source["weight"] for source in bundle["sources"]) - 1.0) < 1e-12
    names = " ".join(source["name"] for source in bundle["sources"])
    assert "stack" not in names
    assert "finewiki" not in names
    assert "python_edu" not in names


def test_resume_disk_preflight_credits_existing_binary_bytes(tmp_path):
    bundle = {
        "sources": [
            {"name": "one", "target_tokens": 100},
            {"name": "two", "target_tokens": 50},
        ]
    }
    (tmp_path / "one_train.bin").write_bytes(b"x" * 120)
    (tmp_path / "one_val.bin").write_bytes(b"x" * 20)
    (tmp_path / "two_train.bin").write_bytes(b"x" * 200)  # capped at its target
    remaining, existing = remaining_output_bytes(bundle, tmp_path)
    assert existing == 240
    assert remaining == 60


def test_dataset_registry_paths_are_portable(tmp_path):
    registry_path = tmp_path / "datasets.json"
    bundle = {"sources": [{"name": "sample", "weight": 1.0}]}
    write_dataset_registry(bundle, registry_path, Path("data/v3_8b"))
    registry = json.loads(registry_path.read_text())
    assert registry["sample"]["train"] == "data/v3_8b/sample_train.bin"
    assert "\\" not in registry["sample"]["train"]


def test_frozen_v3_tokenizer_and_heldout_benchmark():
    tokenizer = AethyxTokenizer(ROOT / "tokenizer/tokenizer_v3_48k.json")
    assert tokenizer.vocab_size == 48_000
    report = json.loads((ROOT / "tokenizer/tokenizer_v3_48k_evaluation.json").read_text())
    assert report["v3_token_reduction_vs_v2"] > 0.30
    aggregate = report["tokenizers"]["v3_48k"]["aggregate"]
    assert aggregate["unknown_token_rate"] < 1e-5
    assert aggregate["normalized_roundtrip_rate"] > 0.999


def test_code_mix_covers_every_requested_language_and_totals_1_2b():
    sources = [source for source in load_bundle()["sources"] if source["name"].startswith("code_")]
    assert {source["dataset_config"] for source in sources} == {
        "py", "js", "ts", "html", "css", "java", "c", "cpp", "go", "rs", "sql", "sh"
    }
    assert sum(source["target_tokens"] for source in sources) == 1_200_000_000
    assert all(source["preserve_formatting"] for source in sources)
    assert all(source["code_quality_filter"] for source in sources)
    assert all("license" in source["required_values"] for source in sources)


def test_code_normalization_preserves_python_indentation():
    sample = "def f():\r\n\tif True:\r\n        return 1"
    normalized = normalize_text(sample, preserve_formatting=True)
    assert "\tif True" in normalized
    assert "        return 1" in normalized
    assert "\r" not in normalized
    assert normalize_text("a   b\n\n\n\nc") == "a b\n\nc"


def test_code_quality_filter_rejects_vendor_generated_and_secrets():
    valid = "def add(a, b):\n    # Add two values.\n    return a + b\n"
    assert is_high_quality_code({"path": "src/math.py"}, valid)
    assert not is_high_quality_code({"path": "node_modules/pkg/index.js"}, valid)
    assert not is_high_quality_code({"path": "src/generated.py"}, "# @generated\n" + valid)
    assert not is_high_quality_code(
        {"path": "src/config.py"}, valid + "\nKEY='AKIAABCDEFGHIJKLMNOP'\n"
    )


def test_dual_t4_config_plans_the_declared_token_budget():
    config = json.loads((ROOT / "configs/train_config_v3_2xt4.json").read_text())
    assert config["model"]["vocab_size"] == 48_000
    assert config["tokenizer"]["vocab_size"] == 48_000
    assert config["checkpoint"]["save_interval"] == 1_000
    sequences = (
        config["data"]["batch_size"]
        * config["training"]["grad_accum_steps"]
        * config["training"]["planned_world_size"]
    )
    schedule = config["training"]["context_schedule"]
    total = 0
    for index, stage in enumerate(schedule):
        end = schedule[index + 1]["step"] if index + 1 < len(schedule) else config["training"]["max_steps"]
        total += (end - stage["step"]) * sequences * stage["context_length"]
    assert total == config["training"]["planned_tokens"] == 8_000_012_288


def test_generated_kaggle_notebooks_are_valid_and_use_two_process_ddp():
    training = nbformat.read(ROOT / "kaggle_train_production.ipynb", as_version=4)
    preparation = nbformat.read(ROOT / "kaggle_prepare_8b.ipynb", as_version=4)
    nbformat.validate(training)
    nbformat.validate(preparation)
    for notebook in (training, preparation):
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"notebook-cell-{index}", "exec")
    training_text = "\n".join("".join(cell.source) for cell in training.cells)
    preparation_text = "\n".join("".join(cell.source) for cell in preparation.cells)
    assert "--nproc_per_node=2" in training_text
    assert "train_config_v3_2xt4.json" in training_text
    assert "save_interval': 1000" in training_text
    assert "prepare_dataset_bundle.py" in preparation_text
    assert "8_000_000_000" in preparation_text
