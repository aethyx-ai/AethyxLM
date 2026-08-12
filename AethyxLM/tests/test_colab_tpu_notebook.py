import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "colab_train_production.ipynb"


def test_colab_notebook_targets_single_v5e_tpu():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", ()))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

    assert "--device', 'xla" in code
    assert "PJRT_DEVICE'] = 'TPU'" in code
    assert "amp_dtype'] = 'bfloat16'" in code
    assert "fused_optimizer'] = False" in code
    assert "train_config_colab_v5e1.json" in code
    assert "--device', 'cuda" not in code


def test_all_colab_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
