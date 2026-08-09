from pathlib import Path

import torch

from model.gpt import GPT
from training.trainer import Trainer


class StateStub:
    def state_dict(self):
        return {}


class TrainingModelStub:
    context_length = 8

    def train(self):
        return self


class TrainingBatches(list):
    batch_size = 2


def make_checkpoint_trainer(tmp_path: Path) -> Trainer:
    config = {
        "vocab_size": 32,
        "context_length": 8,
        "embed_dim": 16,
        "num_heads": 4,
        "num_layers": 1,
        "ffn_dim": 32,
        "dropout": 0.0,
        "normalization": "rmsnorm",
        "position_encoding": "rope",
        "ffn_type": "swiglu",
        "rope_max_seq_len": 16,
    }
    trainer = Trainer.__new__(Trainer)
    trainer.model = GPT(config=config)
    trainer.optimizer = StateStub()
    trainer.scheduler = StateStub()
    trainer.scaler = StateStub()
    trainer.checkpoint_dir = tmp_path / "checkpoints"
    trainer.checkpoint_dir.mkdir()
    trainer.save_interval = 1000
    trainer.step = 0
    trainer.epoch = 1
    trainer.best_val_loss = 1.25
    return trainer


def checkpoint_step(path: Path) -> int:
    return torch.load(path, map_location="cpu", weights_only=False)["step"]


def test_checkpoint_aliases_and_numbered_rotation(tmp_path):
    trainer = make_checkpoint_trainer(tmp_path)
    checkpoint_dir = trainer.checkpoint_dir

    trainer.step = 1000
    trainer.save_checkpoint()
    trainer.step = 1500
    trainer.save_checkpoint(is_best=True)

    assert checkpoint_step(checkpoint_dir / "checkpoint_best.pt") == 1500
    assert checkpoint_step(checkpoint_dir / "checkpoint_latest.pt") == 1000
    assert not (checkpoint_dir / "checkpoint_step_1500.pt").exists()

    trainer.step = 1750
    trainer.save_checkpoint(force=True)
    assert checkpoint_step(checkpoint_dir / "checkpoint_latest.pt") == 1750
    assert not (checkpoint_dir / "checkpoint_step_1750.pt").exists()

    for step in (2000, 3000, 4000, 5000):
        trainer.step = step
        trainer.save_checkpoint()

    numbered = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda path: checkpoint_step(path),
    )
    assert [checkpoint_step(path) for path in numbered] == [3000, 4000, 5000]
    assert checkpoint_step(checkpoint_dir / "checkpoint_latest.pt") == 5000
    assert checkpoint_step(checkpoint_dir / "checkpoint_best.pt") == 1500
    assert not list(checkpoint_dir.glob("*.tmp"))

    checkpoint = torch.load(numbered[-1], map_location="cpu", weights_only=False)
    assert checkpoint["config"]["model"]["normalization"] == "rmsnorm"
    assert checkpoint["config"]["model"]["position_encoding"] == "rope"
    assert checkpoint["config"]["model"]["ffn_type"] == "swiglu"


def test_training_loop_saves_completed_step_numbers(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.model = TrainingModelStub()
    trainer.train_dataloader = TrainingBatches([None, None, None])
    trainer.val_dataloader = None
    trainer.device = "cpu"
    trainer.max_steps = 3
    trainer.warmup_steps = 0
    trainer.grad_accum_steps = 1
    trainer.use_amp = False
    trainer.checkpoint_dir = tmp_path
    trainer.log_interval = 100
    trainer.eval_interval = 100
    trainer.save_interval = 2
    trainer.writer = None
    trainer.step = 0
    trainer.epoch = 0
    trainer.train_step = lambda _batch: 1.0
    trainer.optimizer_step = lambda: 0.0

    saves = []
    trainer._save_checkpoint = (
        lambda is_best=False, force=False: saves.append(
            (trainer.step, is_best, force)
        )
    )

    trainer.train()

    assert saves == [(2, False, False), (3, False, True)]


def test_training_steps_count_optimizer_updates_with_accumulation(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.model = TrainingModelStub()
    trainer.train_dataloader = TrainingBatches([None, None])
    trainer.val_dataloader = None
    trainer.device = "cpu"
    trainer.max_steps = 3
    trainer.warmup_steps = 0
    trainer.grad_accum_steps = 2
    trainer.use_amp = False
    trainer.checkpoint_dir = tmp_path
    trainer.log_interval = 100
    trainer.eval_interval = 100
    trainer.save_interval = 100
    trainer.writer = None
    trainer.step = 0
    trainer.epoch = 0
    microbatches = []
    updates = []
    trainer.train_step = lambda batch: microbatches.append(batch) or 1.0
    trainer.optimizer_step = lambda: updates.append(True) or 0.0
    trainer._save_checkpoint = lambda **_kwargs: None

    trainer.train()

    assert trainer.step == 3
    assert len(microbatches) == 6
    assert len(updates) == 3
