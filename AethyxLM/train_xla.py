"""All-core PyTorch/XLA launcher for AethyxLM training."""

import os
import sys


def _worker(_index: int, forwarded: list[str]):
    # XLA devices must only be acquired below torch_xla.launch().
    sys.argv = ["train.py", *forwarded, "--device", "xla"]
    from train import main

    main()


def main():
    forwarded = sys.argv[1:]
    # Kaggle sets these to single-host placeholders (for example, "local").
    # PJRT multiprocessing mistakes them for an incomplete multi-host topology.
    os.environ.pop("TPU_PROCESS_ADDRESSES", None)
    os.environ.pop("CLOUD_TPU_TASK_ID", None)
    os.environ.setdefault("PJRT_DEVICE", "TPU")

    import torch_xla

    # The supported launcher discovers and starts every addressable TPU worker.
    # Nothing may acquire an XLA device before this call.
    torch_xla.launch(_worker, args=(forwarded,))


if __name__ == "__main__":
    main()
