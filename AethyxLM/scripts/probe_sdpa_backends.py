"""Probe which PyTorch SDPA kernels actually execute on the current CUDA GPU."""

import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def probe(backend):
    query = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16, requires_grad=True)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    try:
        with sdpa_kernel(backend):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )
            output.sum().backward()
        torch.cuda.synchronize()
        return {"available": True}
    except (RuntimeError, NotImplementedError) as error:
        return {"available": False, "reason": str(error).splitlines()[0]}


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    result = {
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "backends": {
            name: probe(backend)
            for name, backend in (
                ("flash_attention", SDPBackend.FLASH_ATTENTION),
                ("efficient_attention", SDPBackend.EFFICIENT_ATTENTION),
                ("cudnn_attention", SDPBackend.CUDNN_ATTENTION),
                ("math", SDPBackend.MATH),
            )
        },
    }
    Path("sdpa_backend_probe.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
