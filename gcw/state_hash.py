from __future__ import annotations

import hashlib
import json

import torch


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and exact bytes with a stable V1 encoding."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
