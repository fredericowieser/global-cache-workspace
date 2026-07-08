from __future__ import annotations

import math

import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate_nll_bpb(model, batches, steps: int, token_bytes: torch.Tensor) -> tuple[float, float]:
    device = model.get_device()
    total_nats = torch.tensor(0.0, device=device)
    total_bytes = torch.tensor(0, dtype=torch.long, device=device)
    total_tokens = torch.tensor(0, dtype=torch.long, device=device)
    it = iter(batches)
    model.eval()
    for _ in range(steps):
        x, y = next(it)[:2]
        loss = model(x, y, loss_reduction="none").view(-1)
        y = y.view(-1)
        valid = y >= 0
        safe = torch.where(valid, y, torch.zeros_like(y))
        counts = torch.where(valid, token_bytes[safe], torch.zeros_like(y, dtype=token_bytes.dtype))
        keep = counts > 0
        total_nats += (loss * keep).sum()
        total_bytes += counts.sum()
        total_tokens += keep.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_nats)
        dist.all_reduce(total_bytes)
        dist.all_reduce(total_tokens)
    nats = float(total_nats.item())
    bytes_ = int(total_bytes.item())
    tokens = int(total_tokens.item())
    return nats / tokens if tokens else float("inf"), nats / (math.log(2) * bytes_) if bytes_ else float("inf")
