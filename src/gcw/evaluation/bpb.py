from __future__ import annotations

from pathlib import Path
import torch

from gcw.checkpoint import load_checkpoint
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, get_token_bytes
from gcw.evaluation.scoring import evaluate_nll_bpb
from gcw.performance.losses import checkpoint_loss_backend


def run_fineweb(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    eval_tokens: int = 10 * 524288,
    batch_size: int = 16,
    *,
    dtype: torch.dtype | str | None = "auto",
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, state = load_checkpoint(checkpoint, device=device, dtype=dtype)
    if cfg.family != "smollm" or cfg.vocab_size != COSMO2_VOCAB_SIZE:
        raise ValueError(
            "FineWeb NLL/BPB eval currently expects a cosmo2 SmolLM checkpoint"
        )
    steps = max(1, eval_tokens // (batch_size * cfg.sequence_len))
    batches = data_loader(
        batch_size,
        cfg.sequence_len,
        split="val",
        device=device,
        data_dir=data_dir,
        infinite=True,
    )
    loss_backend = checkpoint_loss_backend(state, torch.device(device))
    nll, bpb = evaluate_nll_bpb(
        model,
        batches,
        steps,
        get_token_bytes(device),
        loss_backend=loss_backend,
    )
    return {
        "model_step": state.get("step"),
        "val_nll": nll,
        "val_bpb": bpb,
        "eval_tokens": steps * batch_size * cfg.sequence_len,
        "batch_size": batch_size,
        "sequence_len": cfg.sequence_len,
    }
