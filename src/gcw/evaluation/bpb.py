from __future__ import annotations

from pathlib import Path
import torch

from gcw.checkpoint import load_checkpoint
from gcw.config import EvalConfig
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, get_token_bytes
from gcw.evaluation.scoring import evaluate_nll_bpb
from gcw.performance.losses import checkpoint_loss_backend


def run_fineweb(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    eval_tokens: int = 10 * 524288,
    config: EvalConfig | None = None,
    *,
    dtype: torch.dtype | str | None = "auto",
) -> dict:
    config = config or EvalConfig(batch_size=16)
    if config.limit is not None:
        raise ValueError("FineWeb evaluation does not support a row limit")
    if config.apply_chat_template or config.enable_thinking:
        raise ValueError("FineWeb evaluation does not support chat formatting")
    batch_size = config.fixed_batch_size
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
