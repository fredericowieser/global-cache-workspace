from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from gcw.checkpoint import load_checkpoint
from gcw.config import EvalConfig
from gcw.data.shards import ShardDataset, data_loader, default_data_dir
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, get_token_bytes
from gcw.evaluation.scoring import evaluate_nll_bpb
from gcw.evaluation.update_metrics import CacheUpdateMetrics
from gcw.performance.losses import checkpoint_loss_backend


def run_fineweb(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    eval_tokens: int = 10 * 524288,
    config: EvalConfig | None = None,
    *,
    dtype: torch.dtype | str | None = "auto",
    sequence_len: int | None = None,
) -> dict:
    end_to_end_started = time.monotonic()
    config = config or EvalConfig(batch_size=16)
    if config.limit is not None:
        raise ValueError("FineWeb evaluation does not support a row limit")
    if config.apply_chat_template or config.enable_thinking:
        raise ValueError("FineWeb evaluation does not support chat formatting")
    batch_size = config.fixed_batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, state = load_checkpoint(checkpoint, device=device, dtype=dtype)
    sequence_len = cfg.sequence_len if sequence_len is None else sequence_len
    if sequence_len <= 0 or sequence_len > cfg.sequence_len or eval_tokens <= 0:
        raise ValueError(
            "FineWeb evaluation requires positive token budgets and a sequence length within model limits"
        )
    dataset = ShardDataset(data_dir or default_data_dir())
    saved_identity = state.get("fineweb_data_identity", {}).get("tokenizer")
    source = state.get("tokenizer_source") or state.get("source")
    tokenizer_name = state.get("tokenizer")
    if source and tokenizer_name in {None, "hf"}:
        tokenizer_name = f"hf:{source}"
    if dataset.manifest is None:
        if (
            tokenizer_name is None
            and cfg.family == "smollm"
            and cfg.vocab_size == COSMO2_VOCAB_SIZE
        ):
            tokenizer_name = "cosmo2"
        if tokenizer_name is None:
            raise ValueError(
                "Checkpoint does not identify a tokenizer for legacy FineWeb data"
            )
        dataset.validate_tokenizer(tokenizer_name, cfg.vocab_size)
    elif saved_identity is not None:
        current = dataset.manifest["tokenizer"]
        if any(
            saved_identity[key] != current[key]
            for key in ("sha256", "eos_token_id", "max_token_id")
        ):
            raise ValueError(
                "Checkpoint tokenizer differs from FineWeb manifest"
            )
        if current["max_token_id"] >= cfg.vocab_size:
            raise ValueError(
                "Model vocabulary cannot represent FineWeb token IDs"
            )
        tokenizer_name = current["source"]
    else:
        if tokenizer_name is None:
            raise ValueError(
                "Checkpoint does not identify a tokenizer; cannot establish FineWeb compatibility"
            )
        dataset.validate_tokenizer(tokenizer_name, cfg.vocab_size)
    steps = max(1, eval_tokens // (batch_size * sequence_len))
    batches = data_loader(
        batch_size,
        sequence_len,
        split="val",
        device=device,
        data_dir=data_dir,
        infinite=True,
    )
    loss_backend = checkpoint_loss_backend(state, torch.device(device))
    update_metrics = CacheUpdateMetrics(model, config.cache_update_metrics_path)
    execution_started = time.monotonic()
    nll, bpb = evaluate_nll_bpb(
        model,
        batches,
        steps,
        get_token_bytes(device) if tokenizer_name == "cosmo2" else None,
        loss_backend=loss_backend,
        update_metrics=update_metrics,
    )
    metrics_metadata = update_metrics.finish()
    return {
        **(
            {"cache_update_metrics": metrics_metadata}
            if metrics_metadata
            else {}
        ),
        "model_step": state.get("step"),
        "val_nll": nll,
        "val_perplexity": math.exp(nll) if nll < 709 else float("inf"),
        **({"val_bpb": bpb} if bpb is not None else {}),
        "dataset": {
            "path": str(dataset.path.resolve()),
            "manifest_sha256": dataset.manifest_sha256,
        },
        "tokenizer": (
            dataset.manifest["tokenizer"]
            if dataset.manifest
            else {"source": "cosmo2"}
        ),
        "eval_tokens": steps * batch_size * sequence_len,
        "batch_size": batch_size,
        "sequence_len": sequence_len,
        "elapsed_seconds": time.monotonic() - execution_started,
        "end_to_end_elapsed_seconds": time.monotonic() - end_to_end_started,
    }
