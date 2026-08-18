from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import torch

from gcw.checkpoint import save_checkpoint
from gcw.config import CacheProcessorConfig
from gcw.convert.hf_config import ALIASES, config_from_hf
from gcw.convert.hf_state import load_hf_state
from gcw.models import build_model


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"unsupported dtype {dtype!r}; use auto, bfloat16, float16, or float32") from exc


def _infer_model_dtype(model: torch.nn.Module) -> torch.dtype:
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return torch.float32


def _load_hf_model(model_id: str, *, torch_dtype, family: str):
    from transformers import AutoModelForCausalLM

    multimodal_model_cls: Any | None = None
    if family in {"qwen3_5_moe", "gemma4"}:
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError:
            pass
        else:
            multimodal_model_cls = AutoModelForMultimodalLM
    if family == "qwen3_5_moe":
        if multimodal_model_cls is not None:
            return multimodal_model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=False)
    elif family == "gemma4":
        if multimodal_model_cls is not None:
            try:
                return multimodal_model_cls.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=False)
            except ValueError:
                pass
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, trust_remote_code=False)


def convert_hf(
    model_id: str,
    out: str | Path,
    family: str = "llama",
    variant: str = "tf",
    dtype: str = "bfloat16",
) -> None:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise ImportError("Install the convert extra: uv sync --extra convert") from exc

    model_id = ALIASES.get(model_id.lower(), model_id)
    hf_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    cfg = config_from_hf(hf_cfg, family=family, variant=variant)
    if variant in {"bt", "bgt", "icarus", "longshot"}:
        cfg = replace(cfg, cache_processor=CacheProcessorConfig())
    torch_dtype = _resolve_dtype(dtype)
    hf = _load_hf_model(model_id, torch_dtype=torch_dtype, family=family)
    model_dtype = _infer_model_dtype(hf)
    native = build_model(cfg).to(dtype=model_dtype)
    native.init_weights()
    load_hf_state(native, hf.state_dict(), family=family)
    save_checkpoint(
        out,
        native,
        cfg,
        {
            "source": model_id,
            "source_format": "hf",
            "family": family,
            "dtype": str(model_dtype).removeprefix("torch."),
        },
    )
