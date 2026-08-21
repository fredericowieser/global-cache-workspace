#!/usr/bin/env python3
"""Diagnose a failed HF→GCW conversion parity result without changing its gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F

from gcw.checkpoint import load_checkpoint
from gcw.models import TF
from gcw.provenance import file_sha256
from scripts.analysis.verify_hf_conversion import (
    CANONICAL_PROMPT,
    CHECKPOINT_PARITY_FILES,
    HF_PARITY_FILES,
    _atomic_write_new,
    _dtype_names,
    _file_manifest,
    _token_ids_sha256,
)

SCHEMA_VERSION = 1
EVIDENCE_KIND = "hf_to_gcw_conversion_diagnosis"
POLICY_ID = "qwen3-06b-layerwise-distribution-diagnosis-v1"
PARENT_PARITY_POLICY = "qwen3-06b-bf16-parameters-fp32-comparison-v2"
PARENT_PARITY_SHA256 = (
    "1aca29e6db3099d652d8635d57f8f79a9acccfbec915a378c0477e7104b86216"
)
REQUIRED_DTYPE = "bfloat16"
FIXED_PROMPTS: tuple[tuple[str, str], ...] = (
    ("parent_parity", CANONICAL_PROMPT),
    ("short", "Hello, world."),
    ("arithmetic", "Calculate 17 times 23, then state only the result."),
    (
        "reasoning",
        "If every glorp is blue and no blue object is square, can a glorp be square? Explain briefly.",
    ),
    ("unicode", "Résumé: naïve café — 你好，世界。"),
)
DISTRIBUTION_METRIC_KEYS = {
    "raw_max_abs_error",
    "raw_mean_abs_error",
    "offset_abs_max",
    "offset_abs_mean",
    "centred_max_abs_error",
    "centred_mean_abs_error",
    "logprob_max_abs_error",
    "logprob_mean_abs_error",
    "kl_hf_to_native_max",
    "kl_hf_to_native_mean",
    "kl_hf_to_native_min",
    "total_variation_max",
    "total_variation_mean",
    "top1_match_fraction",
    "final_top1_match",
}
HIDDEN_METRIC_KEYS = {
    "max_abs_error",
    "mean_abs_error",
    "relative_l2_error",
    "cosine_similarity",
    "hf_rms",
    "native_rms",
}


def _finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"non-finite {name}")


def distribution_metrics(
    hf_logits: torch.Tensor, native_logits: torch.Tensor
) -> dict[str, float]:
    """Compare [batch, tokens, vocabulary] logits in a common FP32 representation."""
    if hf_logits.shape != native_logits.shape or hf_logits.ndim != 3:
        raise ValueError(
            f"logits must have equal [batch,tokens,vocab] shapes: "
            f"hf={tuple(hf_logits.shape)}, native={tuple(native_logits.shape)}"
        )
    hf = hf_logits.float()
    native = native_logits.float()
    _finite(hf, "HF logits")
    _finite(native, "native logits")

    raw_delta = native - hf
    offsets = raw_delta.mean(dim=-1)
    hf_centred = hf - hf.mean(dim=-1, keepdim=True)
    native_centred = native - native.mean(dim=-1, keepdim=True)
    centred_delta = native_centred - hf_centred

    hf_probability_logits = hf.double()
    native_probability_logits = native.double()
    hf_logp = F.log_softmax(hf_probability_logits, dim=-1)
    native_logp = F.log_softmax(native_probability_logits, dim=-1)
    hf_prob = hf_logp.exp()
    native_prob = native_logp.exp()
    kl_raw = (hf_prob * (hf_logp - native_logp)).sum(dim=-1)
    minimum_kl = float(kl_raw.min().item())
    if minimum_kl < -1e-10:
        raise FloatingPointError(
            f"FP64 KL cancellation exceeded tolerance: {minimum_kl}"
        )
    kl = kl_raw.clamp_min(0.0)
    total_variation = 0.5 * (hf_prob - native_prob).abs().sum(dim=-1)
    logprob_delta = (native_logp - hf_logp).abs()
    for tensor, name in (
        (raw_delta, "raw logit delta"),
        (centred_delta, "centred logit delta"),
        (kl, "KL"),
        (total_variation, "total variation"),
        (logprob_delta, "log-probability delta"),
    ):
        _finite(tensor, name)

    return {
        "raw_max_abs_error": float(raw_delta.abs().max().item()),
        "raw_mean_abs_error": float(raw_delta.abs().mean().item()),
        "offset_abs_max": float(offsets.abs().max().item()),
        "offset_abs_mean": float(offsets.abs().mean().item()),
        "centred_max_abs_error": float(centred_delta.abs().max().item()),
        "centred_mean_abs_error": float(centred_delta.abs().mean().item()),
        "logprob_max_abs_error": float(logprob_delta.max().item()),
        "logprob_mean_abs_error": float(logprob_delta.mean().item()),
        "kl_hf_to_native_max": float(kl.max().item()),
        "kl_hf_to_native_mean": float(kl.mean().item()),
        "kl_hf_to_native_min": float(kl.min().item()),
        "total_variation_max": float(total_variation.max().item()),
        "total_variation_mean": float(total_variation.mean().item()),
        "top1_match_fraction": float(
            (hf.argmax(dim=-1) == native.argmax(dim=-1)).float().mean().item()
        ),
        "final_top1_match": float(
            (hf[:, -1].argmax(dim=-1) == native[:, -1].argmax(dim=-1))
            .float()
            .mean()
            .item()
        ),
    }


def hidden_metrics(
    hf_hidden: torch.Tensor, native_hidden: torch.Tensor
) -> dict[str, float]:
    """Compare aligned hidden tensors in FP32."""
    if hf_hidden.shape != native_hidden.shape:
        raise ValueError(
            f"hidden tensors must have equal shapes: "
            f"hf={tuple(hf_hidden.shape)}, native={tuple(native_hidden.shape)}"
        )
    hf = hf_hidden.float()
    native = native_hidden.float()
    _finite(hf, "HF hidden state")
    _finite(native, "native hidden state")
    delta = native - hf
    hf_flat = hf.reshape(-1).double()
    native_flat = native.reshape(-1).double()
    delta_flat = native_flat - hf_flat
    hf_norm = torch.linalg.vector_norm(hf_flat)
    relative = torch.linalg.vector_norm(delta_flat) / hf_norm.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    cosine_raw = F.cosine_similarity(hf_flat, native_flat, dim=0)
    if (
        float(cosine_raw.item()) < -1.0 - 1e-12
        or float(cosine_raw.item()) > 1.0 + 1e-12
    ):
        raise FloatingPointError(
            f"FP64 cosine outside mathematical range: {float(cosine_raw.item())}"
        )
    cosine = cosine_raw.clamp(-1.0, 1.0)
    return {
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "relative_l2_error": float(relative.item()),
        "cosine_similarity": float(cosine.item()),
        "hf_rms": float(hf.square().mean().sqrt().item()),
        "native_rms": float(native.square().mean().sqrt().item()),
    }


def _capture_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    hf_model: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str], str]:
    core = getattr(model, "model", None) if hf_model else model
    if (
        core is None
        or not hasattr(core, "embed_tokens")
        or not hasattr(core, "layers")
        or not hasattr(core, "norm")
    ):
        raise TypeError(
            f"unsupported model structure for layer capture: {type(model)!r}"
        )
    layers = list(core.layers)
    captures: dict[str, torch.Tensor] = {}
    dtypes: dict[str, str] = {}

    def hook(
        name: str,
    ) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        def capture(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"capture {name!r} did not produce a tensor")
            if name in captures:
                raise RuntimeError(f"capture {name!r} fired more than once")
            dtypes[name] = str(tensor.dtype).removeprefix("torch.")
            captures[name] = tensor.detach().float().cpu()

        return capture

    handles = [core.embed_tokens.register_forward_hook(hook("embedding"))]
    handles.extend(
        layer.register_forward_hook(hook(f"layer_{index:02d}"))
        for index, layer in enumerate(layers)
    )
    handles.append(core.norm.register_forward_hook(hook("final_norm")))
    try:
        with torch.inference_mode():
            if hf_model:
                output = model(input_ids=input_ids, use_cache=False).logits
            else:
                output = model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    expected = [
        "embedding",
        *(f"layer_{index:02d}" for index in range(len(layers))),
        "final_norm",
    ]
    if list(captures) != expected:
        raise RuntimeError(
            f"capture closure mismatch: got={list(captures)!r}, expected={expected!r}"
        )
    logits_dtype = str(output.dtype).removeprefix("torch.")
    return output.detach().float().cpu(), captures, dtypes, logits_dtype


def _checkpoint_model_config(checkpoint: Path) -> dict[str, object]:
    document = json.loads((checkpoint / "config.json").read_text())
    if not isinstance(document, dict) or set(document) != {"model"}:
        raise ValueError("checkpoint config document closure mismatch")
    model = document.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint model config is malformed")
    return model


def _source_revision(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _scriptfile_sha256() -> str:
    return file_sha256(Path(__file__).resolve(strict=True))


def _prompt_payload(tokenizer: Any) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for name, text in FIXED_PROMPTS:
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < 2:
            raise ValueError(
                f"diagnostic prompt {name!r} encoded to fewer than two tokens"
            )
        payload.append(
            {
                "name": name,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "token_ids": token_ids,
                "token_ids_sha256": _token_ids_sha256(token_ids),
                "tokens": len(token_ids),
            }
        )
    return payload


def _metric_number(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"diagnostic metric {key!r} is not numeric")
    return float(value)


def _require_range(
    row: dict[str, object],
    key: str,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = _metric_number(row, key)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(
            f"diagnostic metric {key!r} outside [{minimum}, {maximum}]: {value}"
        )
    return value


def _aggregate(
    per_prompt: list[dict[str, object]], layer_order: list[str]
) -> dict[str, object]:
    first_distribution = per_prompt[0].get("distribution")
    if not isinstance(first_distribution, dict):
        raise TypeError("diagnostic distribution row is malformed")
    distribution_names = list(first_distribution)
    distribution: dict[str, float] = {}
    for name in distribution_names:
        values: list[float] = []
        for item in per_prompt:
            row = item.get("distribution")
            if not isinstance(row, dict):
                raise TypeError("diagnostic distribution row is malformed")
            values.append(_metric_number(row, name))
        distribution[f"prompt_mean_{name}"] = sum(values) / len(values)
        distribution[f"prompt_max_{name}"] = max(values)
    layers: list[dict[str, object]] = []
    for name in layer_order:
        rows: list[dict[str, object]] = []
        for item in per_prompt:
            hidden = item.get("hidden_states")
            if not isinstance(hidden, dict) or not isinstance(
                hidden.get(name), dict
            ):
                raise TypeError("diagnostic hidden-state row is malformed")
            rows.append(hidden[name])
        layers.append(
            {
                "name": name,
                "prompt_mean_relative_l2_error": sum(
                    _metric_number(row, "relative_l2_error") for row in rows
                )
                / len(rows),
                "prompt_max_relative_l2_error": max(
                    _metric_number(row, "relative_l2_error") for row in rows
                ),
                "prompt_mean_mean_abs_error": sum(
                    _metric_number(row, "mean_abs_error") for row in rows
                )
                / len(rows),
                "prompt_max_max_abs_error": max(
                    _metric_number(row, "max_abs_error") for row in rows
                ),
                "prompt_mean_cosine_similarity": sum(
                    _metric_number(row, "cosine_similarity") for row in rows
                )
                / len(rows),
            }
        )
    return {"distribution": distribution, "hidden_states": layers}


def _validate_parent_binding(
    parent: dict[str, object],
    *,
    hf_path: Path,
    checkpoint: Path,
    hf_manifest: dict[str, dict[str, object]],
    checkpoint_manifest: dict[str, dict[str, object]],
) -> None:
    if (
        parent.get("passed") is not False
        or parent.get("policy_id") != PARENT_PARITY_POLICY
    ):
        raise ValueError(
            "parent evidence is not the approved failed parity result"
        )
    if Path(str(parent.get("hf_path"))).resolve(strict=True) != hf_path:
        raise ValueError("diagnostic HF path is not bound to parent parity")
    if Path(str(parent.get("checkpoint"))).resolve(strict=True) != checkpoint:
        raise ValueError(
            "diagnostic checkpoint path is not bound to parent parity"
        )
    if parent.get("hf_files") != hf_manifest:
        raise ValueError("diagnostic HF manifest differs from parent parity")
    if parent.get("checkpoint_files") != checkpoint_manifest:
        raise ValueError(
            "diagnostic checkpoint manifest differs from parent parity"
        )


def run_diagnosis(
    hf_path: Path,
    checkpoint: Path,
    *,
    source_revision: str,
    device: torch.device,
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_path = hf_path.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    parent_path = checkpoint / "parity.json"
    if file_sha256(parent_path) != PARENT_PARITY_SHA256:
        raise ValueError("parent failed-parity evidence hash mismatch")
    parent = json.loads(parent_path.read_text())
    if not isinstance(parent, dict):
        raise ValueError("parent parity evidence is malformed")

    hf_before = _file_manifest(hf_path, HF_PARITY_FILES)
    checkpoint_before = _file_manifest(checkpoint, CHECKPOINT_PARITY_FILES)
    _validate_parent_binding(
        parent,
        hf_path=hf_path,
        checkpoint=checkpoint,
        hf_manifest=hf_before,
        checkpoint_manifest=checkpoint_before,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=False)
    prompts = _prompt_payload(tokenizer)

    hf = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to(device)
    hf.eval()
    hf_parameter_dtypes = _dtype_names(hf.parameters())
    if hf_parameter_dtypes != [REQUIRED_DTYPE]:
        raise TypeError(
            f"HF parameters are not uniformly BF16: {hf_parameter_dtypes!r}"
        )
    hf_results: dict[
        str, tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, str], str]
    ] = {}
    for prompt in prompts:
        tokens = torch.tensor(
            [prompt["token_ids"]], dtype=torch.long, device=device
        )
        hf_results[str(prompt["name"])] = _capture_forward(
            hf, tokens, hf_model=True
        )
    del hf
    gc.collect()
    torch.cuda.empty_cache()

    native, config, state = load_checkpoint(checkpoint, device=device)
    if not isinstance(native, TF):
        raise TypeError(
            f"converted checkpoint is not a TF model: {type(native)!r}"
        )
    if state.get("dtype") != REQUIRED_DTYPE:
        raise TypeError("converted checkpoint state is not BF16")
    native_parameter_dtypes = _dtype_names(native.parameters())
    if native_parameter_dtypes != [REQUIRED_DTYPE]:
        raise TypeError(
            f"native parameters are not uniformly BF16: {native_parameter_dtypes!r}"
        )
    native.eval()

    per_prompt: list[dict[str, object]] = []
    layer_order: list[str] | None = None
    for prompt in prompts:
        name = str(prompt["name"])
        tokens = torch.tensor(
            [prompt["token_ids"]], dtype=torch.long, device=device
        )
        native_logits, native_hidden, native_dtypes, native_logits_dtype = (
            _capture_forward(native, tokens, hf_model=False)
        )
        hf_logits, hf_hidden, hf_dtypes, hf_logits_dtype = hf_results[name]
        if (
            hf_logits_dtype != REQUIRED_DTYPE
            or native_logits_dtype != "float32"
        ):
            raise TypeError(
                f"diagnostic raw-logit dtype mismatch for {name}: "
                f"hf={hf_logits_dtype!r}, native={native_logits_dtype!r}"
            )
        if list(hf_hidden) != list(native_hidden):
            raise RuntimeError(
                f"hidden-state stage mismatch for {name}: "
                f"hf={list(hf_hidden)!r}, native={list(native_hidden)!r}"
            )
        if layer_order is None:
            layer_order = list(hf_hidden)
        elif layer_order != list(hf_hidden):
            raise RuntimeError(
                "hidden-state capture order changed across prompts"
            )
        per_prompt.append(
            {
                **prompt,
                "distribution": distribution_metrics(hf_logits, native_logits),
                "hf_logits_dtype": hf_logits_dtype,
                "native_logits_dtype": native_logits_dtype,
                "hidden_states": {
                    stage: hidden_metrics(
                        hf_hidden[stage], native_hidden[stage]
                    )
                    for stage in layer_order
                },
                "hf_hidden_dtypes": hf_dtypes,
                "native_hidden_dtypes": native_dtypes,
            }
        )
    if layer_order is None:
        raise RuntimeError("no prompts were diagnosed")

    hf_after = _file_manifest(hf_path, HF_PARITY_FILES)
    checkpoint_after = _file_manifest(checkpoint, CHECKPOINT_PARITY_FILES)
    if hf_before != hf_after or checkpoint_before != checkpoint_after:
        raise RuntimeError(
            "model/tokenizer/checkpoint artifacts changed during diagnosis"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"diagnosis requires exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    gpu_name = torch.cuda.get_device_name(device)
    if "H100" not in gpu_name:
        raise RuntimeError(f"diagnosis requires an H100, got {gpu_name!r}")

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "policy_id": POLICY_ID,
        "status": "completed",
        "source_revision": source_revision,
        "script_sha256": _scriptfile_sha256(),
        "hf_path": str(hf_path),
        "checkpoint": str(checkpoint),
        "parent_parity_path": str(parent_path.resolve(strict=True)),
        "parent_parity_sha256": PARENT_PARITY_SHA256,
        "hf_files": hf_after,
        "checkpoint_files": checkpoint_after,
        "hf_parameter_dtypes": hf_parameter_dtypes,
        "native_parameter_dtypes": native_parameter_dtypes,
        "source_state": state,
        "model_config": asdict(config),
        "comparison_dtype": "float32",
        "prompts": per_prompt,
        "aggregate": _aggregate(per_prompt, layer_order),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": gpu_name,
        "cuda_device_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def validate_diagnostic_evidence(
    payload: dict[str, object],
    *,
    hf_path: Path,
    checkpoint: Path,
    source_revision: str,
) -> None:
    expected = {
        "schema_version",
        "kind",
        "policy_id",
        "status",
        "source_revision",
        "script_sha256",
        "hf_path",
        "checkpoint",
        "parent_parity_path",
        "parent_parity_sha256",
        "hf_files",
        "checkpoint_files",
        "hf_parameter_dtypes",
        "native_parameter_dtypes",
        "source_state",
        "model_config",
        "comparison_dtype",
        "prompts",
        "aggregate",
        "torch_version",
        "torch_cuda_version",
        "device",
        "gpu_name",
        "cuda_device_count",
        "slurm_job_id",
    }
    if set(payload) != expected:
        raise ValueError(
            f"diagnostic evidence field closure mismatch: {sorted(set(payload) ^ expected)!r}"
        )
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "policy_id": POLICY_ID,
        "status": "completed",
        "source_revision": source_revision,
        "script_sha256": _scriptfile_sha256(),
        "parent_parity_sha256": PARENT_PARITY_SHA256,
        "comparison_dtype": "float32",
    }
    for key, expected_value in fixed.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"diagnostic evidence {key} mismatch")
    hf_path = hf_path.resolve(strict=True)
    checkpoint = checkpoint.resolve(strict=True)
    if Path(str(payload["hf_path"])).resolve(strict=True) != hf_path:
        raise ValueError("diagnostic HF path mismatch")
    if Path(str(payload["checkpoint"])).resolve(strict=True) != checkpoint:
        raise ValueError("diagnostic checkpoint path mismatch")
    if payload["hf_files"] != _file_manifest(hf_path, HF_PARITY_FILES):
        raise ValueError("diagnostic HF manifest mismatch")
    if payload["checkpoint_files"] != _file_manifest(
        checkpoint, CHECKPOINT_PARITY_FILES
    ):
        raise ValueError("diagnostic checkpoint manifest mismatch")
    if payload["parent_parity_path"] != str(
        (checkpoint / "parity.json").resolve(strict=True)
    ):
        raise ValueError("diagnostic parent parity path mismatch")
    if file_sha256(checkpoint / "parity.json") != PARENT_PARITY_SHA256:
        raise ValueError("diagnostic parent parity bytes changed")
    parent = json.loads((checkpoint / "parity.json").read_text())
    if not isinstance(parent, dict):
        raise ValueError("diagnostic parent parity is malformed")
    _validate_parent_binding(
        parent,
        hf_path=hf_path,
        checkpoint=checkpoint,
        hf_manifest=cast(dict[str, dict[str, object]], payload["hf_files"]),
        checkpoint_manifest=cast(
            dict[str, dict[str, object]], payload["checkpoint_files"]
        ),
    )
    if payload.get("hf_parameter_dtypes") != [REQUIRED_DTYPE]:
        raise ValueError("diagnostic HF parameter dtype mismatch")
    if payload.get("native_parameter_dtypes") != [REQUIRED_DTYPE]:
        raise ValueError("diagnostic native parameter dtype mismatch")
    checkpoint_state = json.loads((checkpoint / "train_state.json").read_text())
    if payload.get("source_state") != checkpoint_state:
        raise ValueError("diagnostic source state mismatch")
    checkpoint_config = _checkpoint_model_config(checkpoint)
    if payload.get("model_config") != checkpoint_config:
        raise ValueError("diagnostic model config mismatch")
    num_layers = checkpoint_config.get("num_layers")
    if (
        isinstance(num_layers, bool)
        or not isinstance(num_layers, int)
        or num_layers <= 0
    ):
        raise ValueError("diagnostic model layer count is invalid")
    layer_order = [
        "embedding",
        *(f"layer_{index:02d}" for index in range(num_layers)),
        "final_norm",
    ]

    prompts = payload.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != len(FIXED_PROMPTS):
        raise ValueError("diagnostic prompt closure mismatch")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=False)
    expected_prompt_fields = {
        "name",
        "text",
        "text_sha256",
        "token_ids",
        "token_ids_sha256",
        "tokens",
        "distribution",
        "hf_logits_dtype",
        "native_logits_dtype",
        "hidden_states",
        "hf_hidden_dtypes",
        "native_hidden_dtypes",
    }
    for item, (name, text) in zip(prompts, FIXED_PROMPTS, strict=True):
        if not isinstance(item, dict) or set(item) != expected_prompt_fields:
            raise ValueError("diagnostic prompt field closure mismatch")
        if item.get("name") != name or item.get("text") != text:
            raise ValueError("diagnostic fixed prompt mismatch")
        if item.get("text_sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("diagnostic prompt text hash mismatch")
        token_ids = item.get("token_ids")
        if not isinstance(token_ids, list) or not all(
            type(value) is int for value in token_ids
        ):
            raise ValueError("diagnostic token IDs malformed")
        if token_ids != tokenizer.encode(text, add_special_tokens=True):
            raise ValueError("diagnostic tokenizer replay mismatch")
        if item.get("tokens") != len(token_ids):
            raise ValueError("diagnostic token count mismatch")
        if item.get("token_ids_sha256") != _token_ids_sha256(token_ids):
            raise ValueError("diagnostic token hash mismatch")
        if (
            item.get("hf_logits_dtype") != REQUIRED_DTYPE
            or item.get("native_logits_dtype") != "float32"
        ):
            raise ValueError("diagnostic raw-logit dtype mismatch")
        distribution = item.get("distribution")
        if (
            not isinstance(distribution, dict)
            or set(distribution) != DISTRIBUTION_METRIC_KEYS
        ):
            raise ValueError("diagnostic distribution metric closure mismatch")
        for key in DISTRIBUTION_METRIC_KEYS:
            if not math.isfinite(_metric_number(distribution, key)):
                raise ValueError(
                    f"diagnostic distribution metric {key!r} is non-finite"
                )
        for key in (
            "raw_max_abs_error",
            "raw_mean_abs_error",
            "offset_abs_max",
            "offset_abs_mean",
            "centred_max_abs_error",
            "centred_mean_abs_error",
            "logprob_max_abs_error",
            "logprob_mean_abs_error",
            "kl_hf_to_native_max",
            "kl_hf_to_native_mean",
            "kl_hf_to_native_min",
        ):
            _require_range(distribution, key, minimum=0.0)
        for key in (
            "total_variation_max",
            "total_variation_mean",
            "top1_match_fraction",
            "final_top1_match",
        ):
            _require_range(distribution, key, minimum=0.0, maximum=1.0)
        for mean_key, max_key in (
            ("raw_mean_abs_error", "raw_max_abs_error"),
            ("offset_abs_mean", "offset_abs_max"),
            ("centred_mean_abs_error", "centred_max_abs_error"),
            ("logprob_mean_abs_error", "logprob_max_abs_error"),
            ("kl_hf_to_native_mean", "kl_hf_to_native_max"),
            ("total_variation_mean", "total_variation_max"),
        ):
            if _metric_number(distribution, mean_key) > _metric_number(
                distribution, max_key
            ):
                raise ValueError(
                    f"diagnostic mean {mean_key!r} exceeds {max_key!r}"
                )
        if not (
            _metric_number(distribution, "kl_hf_to_native_min")
            <= _metric_number(distribution, "kl_hf_to_native_mean")
            <= _metric_number(distribution, "kl_hf_to_native_max")
        ):
            raise ValueError("diagnostic KL min/mean/max ordering mismatch")
        hidden_states = item.get("hidden_states")
        if not isinstance(hidden_states, dict) or set(hidden_states) != set(
            layer_order
        ):
            raise ValueError("diagnostic hidden-state stage closure mismatch")
        for stage in layer_order:
            row = hidden_states.get(stage)
            if not isinstance(row, dict) or set(row) != HIDDEN_METRIC_KEYS:
                raise ValueError(
                    f"diagnostic hidden metric closure mismatch at {stage}"
                )
            for key in HIDDEN_METRIC_KEYS:
                if not math.isfinite(_metric_number(row, key)):
                    raise ValueError(
                        f"diagnostic hidden metric {stage}.{key} is non-finite"
                    )
            for key in (
                "max_abs_error",
                "mean_abs_error",
                "relative_l2_error",
                "hf_rms",
                "native_rms",
            ):
                _require_range(row, key, minimum=0.0)
            _require_range(row, "cosine_similarity", minimum=-1.0, maximum=1.0)
            if _metric_number(row, "mean_abs_error") > _metric_number(
                row, "max_abs_error"
            ):
                raise ValueError(
                    f"diagnostic hidden mean exceeds max at {stage}"
                )
        for dtype_field in ("hf_hidden_dtypes", "native_hidden_dtypes"):
            dtypes = item.get(dtype_field)
            if not isinstance(dtypes, dict) or set(dtypes) != set(layer_order):
                raise ValueError(f"diagnostic {dtype_field} closure mismatch")
            if set(dtypes.values()) != {REQUIRED_DTYPE}:
                raise ValueError(
                    f"diagnostic {dtype_field} is not uniformly BF16"
                )
    recomputed_aggregate = _aggregate(prompts, layer_order)
    if payload.get("aggregate") != recomputed_aggregate:
        raise ValueError(
            "diagnostic aggregate does not match prompt-level evidence"
        )
    if payload.get("device") != "cuda" or payload.get("cuda_device_count") != 1:
        raise ValueError("diagnostic CUDA runtime metadata mismatch")
    if (
        payload.get("torch_version") != torch.__version__
        or payload.get("torch_cuda_version") != torch.version.cuda
    ):
        raise ValueError("diagnostic PyTorch/CUDA version mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "diagnostic validator requires the live one-GPU allocation"
        )
    gpu_name = payload.get("gpu_name")
    live_gpu_name = torch.cuda.get_device_name(0)
    if (
        not isinstance(gpu_name, str)
        or "H100" not in gpu_name
        or gpu_name != live_gpu_name
    ):
        raise ValueError("diagnostic GPU metadata does not match the live H100")
    slurm_job_id = payload.get("slurm_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
        raise ValueError("diagnostic Slurm job ID is malformed")
    json.dumps(payload, allow_nan=False)


def _failure_payload(
    *,
    hf_path: Path,
    checkpoint: Path,
    source_revision: str,
    error: BaseException,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "policy_id": POLICY_ID,
        "status": "failed",
        "source_revision": source_revision,
        "script_sha256": _scriptfile_sha256(),
        "hf_path": str(hf_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "parent_parity_sha256": PARENT_PARITY_SHA256,
        "error_type": type(error).__name__,
        "error": str(error),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("source revision must be a full lowercase Git SHA")
    repo = Path(__file__).resolve(strict=True).parents[2]
    if _source_revision(repo) != args.source_revision:
        raise ValueError("diagnostic source revision mismatch")
    if args.validate_only:
        payload = json.loads(args.output.read_text())
        validate_diagnostic_evidence(
            payload,
            hf_path=args.hf_path,
            checkpoint=args.checkpoint,
            source_revision=args.source_revision,
        )
        print(
            json.dumps(
                {
                    "status": "validated",
                    "output": str(args.output.resolve(strict=True)),
                },
                sort_keys=True,
            )
        )
        return
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite diagnostic output: {args.output}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    try:
        result = run_diagnosis(
            args.hf_path,
            args.checkpoint,
            source_revision=args.source_revision,
            device=device,
        )
        validate_diagnostic_evidence(
            result,
            hf_path=args.hf_path,
            checkpoint=args.checkpoint,
            source_revision=args.source_revision,
        )
    except BaseException as error:
        _atomic_write_new(
            args.output,
            _failure_payload(
                hf_path=args.hf_path,
                checkpoint=args.checkpoint,
                source_revision=args.source_revision,
                error=error,
            ),
        )
        raise
    _atomic_write_new(args.output, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
