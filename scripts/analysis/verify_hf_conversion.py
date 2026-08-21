from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from gcw.checkpoint import load_checkpoint
from gcw.models import TF
from gcw.provenance import file_sha256
from gcw.runtime import atomic_write_json

SCHEMA_VERSION = 3
EVIDENCE_KIND = "hf_to_gcw_conversion_parity"
POLICY_ID = "qwen3-06b-bf16-parameters-fp32-comparison-v2"
CANONICAL_PROMPT = (
    "Verify the converted model logits for this deterministic prompt."
)
MAX_ABS_THRESHOLD = 0.25
MEAN_ABS_THRESHOLD = 0.01
REQUIRED_DTYPE = "bfloat16"
HF_PARITY_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
CHECKPOINT_PARITY_FILES = (
    "CONVERSION.json",
    "config.json",
    "model.safetensors",
    "train_state.json",
)


def _file_manifest(
    root: Path, names: tuple[str, ...]
) -> dict[str, dict[str, object]]:
    resolved = root.resolve(strict=True)
    manifest: dict[str, dict[str, object]] = {}
    for name in names:
        path = resolved / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(
                f"required regular parity artifact missing: {path}"
            )
        manifest[name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return manifest


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()


def _dtype_names(parameters) -> list[str]:
    return sorted(
        {
            str(parameter.dtype).removeprefix("torch.")
            for parameter in parameters
        }
    )


def _evidence_base(hf_path: Path, checkpoint: Path) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "policy_id": POLICY_ID,
        "source_revision": os.environ.get("SOURCE_REVISION"),
        "script_sha256": file_sha256(script_path),
        "hf_path": str(hf_path.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "canonical_prompt_sha256": hashlib.sha256(
            CANONICAL_PROMPT.encode()
        ).hexdigest(),
        "max_abs_threshold": MAX_ABS_THRESHOLD,
        "mean_abs_threshold": MEAN_ABS_THRESHOLD,
        "required_dtype": REQUIRED_DTYPE,
    }


def verify_hf_conversion(
    hf_path: Path,
    checkpoint: Path,
    *,
    device: torch.device,
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result = _evidence_base(hf_path, checkpoint)
    hf_files_before = _file_manifest(hf_path, HF_PARITY_FILES)
    checkpoint_files_before = _file_manifest(
        checkpoint, CHECKPOINT_PARITY_FILES
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=False)
    token_ids = tokenizer.encode(CANONICAL_PROMPT, add_special_tokens=True)
    if len(token_ids) < 2:
        raise ValueError(
            "canonical parity prompt encoded to fewer than two tokens"
        )
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    hf = AutoModelForCausalLM.from_pretrained(
        hf_path, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to(device)
    hf.eval()
    hf_parameter_dtypes = _dtype_names(hf.parameters())
    if hf_parameter_dtypes != [REQUIRED_DTYPE]:
        raise TypeError(
            f"HF parameters are not uniformly BF16: {hf_parameter_dtypes!r}"
        )
    with torch.inference_mode():
        hf_logits_raw = hf(input_ids=tokens).logits
        hf_logits_dtype = str(hf_logits_raw.dtype).removeprefix("torch.")
        hf_logits = hf_logits_raw.float().cpu()
    del hf, hf_logits_raw
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    native, config, state = load_checkpoint(checkpoint, device=device)
    if not isinstance(native, TF):
        raise TypeError(
            f"converted checkpoint is not a TF model: {type(native)!r}"
        )
    if state.get("dtype") != REQUIRED_DTYPE:
        raise TypeError(
            f"converted checkpoint state dtype is {state.get('dtype')!r}, "
            f"expected {REQUIRED_DTYPE!r}"
        )
    native_parameter_dtypes = _dtype_names(native.parameters())
    if native_parameter_dtypes != [REQUIRED_DTYPE]:
        raise TypeError(
            f"native parameters are not uniformly BF16: {native_parameter_dtypes!r}"
        )
    native.eval()
    with torch.inference_mode():
        native_logits_raw = native(tokens)
        native_logits_dtype = str(native_logits_raw.dtype).removeprefix(
            "torch."
        )
        native_logits = native_logits_raw.float().cpu()
    if hf_logits_dtype != REQUIRED_DTYPE or native_logits_dtype != "float32":
        raise TypeError(
            "parity logit dtypes violate the fixed operational contract: "
            f"hf={hf_logits_dtype!r} (expected 'bfloat16'), "
            f"native={native_logits_dtype!r} (expected 'float32')"
        )

    if native_logits.shape != hf_logits.shape:
        raise RuntimeError(
            f"logit shape mismatch: native={tuple(native_logits.shape)}, "
            f"hf={tuple(hf_logits.shape)}"
        )
    delta = (native_logits - hf_logits).abs()
    max_abs = float(delta.max().item())
    mean_abs = float(delta.mean().item())
    if not math.isfinite(max_abs) or not math.isfinite(mean_abs):
        raise FloatingPointError(
            f"non-finite parity errors: max_abs={max_abs}, mean_abs={mean_abs}"
        )
    top1_match_fraction = float(
        (native_logits.argmax(dim=-1) == hf_logits.argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    final_top1_match = bool(
        native_logits[0, -1].argmax().item() == hf_logits[0, -1].argmax().item()
    )
    passed = (
        max_abs <= MAX_ABS_THRESHOLD
        and mean_abs <= MEAN_ABS_THRESHOLD
        and final_top1_match
    )
    hf_files_after = _file_manifest(hf_path, HF_PARITY_FILES)
    checkpoint_files_after = _file_manifest(checkpoint, CHECKPOINT_PARITY_FILES)
    if hf_files_after != hf_files_before:
        raise RuntimeError(
            "HF model/tokenizer artifacts changed during parity execution"
        )
    if checkpoint_files_after != checkpoint_files_before:
        raise RuntimeError(
            "converted checkpoint artifacts changed during parity execution"
        )
    result.update(
        {
            "passed": passed,
            "hf_model_sha256": hf_files_after["model.safetensors"]["sha256"],
            "checkpoint_model_sha256": checkpoint_files_after[
                "model.safetensors"
            ]["sha256"],
            "hf_files": hf_files_after,
            "checkpoint_files": checkpoint_files_after,
            "token_ids": token_ids,
            "token_ids_sha256": _token_ids_sha256(token_ids),
            "tokens": len(token_ids),
            "logit_shape": list(hf_logits.shape),
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "top1_match_fraction": top1_match_fraction,
            "final_top1_match": final_top1_match,
            "hf_parameter_dtypes": hf_parameter_dtypes,
            "native_parameter_dtypes": native_parameter_dtypes,
            "hf_logits_dtype": hf_logits_dtype,
            "native_logits_dtype": native_logits_dtype,
            "comparison_dtype": "float32",
            "source_state": state,
            "model_config": asdict(config),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        }
    )
    return result


def validate_parity_evidence(
    payload: dict[str, object],
    *,
    hf_path: Path,
    checkpoint: Path,
    source_revision: str,
) -> None:
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "policy_id": POLICY_ID,
        "source_revision": source_revision,
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "canonical_prompt_sha256": hashlib.sha256(
            CANONICAL_PROMPT.encode()
        ).hexdigest(),
        "max_abs_threshold": MAX_ABS_THRESHOLD,
        "mean_abs_threshold": MEAN_ABS_THRESHOLD,
        "required_dtype": REQUIRED_DTYPE,
    }
    for key, expected in expected_header.items():
        if payload.get(key) != expected:
            raise ValueError(f"parity evidence {key} mismatch")
    required_generated_fields = {
        "passed",
        "hf_model_sha256",
        "checkpoint_model_sha256",
        "hf_files",
        "checkpoint_files",
        "token_ids",
        "token_ids_sha256",
        "tokens",
        "logit_shape",
        "max_abs_error",
        "mean_abs_error",
        "top1_match_fraction",
        "final_top1_match",
        "hf_parameter_dtypes",
        "native_parameter_dtypes",
        "hf_logits_dtype",
        "native_logits_dtype",
        "comparison_dtype",
        "source_state",
        "model_config",
        "torch_version",
        "torch_cuda_version",
        "device",
        "gpu_name",
    }
    missing = required_generated_fields - payload.keys()
    if missing:
        raise ValueError(f"parity evidence missing fields: {sorted(missing)!r}")
    expected_fields = (
        set(expected_header)
        | required_generated_fields
        | {
            "hf_path",
            "checkpoint",
        }
    )
    unexpected = payload.keys() - expected_fields
    if unexpected:
        raise ValueError(
            f"parity evidence has unexpected fields: {sorted(unexpected)!r}"
        )
    if payload.get("passed") is not True:
        raise ValueError("parity evidence did not pass")
    if Path(str(payload.get("hf_path", ""))).resolve() != hf_path.resolve(
        strict=True
    ):
        raise ValueError("parity HF source path mismatch")
    if Path(str(payload.get("checkpoint", ""))).resolve() != checkpoint.resolve(
        strict=True
    ):
        raise ValueError("parity checkpoint path mismatch")
    hf_files = _file_manifest(hf_path, HF_PARITY_FILES)
    checkpoint_files = _file_manifest(checkpoint, CHECKPOINT_PARITY_FILES)
    if payload.get("hf_files") != hf_files:
        raise ValueError("parity HF artifact manifest mismatch")
    if payload.get("checkpoint_files") != checkpoint_files:
        raise ValueError("parity checkpoint artifact manifest mismatch")
    hf_config = json.loads((hf_path / "config.json").read_text())
    expected_hf_config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "tie_word_embeddings": True,
    }
    if any(
        hf_config.get(key) != value for key, value in expected_hf_config.items()
    ):
        raise ValueError("parity HF config does not match canonical Qwen3-0.6B")
    token_ids = payload.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) < 2
        or any(
            not isinstance(token, int) or isinstance(token, bool)
            for token in token_ids
        )
        or payload.get("tokens") != len(token_ids)
        or payload.get("token_ids_sha256") != _token_ids_sha256(token_ids)
    ):
        raise ValueError("parity canonical token sequence is unauthenticated")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=False)
    expected_token_ids = tokenizer.encode(
        CANONICAL_PROMPT, add_special_tokens=True
    )
    if token_ids != expected_token_ids:
        raise ValueError(
            "parity token IDs do not match the current canonical tokenizer"
        )
    if payload.get("logit_shape") != [1, len(token_ids), 151936]:
        raise ValueError(
            "parity logit shape does not match canonical Qwen3-0.6B"
        )
    model_config = payload.get("model_config")
    checkpoint_config = json.loads((checkpoint / "config.json").read_text())
    if model_config != checkpoint_config:
        raise ValueError(
            "parity model config does not match checkpoint config.json"
        )
    expected_model_config = {
        "variant": "tf",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_layers": 28,
        "num_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "tie_embeddings": True,
    }
    if not isinstance(model_config, dict) or any(
        model_config.get(key) != value
        for key, value in expected_model_config.items()
    ):
        raise ValueError(
            "parity model config does not match canonical Qwen3-0.6B TF"
        )
    if payload.get("torch_version") != torch.__version__:
        raise ValueError(
            "parity torch version does not match validator runtime"
        )
    if payload.get("torch_cuda_version") != torch.version.cuda:
        raise ValueError("parity CUDA version does not match validator runtime")
    if payload.get("device") != "cuda":
        raise ValueError("parity device is not exactly CUDA")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "parity validation requires exactly one live CUDA device"
        )
    if payload.get("gpu_name") != torch.cuda.get_device_name(0):
        raise ValueError("parity GPU name does not match validator runtime")
    if "H100" not in str(payload.get("gpu_name")):
        raise ValueError("parity validator runtime is not an H100")
    for key, threshold in (
        ("max_abs_error", MAX_ABS_THRESHOLD),
        ("mean_abs_error", MEAN_ABS_THRESHOLD),
    ):
        value = payload.get(key)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or float(value) < 0.0
            or float(value) > threshold
        ):
            raise ValueError(f"parity evidence {key} failed")
    if payload.get("final_top1_match") is not True:
        raise ValueError("parity final-token top-1 mismatch")
    fraction = payload.get("top1_match_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isfinite(float(fraction))
        or not 0.0 <= float(fraction) <= 1.0
    ):
        raise ValueError("parity top-1 match fraction is invalid")
    if payload.get("hf_parameter_dtypes") != [REQUIRED_DTYPE]:
        raise ValueError("parity HF parameter dtype mismatch")
    if payload.get("native_parameter_dtypes") != [REQUIRED_DTYPE]:
        raise ValueError("parity native parameter dtype mismatch")
    if payload.get("hf_logits_dtype") != REQUIRED_DTYPE:
        raise ValueError("parity HF logits dtype mismatch")
    if payload.get("native_logits_dtype") != "float32":
        raise ValueError("parity native logits dtype mismatch")
    if payload.get("comparison_dtype") != "float32":
        raise ValueError("parity comparison dtype mismatch")
    source_state = payload.get("source_state")
    checkpoint_state = json.loads((checkpoint / "train_state.json").read_text())
    if source_state != checkpoint_state:
        raise ValueError(
            "parity source state does not match checkpoint train_state.json"
        )
    if (
        not isinstance(source_state, dict)
        or source_state.get("dtype") != REQUIRED_DTYPE
    ):
        raise ValueError("parity checkpoint state dtype mismatch")
    if Path(str(source_state.get("source", ""))).resolve() != hf_path.resolve(
        strict=True
    ):
        raise ValueError("parity checkpoint source-state path mismatch")
    for path, key in (
        (hf_path / "model.safetensors", "hf_model_sha256"),
        (checkpoint / "model.safetensors", "checkpoint_model_sha256"),
    ):
        if payload.get(key) != file_sha256(path):
            raise ValueError(f"parity artifact digest mismatch for {path}")


def _atomic_write_new(path: Path, payload: dict[str, object]) -> None:
    atomic_write_json(
        path,
        payload,
        allow_nan=False,
        exclusive=True,
        indent=2,
        overwrite_error="refusing to overwrite parity result: {path}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        source_revision = os.environ.get("SOURCE_REVISION")
        if not isinstance(source_revision, str) or len(source_revision) != 40:
            raise RuntimeError(
                "SOURCE_REVISION must be a 40-character Git revision"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("conversion parity requires CUDA")
        result = verify_hf_conversion(
            args.hf_path,
            args.checkpoint,
            device=torch.device("cuda"),
        )
        if result["passed"] is True:
            validate_parity_evidence(
                result,
                hf_path=args.hf_path,
                checkpoint=args.checkpoint,
                source_revision=source_revision,
            )
    except Exception as error:
        result = _evidence_base(args.hf_path, args.checkpoint)
        result.update(
            {
                "passed": False,
                "failure_stage": "parity_execution",
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        _atomic_write_new(args.out, result)
        print(json.dumps(result, sort_keys=True))
        raise
    _atomic_write_new(args.out, result)
    print(json.dumps(result, sort_keys=True))
    if result["passed"] is not True:
        raise RuntimeError(
            "HF conversion parity thresholds failed; see persisted result"
        )


if __name__ == "__main__":
    main()
