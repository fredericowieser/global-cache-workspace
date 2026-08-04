from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import torch
from safetensors import safe_open

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import ModelConfig
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.state_hash import tensor_state_sha256


MIXED_DTYPE = "mixed_bfloat16_backbone_float32_processor"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_revision(repository: Path, expected: str) -> None:
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("source revision must be a full lowercase Git SHA")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if actual != expected:
        raise ValueError("source revision differs from the live checkout")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    ):
        raise ValueError("untrained control derivation requires a clean source checkout")


def derive_untrained_processor_control(
    source: str | Path,
    destination: str | Path,
    *,
    seed: int,
    source_revision: str,
    tokenizer_source: str,
    repository: str | Path,
) -> dict[str, Any]:
    """Reset only a trained mixed checkpoint's processor to its exact FP32 initial state."""
    source_path = Path(source).resolve()
    destination_path = Path(destination)
    repository_path = Path(repository).resolve()
    _verify_revision(repository_path, source_revision)
    if os.path.lexists(destination_path):
        raise FileExistsError(f"refusing existing control checkpoint: {destination_path}")
    for name in ("model.safetensors", "config.json", "train_state.json"):
        if not (source_path / name).is_file():
            raise FileNotFoundError(f"source checkpoint is missing {name}")

    payload = json.loads((source_path / "config.json").read_text(encoding="utf-8"))
    config = ModelConfig(**payload["model"])
    source_train_state = json.loads(
        (source_path / "train_state.json").read_text(encoding="utf-8")
    )
    source_model_sha256 = _sha256_file(source_path / "model.safetensors")
    run_config_path = source_path.parent / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError("source checkpoint is missing its parent run_config.json")
    source_run_config_sha256 = _sha256_file(run_config_path)
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    normalized_tokenizer = tokenizer_source.removeprefix("hf:")
    if (
        source_train_state.get("dtype") != MIXED_DTYPE
        or source_train_state.get("checkpoint_fidelity_verified") is not True
        or source_train_state.get("source_revision") != source_revision
        or source_train_state.get("source_clean") is not True
        or source_train_state.get("training_seed") != seed
        or source_train_state.get("processor_initialization_recipe")
        != "torch.manual_seed(training_seed); build_model(model_config); v1"
        or source_train_state.get("torch_version") != torch.__version__
        or source_train_state.get("tokenizer_source") != normalized_tokenizer
        or not source_train_state.get("tokenizer_identity_sha256")
        or source_train_state.get("run_config_sha256") != source_run_config_sha256
        or run_config.get("seed") != seed
        or Path(str(run_config.get("output_dir", ""))).resolve() != source_path.parent
        or str(run_config.get("data", {}).get("tokenizer", "")).removeprefix("hf:")
        != normalized_tokenizer
    ):
        raise ValueError("source pilot provenance does not match control derivation contract")

    with safe_open(source_path / "model.safetensors", framework="pt", device="cpu") as handle:
        source_dtype_counts = Counter(
            str(handle.get_tensor(name).dtype) for name in handle.keys()
        )
    if set(source_dtype_counts) != {"torch.bfloat16", "torch.float32"}:
        raise ValueError(
            f"source checkpoint tensor dtypes differ from mixed contract: {source_dtype_counts}"
        )

    torch.manual_seed(seed)
    initial_model = build_model(config)
    if not isinstance(initial_model, BGT):
        raise TypeError("untrained processor control requires a BGT model")
    initial_processor = {
        name: tensor.detach().cpu().clone()
        for name, tensor in initial_model.cache_processor.state_dict().items()
    }
    if any(tensor.is_floating_point() and tensor.dtype != torch.float32 for tensor in initial_processor.values()):
        raise ValueError("fresh processor initialization is not FP32")
    initial_processor_sha256 = tensor_state_sha256(initial_processor)
    initial_gate = float(initial_processor["block.gate"].item())
    if (
        initial_processor_sha256 != source_train_state.get("initial_processor_sha256")
        or initial_gate != source_train_state.get("initial_processor_gate")
    ):
        raise ValueError("regenerated processor does not match pilot-recorded initialization")
    del initial_model
    gc.collect()

    trained_model, loaded_config, loaded_train_state = load_checkpoint(source_path, device="cpu")
    if not isinstance(trained_model, BGT):
        raise TypeError("source checkpoint is not BGT")
    if loaded_config != config or loaded_train_state != source_train_state:
        raise ValueError("source checkpoint changed while deriving control")
    trained_processor = {
        name: tensor.detach().cpu().clone()
        for name, tensor in trained_model.cache_processor.state_dict().items()
    }
    if trained_processor.keys() != initial_processor.keys():
        raise ValueError("source and initial processor state keys differ")
    changed = 0
    total = 0
    for name, initial in initial_processor.items():
        trained = trained_processor[name]
        if trained.shape != initial.shape or trained.dtype != initial.dtype:
            raise ValueError(f"processor tensor contract changed: {name}")
        total += trained.numel()
        changed += int(torch.count_nonzero(trained != initial).item())
    if changed == 0:
        raise ValueError("trained processor is identical to initialization")
    trained_processor_sha256 = tensor_state_sha256(trained_processor)
    if trained_processor_sha256 != source_train_state.get("post_load_processor_sha256"):
        raise ValueError("source processor hash differs from pilot publication evidence")
    source_backbone_sha256 = tensor_state_sha256(
        {name: tensor for name, tensor in trained_model.backbone.state_dict().items()}
    )
    trained_model.cache_processor.load_state_dict(initial_processor, strict=True)

    control_state = {
        "kind": "derived_untrained_fp32_cache_processor_control",
        "dtype": MIXED_DTYPE,
        "source_revision": source_revision,
        "source_checkpoint": str(source_path),
        "source_model_sha256": source_model_sha256,
        "source_run_config_sha256": source_run_config_sha256,
        "source_train_state": source_train_state,
        "tokenizer_source": normalized_tokenizer,
        "tokenizer_identity_sha256": source_train_state["tokenizer_identity_sha256"],
        "processor_initialization_seed": seed,
        "processor_initialization_recipe": source_train_state[
            "processor_initialization_recipe"
        ],
        "torch_version": torch.__version__,
        "processor_storage_dtype": "float32",
        "initial_processor_sha256": initial_processor_sha256,
        "trained_processor_sha256": trained_processor_sha256,
        "source_backbone_sha256": source_backbone_sha256,
        "control_backbone_sha256": source_backbone_sha256,
        "checkpoint_dtype_counts": dict(source_dtype_counts),
        "processor_values_different_from_trained": changed,
        "processor_values_total": total,
        "initial_gate_float32": initial_gate,
        "trained_checkpoint_gate_float32": float(trained_processor["block.gate"].item()),
    }

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}-", dir=destination_path.parent)
    )
    try:
        save_checkpoint(temporary, trained_model, config, control_state)
        reloaded, reloaded_config, reloaded_state = load_checkpoint(temporary, device="cpu")
        if not isinstance(reloaded, BGT):
            raise TypeError("derived control did not reload as BGT")
        if reloaded_config != config or reloaded_state != control_state:
            raise ValueError("derived control metadata failed standard-loader round trip")
        reloaded_processor = {
            name: tensor.detach().cpu()
            for name, tensor in reloaded.cache_processor.state_dict().items()
        }
        if tensor_state_sha256(reloaded_processor) != initial_processor_sha256:
            raise ValueError("derived processor failed exact standard-loader round trip")
        reloaded_backbone_sha256 = tensor_state_sha256(
            {name: tensor for name, tensor in reloaded.backbone.state_dict().items()}
        )
        if reloaded_backbone_sha256 != source_backbone_sha256:
            raise ValueError("derived control changed the frozen backbone")
        with safe_open(temporary / "model.safetensors", framework="pt", device="cpu") as handle:
            dtype_counts = Counter(str(handle.get_tensor(name).dtype) for name in handle.keys())
        if dtype_counts != source_dtype_counts:
            raise ValueError(
                "derived control tensor dtypes differ from source mixed checkpoint contract: "
                f"{dtype_counts} != {source_dtype_counts}"
            )
        del reloaded
        for child in temporary.iterdir():
            if child.is_file():
                with child.open("rb") as handle:
                    os.fsync(handle.fileno())
        if _sha256_file(source_path / "model.safetensors") != source_model_sha256:
            raise ValueError("source checkpoint changed during control derivation")
        os.rename(temporary, destination_path)
        directory_descriptor = os.open(destination_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "destination": str(destination_path.resolve()),
        "model_sha256": _sha256_file(destination_path / "model.safetensors"),
        **control_state,
    }
