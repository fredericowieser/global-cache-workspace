#!/usr/bin/env python3
"""Validate the narrowly scoped OPSD pragmatic parity waiver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from gcw.config import TrainConfig, load_train_config

SCHEMA_VERSION = 2
KIND = "opsd_pragmatic_conversion_parity_waiver"
VALIDATED_KIND = "validated_opsd_pragmatic_conversion_parity_waiver"
SCOPE = "pragmatic_same_gcw_backbone_sensitivity_after_failed_hf_parity"
BACKBONE_LABEL = "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"
WAIVER_RELATIVE = Path("configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v2.json")
WAIVER_FILE_SHA256 = "f3c481927cf394072770db05fa8bd2a09535044cdf2a9c48168903be456d290f"
WAIVER_CANONICAL_SHA256 = "23bb60f3818d11614658c0773646f151d99ff277429410610dd9ac863c8dde4b"
PARITY_PATH = Path("/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1/parity.json")
PARITY_SHA256 = "1aca29e6db3099d652d8635d57f8f79a9acccfbec915a378c0477e7104b86216"
PARITY_POLICY = "qwen3-06b-bf16-parameters-fp32-comparison-v2"
DIAGNOSTIC_PATH = Path(
    "/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1/diagnostics/"
    "layerwise-distribution-v1-fp64.json"
)
DIAGNOSTIC_SHA256 = "a02e037e2cb666f6ca38dc7e4c07e193be8e4e70b0c89b6beb95213a7ef94821"
DIAGNOSTIC_SOURCE_REVISION = "fbb55d8f886021e14db0e3a8be3f54b1892e376e"
BACKBONE = Path("/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1")
TOKENIZER = Path("/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1")
OUTPUT_ROOT = Path("/mnt/jfs-01/shared/gcw/experiments/qwen3_06b_opsd_native_base_pragmatic_v1")
CANARY_SUCCESS = OUTPUT_ROOT / "canary-steps1-bs1/SUCCESS.json"
TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa_opsd_generate",
    "agieval_gaokao_mathqa_opsd_generate",
)
CONDITIONS = (
    "cot-off__cp-off",
    "cot-off__cp-on",
    "cot-on__cp-off",
    "cot-on__cp-on",
)
CONFIGS: dict[str, dict[str, Any]] = {
    "qwen3_06b_opsd_native_base_pragmatic_canary_v1.yaml": {
        "steps": 1,
        "effective_batch": 1,
        "take_rows": 64,
        "output": OUTPUT_ROOT / "canary-steps1-bs1",
        "sha256": "0478dc84cdbd9c11c5aadd3e705fe7f2392ddc9ab487e376cfe385b08262b059",
    },
    "qwen3_06b_opsd_native_base_pragmatic_steps1000_bs16_v1.yaml": {
        "steps": 1000,
        "effective_batch": 16,
        "take_rows": 128000,
        "output": OUTPUT_ROOT / "steps1000-bs16",
        "sha256": "72b08249a64cefe038080867cd21d6fdd1e163ce5645f82ff2046b8709f0935b",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(document: dict[str, Any]) -> str:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_waiver_document(document: dict[str, Any]) -> None:
    if not isinstance(document, dict) or _canonical_sha256(document) != WAIVER_CANONICAL_SHA256:
        raise ValueError("waiver document differs from the approved closed contract")
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("kind") != KIND
        or document.get("scientific_scope") != SCOPE
        or document.get("decision")
        != "proceed_as_pragmatic_same_gcw_backbone_sensitivity_experiment"
    ):
        raise ValueError("waiver document identity mismatch")
    constraints = document.get("constraints")
    if not isinstance(constraints, dict) or (
        constraints.get("canary_required") is not True
        or constraints.get("max_optimizer_steps") != 1000
        or constraints.get("effective_batch_examples") != 16
        or constraints.get("ten_thousand_steps_authorized") is not False
        or constraints.get("training_gpu_families") != ["H100", "H200"]
        or constraints.get("training_gpu_model_topology_must_match_canary") is not True
        or tuple(constraints.get("evaluation_tasks", ())) != TASKS
        or tuple(constraints.get("conditions", ())) != CONDITIONS
    ):
        raise ValueError("waiver document constraint mismatch")


def validate_training_gpu(document: dict[str, Any], gpu_name: str) -> str:
    validate_waiver_document(document)
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError("training GPU name is missing")
    match = re.fullmatch(r"NVIDIA (H100|H200)(?:\b.*)?", gpu_name)
    if match is None:
        raise ValueError(f"training GPU is outside the approved families: {gpu_name!r}")
    family = match.group(1)
    if family not in document["constraints"]["training_gpu_families"]:
        raise ValueError(f"training GPU family is not approved: {family!r}")
    return family


def detect_runtime_gpu(document: dict[str, Any]) -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    if count != 1:
        raise ValueError(f"training requires exactly one visible GPU; got {count}")
    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    gpu_family = validate_training_gpu(document, gpu_name)
    total_memory = int(properties.total_memory)
    if total_memory <= 0:
        raise ValueError("training GPU reports invalid total memory")
    capability = torch.cuda.get_device_capability(0)
    if len(capability) != 2 or any(not isinstance(value, int) for value in capability):
        raise ValueError("training GPU reports invalid compute capability")
    return {
        "gpu_name": gpu_name,
        "gpu_family": gpu_family,
        "gpu_total_memory_bytes": total_memory,
        "gpu_compute_capability": f"{capability[0]}.{capability[1]}",
    }


def _expected_metadata(*, steps: int, effective_batch: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "campaign": "opsd-native-base-pragmatic-v1",
        "scientific_scope": SCOPE,
        "required_conversion_parity": str(PARITY_PATH),
        "required_conversion_diagnostic": str(DIAGNOSTIC_PATH),
        "conversion_parity_waiver": str(WAIVER_RELATIVE),
        "comparison_reference": "configs/train/bottleneck_llm/qwen3_06b_bgt_from_sft_frozen.yaml",
        "backbone_label": BACKBONE_LABEL,
        "distillation_effective_batch_examples": effective_batch,
        "requested_optimizer_steps": steps,
    }
    if steps == 1000:
        metadata["promotion_evidence"] = str(CANARY_SUCCESS)
    return metadata


def validate_train_config(path: Path) -> TrainConfig:
    path = path.resolve(strict=True)
    expected = CONFIGS.get(path.name)
    if expected is None:
        raise ValueError("train config is not one of the two approved pragmatic configs")
    if _sha256(path) != expected["sha256"]:
        raise ValueError("pragmatic train config digest differs from the approved protocol")
    config = load_train_config(path)
    distill = config.distillation
    tokenizer_source = config.data.tokenizer
    backbone_checkpoint = config.backbone_checkpoint
    if distill is None:
        raise ValueError("pragmatic config lacks distillation")
    if not isinstance(tokenizer_source, str) or not isinstance(backbone_checkpoint, str):
        raise ValueError("pragmatic config lacks tokenizer or backbone checkpoint")
    steps = int(expected["steps"])
    effective_batch = int(expected["effective_batch"])
    if (
        config.metadata != _expected_metadata(steps=steps, effective_batch=effective_batch)
        or config.mode != "cot_distillation"
        or config.model.variant != "bgt"
        or config.model.family != "qwen3"
        or not config.freeze_backbone
        or Path(backbone_checkpoint).resolve(strict=True) != BACKBONE.resolve(strict=True)
        or Path(tokenizer_source.removeprefix("hf:")).resolve(strict=True)
        != TOKENIZER.resolve(strict=True)
        or config.data.sft_take_rows != expected["take_rows"]
        or Path(config.output_dir) != expected["output"]
        or config.max_steps != 1
        or not config.bf16
        or config.seed != 24
        or distill.teacher_thinking is not True
        or distill.student_thinking is not False
        or distill.teacher_trajectory_mode != "native_full"
        or distill.visible_trajectory_mode != "both"
        or distill.max_optimizer_steps != steps
        or distill.gradient_accumulation_steps != effective_batch
        or distill.record_cache_displacements is not True
        or not config.wandb.enabled
        or config.wandb.mode != "online"
        or config.wandb.resume != "never"
    ):
        raise ValueError("pragmatic train config differs from the approved protocol")
    return config


def validate_parity_failure_document(document: dict[str, Any]) -> None:
    error = document.get("mean_abs_error")
    threshold = document.get("mean_abs_threshold")
    valid_error = isinstance(error, (int, float)) and not isinstance(error, bool)
    valid_threshold = isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
    if (
        document.get("passed") is not False
        or document.get("policy_id") != PARITY_POLICY
        or not valid_error
        or not valid_threshold
    ):
        raise ValueError("waiver parity evidence is not the approved numerical failure")
    assert isinstance(error, (int, float)) and not isinstance(error, bool)
    assert isinstance(threshold, (int, float)) and not isinstance(threshold, bool)
    error_value = float(error)
    threshold_value = float(threshold)
    if (
        not math.isfinite(error_value)
        or not math.isfinite(threshold_value)
        or threshold_value < 0.0
        or error_value <= threshold_value
    ):
        raise ValueError("waiver parity evidence is not the approved numerical failure")


def validate_external_evidence() -> None:
    parity = PARITY_PATH.resolve(strict=True)
    diagnostic = DIAGNOSTIC_PATH.resolve(strict=True)
    if _sha256(parity) != PARITY_SHA256 or _sha256(diagnostic) != DIAGNOSTIC_SHA256:
        raise ValueError("pragmatic waiver external evidence hash mismatch")
    parity_document = json.loads(parity.read_text())
    validate_parity_failure_document(parity_document)
    diagnostic_document = json.loads(diagnostic.read_text())
    if (
        diagnostic_document.get("status") != "completed"
        or diagnostic_document.get("source_revision") != DIAGNOSTIC_SOURCE_REVISION
        or diagnostic_document.get("parent_parity_sha256") != PARITY_SHA256
        or diagnostic_document.get("slurm_job_id") != "22525"
    ):
        raise ValueError("waiver diagnostic evidence identity mismatch")
    prompts = diagnostic_document.get("prompts", [])
    if len(prompts) != 5 or any(
        row.get("distribution", {}).get("top1_match_fraction") != 1.0 for row in prompts
    ):
        raise ValueError("waiver diagnostic prompt evidence mismatch")


def validate_static(
    *, repo: Path, config_path: Path, waiver_path: Path, source_revision: str
) -> dict[str, Any]:
    repo = repo.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    waiver_path = waiver_path.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError("source revision must be a full lowercase commit hash")
    if config_path.parent != repo / "configs/train/opsd" or config_path.name not in CONFIGS:
        raise ValueError("pragmatic config path is outside the approved source location")
    if waiver_path != repo / WAIVER_RELATIVE:
        raise ValueError("pragmatic waiver path differs from the approved source location")
    if _sha256(waiver_path) != WAIVER_FILE_SHA256:
        raise ValueError("pragmatic waiver source bytes changed")
    waiver = json.loads(waiver_path.read_text())
    validate_waiver_document(waiver)
    config = validate_train_config(config_path)
    validate_external_evidence()
    distill = config.distillation
    if distill is None:
        raise ValueError("validated pragmatic config unexpectedly lacks distillation")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "validated_opsd_pragmatic_static_waiver",
        "source_revision": source_revision,
        "scientific_scope": SCOPE,
        "backbone_label": BACKBONE_LABEL,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "waiver": str(waiver_path),
        "waiver_sha256": WAIVER_FILE_SHA256,
        "failed_conversion_parity": str(PARITY_PATH),
        "failed_conversion_parity_sha256": PARITY_SHA256,
        "conversion_diagnostic": str(DIAGNOSTIC_PATH),
        "conversion_diagnostic_sha256": DIAGNOSTIC_SHA256,
        "optimizer_steps": distill.max_optimizer_steps,
        "effective_batch_examples": distill.gradient_accumulation_steps,
        "parity_gate_remains_failed": True,
        "ten_thousand_steps_authorized": False,
        "paired_cp_contrasts_share_converted_backbone": True,
    }


def validate_all(
    *, repo: Path, config_path: Path, waiver_path: Path, source_revision: str
) -> dict[str, Any]:
    evidence = validate_static(
        repo=repo,
        config_path=config_path,
        waiver_path=waiver_path,
        source_revision=source_revision,
    )
    waiver = json.loads(waiver_path.resolve(strict=True).read_text())
    evidence.update(detect_runtime_gpu(waiver))
    evidence["kind"] = VALIDATED_KIND
    evidence["training_gpu_model_topology_must_match_canary"] = True
    return evidence


def _write_new(path: Path, document: dict[str, Any]) -> None:
    path.parent.resolve(strict=True)
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--waiver", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only == (args.out is not None):
        raise SystemExit("choose exactly one of --check-only or --out")
    evidence = validate_all(
        repo=args.repo,
        config_path=args.config,
        waiver_path=args.waiver,
        source_revision=args.source_revision,
    )
    if args.out is not None:
        _write_new(args.out, evidence)
        print(f"pragmatic_waiver_evidence={args.out}")
    else:
        print("pragmatic_waiver_validated=true")


if __name__ == "__main__":
    main()
