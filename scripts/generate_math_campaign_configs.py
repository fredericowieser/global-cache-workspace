from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from gcw.convert import config_from_hf


WAND_ENTITY = "huawei-noahs-ark"
WAND_PROJECT = "global-cache-workspace"


def _processor(*, width: int, layers: int = 1, longshot: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {
        "process_every": 64,
        "topk": 32,
        "topk_heads": 32,
        "score": "kv_norm" if longshot else "attn_l2",
        "hidden_size": width,
        "intermediate_size": 4 * width,
        "num_layers": layers,
        "num_heads": max(1, width // 128),
        "num_kv_heads": max(1, width // 128),
        "gate_init": -5.0,
        "gradient_checkpointing": True,
        "update_stream": "kv",
        "unrope_keys": True,
    }
    if longshot:
        config.update(encoder_method="query_attention", encoder_query_source="final_token")
    return config


def _arms(width: int) -> dict[str, dict[str, Any]]:
    standard = _processor(width=width)
    icarus = {**standard, "topk_heads": 56}
    longshot = _processor(width=width, longshot=True)
    return {
        "full-sft": {"variant": "tf", "trainable": "backbone", "lr": 1e-5},
        "lora-r32": {
            "variant": "tf",
            "trainable": "lora",
            "lr": 1e-4,
            "lora": {"rank": 32, "alpha": 32.0, "dropout": 0.0},
        },
        "bt": {"variant": "bt", "trainable": "cache_processor", "lr": 1e-4, "cache_processor": standard},
        "bgt": {"variant": "bgt", "trainable": "cache_processor", "lr": 1e-4, "cache_processor": standard},
        "icarus": {"variant": "icarus", "trainable": "cache_processor", "lr": 1e-4, "cache_processor": icarus},
        "longshot": {
            "variant": "longshot",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": longshot,
        },
        "longshot-internal-r4": {
            "variant": "longshot",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": {**longshot, "iterations": 4, "iteration_mode": "internal"},
        },
        "longshot-external-r4": {
            "variant": "longshot",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": {**longshot, "iterations": 4, "iteration_mode": "external"},
        },
        "flow-bgt-t4": {
            "variant": "bgt",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": standard,
            "flow_processor": {"num_steps": 4, "time_conditioning": True, "eta_max": 1.0, "flow_gate_init": -4.0},
        },
        "diffusion-bgt-t4-sigma015": {
            "variant": "bgt",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": standard,
            "diffusion_processor": {
                "num_steps": 4,
                "time_conditioning": True,
                "eta_max": 1.0,
                "refinement_gate_init": -4.0,
                "noise_scale": 0.15,
                "conditioning_scale": 1.0,
                "inference_mode": "deterministic",
            },
        },
        "diffusion-bgt-t4-sigma0": {
            "variant": "bgt",
            "trainable": "cache_processor",
            "lr": 1e-4,
            "cache_processor": standard,
            "diffusion_processor": {
                "num_steps": 4,
                "time_conditioning": True,
                "eta_max": 1.0,
                "refinement_gate_init": -4.0,
                "noise_scale": 0.0,
                "conditioning_scale": 1.0,
                "inference_mode": "deterministic",
            },
        },
    }


def generate(args: argparse.Namespace) -> list[Path]:
    from transformers import AutoConfig

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    data_manifest = json.loads(Path(args.data_manifest).read_text())
    hf_config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    model = config_from_hf(hf_config, family="qwen3")
    model.sequence_len = args.sequence_length
    model_payload = _plain(model)
    rows = int(data_manifest["rows"])
    max_steps = math.ceil(rows / args.global_batch_rows)
    contract = {
        "schema": "gcw.math_campaign.v1",
        "dataset_sha256": data_manifest["dataset_sha256"],
        "tokenizer_identity_sha256": data_manifest["tokenizer_identity_sha256"],
        "backbone_sha256": _sha256_file(Path(args.backbone_checkpoint) / "model.safetensors"),
        "sequence_length": args.sequence_length,
        "global_batch_rows": args.global_batch_rows,
        "max_steps": max_steps,
        "dataset_path": data_manifest["dataset_path"],
        "tokenizer_source": args.model,
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
    }
    contract_path = output / "campaign.json"
    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    paths: list[Path] = []
    for name, arm in _arms(args.processor_width).items():
        arm_model = deepcopy(model_payload)
        arm_model["variant"] = arm["variant"]
        arm_model["cache_processor"] = arm.get("cache_processor")
        arm_model["lora"] = arm.get("lora")
        arm_model["flow_processor"] = arm.get("flow_processor")
        arm_model["diffusion_processor"] = arm.get("diffusion_processor")
        payload = {
            "model": arm_model,
            "data": {
                "kind": "sft",
                "dataset": data_manifest["dataset_path"],
                "tokenizer": "hf",
                "tokenizer_source": args.model,
                "sft_chat_template": True,
                "sft_enable_thinking": False,
                "sft_loss_scope": "answer_only",
                "canonical_execution": True,
                "expected_artifact_sha256": data_manifest["dataset_sha256"],
                "expected_source_ids_sha256": data_manifest["source_ids_sha256"],
                "campaign_manifest": str(contract_path.resolve()),
            },
            "mode": "sft",
            "distributed": args.distributed,
            "trainable": arm["trainable"],
            "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
            "output_dir": str((Path(args.run_root) / name).resolve()),
            "batch_size": args.micro_batch,
            "sequence_len": args.sequence_length,
            "max_steps": max_steps,
            "global_batch_rows": args.global_batch_rows,
            "learning_rate": arm["lr"],
            "adam_beta2": 0.999,
            "warmup_steps": max(1, round(0.03 * max_steps)),
            "warmdown_ratio": 0.1,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "bf16": True,
            "seed": args.seed,
            "save_every": max(1, max_steps // 10),
            "wandb": {
                "enabled": True,
                "entity": WAND_ENTITY,
                "project": WAND_PROJECT,
                "name": name,
                "group": args.campaign,
                "tags": ["qwen3-0.6b", "instruct", "math-campaign", name],
            },
            "metadata": {
                "campaign": args.campaign,
                "arm": name,
                "data_manifest": str(Path(args.data_manifest).resolve()),
            },
        }
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False))
        paths.append(path)
    return paths


def _plain(value: Any) -> Any:
    from gcw.config import to_dict

    return to_dict(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the matched Qwen3-0.6B campaign matrix")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--campaign", default="qwen3-06b-instruct-openmath-openr1-v1")
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--global-batch-rows", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--processor-width", type=int, default=1024)
    parser.add_argument("--distributed", choices=("none", "ddp", "fsdp"), default="none")
    parser.add_argument("--seed", type=int, default=24)
    paths = generate(parser.parse_args())
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
