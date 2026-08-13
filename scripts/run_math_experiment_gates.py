from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
import yaml
from gcw.checkpoint import load_backbone_checkpoint
from gcw.config import load_train_config
from gcw.evaluation.core import ProcessorBypass
from gcw.models import build_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_train(config: Path, log: Path, env: dict[str, str]) -> None:
    with log.open("wb") as handle:
        subprocess.run(
            [sys.executable, "-m", "gcw.cli", "train", "--config", str(config)],
            check=True,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )


def _bypass_gate(config_path: Path, output: Path) -> dict[str, Any]:
    config = load_train_config(config_path)
    if config.model.variant == "tf":
        return {"applicable": False}
    base_config = deepcopy(config.model)
    base_config.variant = "tf"
    base_config.cache_processor = None
    base_config.flow_processor = None
    base_config.diffusion_processor = None
    base_config.workspace_flow_refinement = None
    base_config.lora = None
    reference = build_model(base_config).to(dtype=torch.bfloat16, device="cuda")
    reference.init_weights()
    load_backbone_checkpoint(reference, config.backbone_checkpoint or "")
    composed = build_model(config.model).to(dtype=torch.bfloat16, device="cuda")
    composed.init_weights()
    load_backbone_checkpoint(composed, config.backbone_checkpoint or "")
    bypass = ProcessorBypass(composed)
    reference.eval()
    bypass.eval()
    torch.manual_seed(24)
    input_ids = torch.randint(0, config.model.vocab_size, (1, 32), device="cuda")
    with torch.no_grad():
        observed = bypass(input_ids)
        expected = reference(input_ids)
    exact = torch.equal(observed, expected)
    payload = {
        "applicable": True,
        "exact": exact,
        "max_abs_error": float((observed.float() - expected.float()).abs().max()),
        "config_sha256": _sha256(config_path),
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not exact:
        raise RuntimeError(f"processor bypass drift for {config_path.name}: {payload}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-step CUDA and processor-bypass experiment gates")
    parser.add_argument("--config-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    config_dir = Path(args.config_dir).resolve()
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["GCW_DISABLE_TORCH_COMPILE"] = "1"
    summary: dict[str, Any] = {}
    for source in sorted(config_dir.glob("*.yaml")):
        arm = source.stem
        payload = yaml.safe_load(source.read_text())
        payload["output_dir"] = str(output / "runs" / arm)
        payload["wandb"]["enabled"] = False
        config = output / f"{arm}.yaml"
        config.write_text(yaml.safe_dump(payload, sort_keys=False))
        bypass = _bypass_gate(config, output / f"{arm}-bypass.json")
        log = output / f"{arm}.log"
        _run_train(config, log, env)
        checkpoint = output / "runs" / arm / "checkpoint"
        summary[arm] = {
            "config_sha256": _sha256(config),
            "checkpoint_manifest_sha256": _sha256(checkpoint / "manifest.json"),
            "log_sha256": _sha256(log),
            "bypass": bypass,
        }
    result = {
        "schema": "gcw.math_experiment_gates.v1",
        "source_commit": args.source_commit,
        "arms": summary,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
