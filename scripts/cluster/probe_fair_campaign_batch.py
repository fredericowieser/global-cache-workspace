#!/usr/bin/env python3
"""Find the largest power-of-two microbatch that completes three train steps."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


BATCH_SIZES = (128, 64, 32, 16, 8, 4, 2, 1)
STEP_LINE = re.compile(r"^step ([1-3])/3 loss=[0-9.eE+-]+(?:\s|$)")
OOM_LINE = re.compile(
    r"(?:CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|"
    r"CUBLAS_STATUS_ALLOC_FAILED|HIP out of memory|Triton Error \[CUDA\]: out of memory)",
    re.IGNORECASE,
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    configs = sorted((root / "configs/experiments").glob("*fair_campaign*.yaml"))
    selected = os.environ.get("GCW_PROBE_CONFIG")
    if selected:
        configs = [config for config in configs if config.name == selected]
    if not configs:
        raise RuntimeError("no fair_campaign configs found")
    result_dir = root / "slurm_out" / f"fair-campaign-batch-{os.environ['SLURM_JOB_ID']}"
    result_dir.mkdir(parents=True, exist_ok=False)
    report = result_dir / "report.json"
    results = []
    environment = os.environ.copy()
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    environment["WANDB_MODE"] = "disabled"
    prior_path = os.environ.get("GCW_PROBE_START_REPORT")
    prior_sizes = {}
    if prior_path:
        prior_sizes = {
            row["config"]: row["optimal_batch_size"]
            for row in json.loads(Path(prior_path).read_text(encoding="utf-8"))
        }
    print(f"results={result_dir} configs={len(configs)}", flush=True)

    for source in configs:
        base = yaml.safe_load(source.read_text(encoding="utf-8"))
        row = {"config": str(source.relative_to(root)), "optimal_batch_size": None, "attempts": []}
        results.append(row)
        start_size = prior_sizes.get(row["config"], BATCH_SIZES[0])
        for batch_size in BATCH_SIZES:
            if batch_size > start_size:
                continue
            name = f"{source.stem}-batch{batch_size}"
            config = dict(base)
            if config["model"]["cache_processor"] is not None:
                config["model"] = dict(base["model"])
                config["model"]["cache_processor"] = dict(base["model"]["cache_processor"])
                config["model"]["cache_processor"]["gate_init"] = float(
                    config["model"]["cache_processor"]["gate_init"]
                )
            config["batch_size"] = batch_size
            config["max_steps"] = 3
            config["save_every"] = 0
            config["save_final"] = False
            config["output_dir"] = str(result_dir / name / "state")
            config["wandb"] = {**base["wandb"], "enabled": False}
            config_path = result_dir / f"{name}.yaml"
            log_path = result_dir / f"{name}.log"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                "uv", "run", "--locked", "--no-sync", "--extra", "gpu",
                "--extra", "qwen35", "gcw", "train", "--config", str(config_path),
            ]
            print(f"probing {source.name} batch={batch_size}", flush=True)
            steps = set()
            oom = False
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command, cwd=root, env=environment, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if match := STEP_LINE.match(line):
                        steps.add(int(match.group(1)))
                        print(f"  batch={batch_size} step={match.group(1)}/3", flush=True)
                    if OOM_LINE.search(line):
                        oom = True
                returncode = process.wait()
            status = "success" if returncode == 0 and steps == {1, 2, 3} else (
                "oom" if oom else "error"
            )
            row["attempts"].append({
                "batch_size": batch_size, "status": status, "steps": sorted(steps),
                "exit_code": returncode, "log": str(log_path.relative_to(root)),
            })
            if status == "success":
                row["optimal_batch_size"] = batch_size
                print(f"selected {source.name} batch={batch_size}", flush=True)
                break
            if status != "oom":
                print(f"error {source.name} batch={batch_size}: see {log_path}", flush=True)
                break
        report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print(f"report={report}", flush=True)
    return 0 if all(row["optimal_batch_size"] is not None for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
