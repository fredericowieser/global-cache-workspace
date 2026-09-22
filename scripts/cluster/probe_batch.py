#!/usr/bin/env python3
"""Find a fitting microbatch by increasing gradient accumulation for supplied configs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml


STEP_LINE = re.compile(r"^step (\d+)/(\d+) loss=(\S+)(?:\s|$)")
OOM_LINE = re.compile(
    r"(?:CUDA out of memory|OutOfMemoryError|CUDNN_STATUS_ALLOC_FAILED|"
    r"CUBLAS_STATUS_ALLOC_FAILED|HIP out of memory|Triton Error \[CUDA\]: out of memory)",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path, help="Training YAML paths; shell globs are supported.")
    parser.add_argument("--steps", type=int, default=3, help="Optimizer steps per attempt (default: 3).")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs per attempt (default: 1).")
    parser.add_argument("--result-dir", type=Path, help="New directory for temporary configs, logs, and report.json.")
    args = parser.parse_args(argv)
    if args.steps < 1 or args.gpus < 1:
        parser.error("--steps and --gpus must be positive")
    root = Path(__file__).resolve().parents[2]
    configs = list(dict.fromkeys(path.resolve() for path in args.configs))
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("run the probe controller once, outside torchrun")
    prepared = []
    for source in configs:
        base = yaml.safe_load(source.read_text(encoding="utf-8"))
        if args.gpus > 1 and base.get("distributed", "none") not in {"ddp", "fsdp"}:
            raise ValueError(f"multi-GPU probing requires a ddp/fsdp config: {source}")
        if base.get("resume") is not None:
            raise ValueError(f"resume is incompatible with fresh batch-size attempts: {source}")
        data = base.get("data", {})
        if data.get("kind") == "sft" and not (root / data["dataset"]).is_file():
            raise FileNotFoundError(f"prepared SFT dataset is missing: {data['dataset']}")
        sequence_len = base["sequence_len"]
        token_budget = base.get("total_batch_size", 0)
        global_rows = base.get("global_batch_rows", 0)
        if not global_rows:
            token_budget = token_budget or base["batch_size"] * sequence_len * args.gpus
            if token_budget <= 0 or token_budget % sequence_len:
                raise ValueError(f"token batch budget must be divisible by sequence_len: {source}")
            global_rows = token_budget // sequence_len
        if global_rows <= 0:
            raise ValueError(f"global batch rows must be positive: {source}")
        identity = str(source.relative_to(root)) if source.is_relative_to(root) else str(source)
        if global_rows % args.gpus:
            raise ValueError(f"global batch rows must be divisible by GPU count: {source}")
        rows_per_rank = global_rows // args.gpus
        candidates = [
            (rows_per_rank // accumulation, accumulation)
            for accumulation in range(1, rows_per_rank + 1)
            if rows_per_rank % accumulation == 0
        ]
        prepared.append((source, base, identity, global_rows, candidates))
    result_dir = args.result_dir or (
        root / "slurm_out" / f"batch-probe-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    )
    result_dir = result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=False)
    report = result_dir / "report.json"
    results = []
    environment = os.environ.copy()
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    environment["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    environment["WANDB_MODE"] = "disabled"
    environment["PYTHONUNBUFFERED"] = "1"
    print(f"results={result_dir} configs={len(configs)}", flush=True)

    for index, (source, base, identity, global_rows, candidates) in enumerate(prepared):
        row = {
            "config": identity,
            "selection_criterion": "first_success_in_increasing_gradient_accumulation",
            "probe_steps": args.steps,
            "sequence_len": base["sequence_len"],
            "global_batch_rows": global_rows,
            "tokens_per_step": global_rows * base["sequence_len"],
            "gpus": args.gpus,
            "candidates": [
                {"batch_size": batch, "grad_accum_steps": accumulation}
                for batch, accumulation in candidates
            ],
            "optimal_batch_size": None,
            "optimal_grad_accum_steps": None,
            "attempts": [],
        }
        results.append(row)
        for batch_size, accumulation in candidates:
            name = f"{index:03d}-{source.stem}-batch{batch_size}"
            config = dict(base)
            if base["model"].get("cache_processor") is not None:
                config["model"] = dict(base["model"])
                config["model"]["cache_processor"] = dict(base["model"]["cache_processor"])
                if "gate_init" in config["model"]["cache_processor"]:
                    config["model"]["cache_processor"]["gate_init"] = float(
                        config["model"]["cache_processor"]["gate_init"]
                    )
            config["batch_size"] = batch_size
            if not base.get("global_batch_rows", 0) and not base.get("total_batch_size", 0):
                config["total_batch_size"] = global_rows * base["sequence_len"]
            config["max_steps"] = args.steps
            config["save_every"] = 0
            config["save_final"] = False
            config["eval_every"] = 0
            config["eval_on_start"] = False
            config["output_dir"] = str(result_dir / name / "state")
            config["wandb"] = {**base.get("wandb", {}), "enabled": False}
            config_path = result_dir / f"{name}.yaml"
            log_path = result_dir / f"{name}.log"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            command = [
                "uv", "run", "--locked", "--no-sync", "--extra", "gpu",
                "--extra", "qwen35",
            ]
            if args.gpus > 1:
                command.extend([
                    "torchrun", "--standalone", f"--nproc-per-node={args.gpus}",
                    "-m", "gcw.cli",
                ])
            else:
                command.append("gcw")
            command.extend(["train", "--config", str(config_path)])
            print(f"probing {source.name} batch={batch_size} grad_accum={accumulation} gpus={args.gpus}", flush=True)
            steps = set()
            oom = False
            invalid_loss = False
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
                        try:
                            finite_loss = math.isfinite(float(match.group(3)))
                        except ValueError:
                            finite_loss = False
                        if finite_loss and int(match.group(2)) == args.steps:
                            steps.add(int(match.group(1)))
                        else:
                            invalid_loss = True
                        print(f"  batch={batch_size} step={match.group(1)}/{args.steps}", flush=True)
                    if OOM_LINE.search(line):
                        oom = True
                returncode = process.wait()
            success = (
                returncode == 0 and steps == set(range(1, args.steps + 1))
                and not invalid_loss and not oom
            )
            status = "success" if success else ("oom" if oom else "error")
            row["attempts"].append({
                "batch_size": batch_size, "grad_accum_steps": accumulation,
                "status": status, "steps": sorted(steps),
                "exit_code": returncode, "log": str(log_path),
            })
            if success:
                row["optimal_batch_size"] = batch_size
                row["optimal_grad_accum_steps"] = accumulation
            report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
            if success:
                print(f"selected {source.name} batch={batch_size}", flush=True)
                break
            if status != "oom":
                print(f"error {source.name} batch={batch_size}: see {log_path}", flush=True)
                break

    print(f"report={report}", flush=True)
    return 0 if all(row["optimal_batch_size"] is not None for row in results) else 1


if __name__ == "__main__":
    sys.exit(main())
