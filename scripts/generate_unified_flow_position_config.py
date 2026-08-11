#!/usr/bin/env python3
"""Generate the canonical H200 unified-flow experiment configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def canonical_config() -> dict[str, object]:
    return {
        "schema": "gcw.unified_flow_position.experiment.v1",
        "paths": {
            "backbone_checkpoint": "artifacts/checkpoints/base_backbones/qwen3_06b/tf-v1",
            "tokenizer": "artifacts/models/Qwen3-0.6B-base-v1",
            "train_data": "data/process_branch/gsm8k-answer-only-v1.jsonl",
            "eval_data": "data/eval/gsm8k-test-v1.jsonl",
            "output": "artifacts/experiments/unified-flow-position-qwen3-06b-v1",
        },
        "processor": {
            "process_every": 64,
            "topk": 32,
            "topk_heads": 32,
            "score": "kv_norm",
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_layers": 1,
            "num_heads": 8,
            "num_kv_heads": 8,
            "norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "gate_init": 0.0,
            "attention_dropout": 0.0,
            "per_cache_head_projections": False,
            "gradient_checkpointing": True,
            "update_stream": "v",
            "out_proj_init": "normal",
            "gate_mode": "direct",
            "processor_mode": "standard",
            "encoder_method": "query_attention",
            "encoder_query_source": "final_token",
            "unrope_keys": True,
            "include_self_head_kv": False,
            "iterations": 1,
            "iteration_mode": "internal",
        },
        "flow_refinement": {
            "solver": "midpoint",
            "solver_steps": 2,
            "max_velocity": 0.1,
        },
        "hybrid": {
            "workspace_cycles": 2,
            "position_cycles": 2,
            "latent_tokens_per_cycle": 2,
            "emitter_width": 512,
            "source_delta_bound": 0.1,
            "select_all": True,
        },
        "training": {
            "seed": 20260808,
            "train_rows": 6000,
            "epochs": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "shuffle_contrastive_weight": 0.25,
            "shuffle_contrastive_margin": 0.1,
            "max_prompt_tokens": 512,
            "max_answer_tokens": 64,
            "prompt_suffix": "Solve the problem internally. Return only the final answer in \\boxed{...} form.",
            "response_prefix": "\n\nAnswer: ",
        },
        "evaluation": {
            "expected_rows": 1319,
            "max_prompt_tokens": 768,
            "max_answer_tokens": 64,
            "max_new_tokens": 256,
            "conditions": ["off", "matched", "shuffled"],
            "thinking_modes": {
                "thinking_off": {
                    "enable_thinking": False,
                    "use_chat_template": True,
                    "prompt_suffix": "Solve the problem internally. Return only the final answer in \\boxed{...} form.",
                    "response_prefix": "\n\nAnswer: ",
                },
                "thinking_on": {
                    "enable_thinking": True,
                    "use_chat_template": True,
                    "prompt_suffix": "Solve the problem step by step, then give the final answer in \\boxed{...} form.",
                    "response_prefix": "\n\n<think>\n",
                },
            },
        },
        "runtime": {"device": "cuda", "dtype": "bfloat16"},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(canonical_config(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
