#!/usr/bin/env python3
"""Train a continuous-thought emitter against frozen local Qwen3 GCW weights."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcw.checkpoint import load_checkpoint  # noqa: E402
from gcw.continuous_thought import (  # noqa: E402
    ContinuousThoughtEmitter,
    ContinuousThoughtEmitterConfig,
    continuous_thought_answer_loss,
    frozen_qwen_backbone,
    save_emitter_checkpoint,
)
from gcw.data.tokenizers import get_tokenizer  # noqa: E402
from gcw.process_branch import InvalidProcessBranchExample, tokenize_answer_only_record  # noqa: E402
from gcw.process_data import AnswerOnlyRecord, read_answer_only_records  # noqa: E402


DEFAULT_BACKBONE = Path("artifacts/checkpoints/base_backbones/qwen3_06b/tf-v1")
DEFAULT_TOKENIZER = next((Path("artifacts") / "models").glob("Qwen3*"))
DEFAULT_DATA = Path("data/process_branch/gsm8k-answer-only-v1.jsonl")
H200_COMMAND = """uv run --extra gpu --extra eval python scripts/explore_continuous_thought.py \\
  --backbone-checkpoint artifacts/checkpoints/base_backbones/qwen3_06b/tf-v1 \\
  --tokenizer artifacts/models/Qwen3-0.6B-base-v1 \\
  --data data/process_branch/gsm8k-answer-only-v1.jsonl \\
  --output runs/continuous-thought-qwen3-06b-k4-l2 \\
  --k 4 --latent-tokens-per-cycle 2 --steps 200 --lr 3e-4 \\
  --train-rows 128 --heldout-rows 32 --dtype bfloat16 --device cuda"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory answer-CE training for a continuous-thought emitter on frozen Qwen3.",
        epilog=f"Exact H200 smoke/training command:\n{H200_COMMAND}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backbone-checkpoint", type=Path, default=DEFAULT_BACKBONE)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=4, help="recurrent thought cycles; 0 is an exact skip")
    parser.add_argument("--latent-tokens-per-cycle", type=int, default=1)
    parser.add_argument("--processor-width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--shuffle-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--shuffle-contrastive-margin", type=float, default=0.1)
    parser.add_argument("--train-rows", type=int, default=128)
    parser.add_argument("--heldout-rows", type=int, default=32)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=64)
    parser.add_argument("--prompt-suffix", default="Return only the final answer in a box.")
    parser.add_argument("--answer-prefix", default="\n\nAnswer: ")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=20260808)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.k < 0:
        raise ValueError("--k must be non-negative")
    if args.latent_tokens_per_cycle <= 0:
        raise ValueError("--latent-tokens-per-cycle must be positive")
    if args.processor_width <= 0:
        raise ValueError("--processor-width must be positive")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")
    if args.shuffle_contrastive_weight < 0 or args.shuffle_contrastive_margin < 0:
        raise ValueError("shuffle contrastive weight and margin must be non-negative")
    if args.shuffle_contrastive_weight > 0 and args.k == 0:
        raise ValueError("shuffle contrastive training requires --k > 0")
    if args.train_rows <= 0 or args.heldout_rows < 0:
        raise ValueError("--train-rows must be positive and --heldout-rows non-negative")
    for path in (args.backbone_checkpoint, args.tokenizer, args.data):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(name)


def _tokenize_rows(
    records: Sequence[AnswerOnlyRecord],
    tokenizer,
    args: argparse.Namespace,
    device: torch.device,
):
    examples = []
    rejected = 0
    for record in records:
        try:
            examples.append(
                tokenize_answer_only_record(
                    record,
                    tokenizer,
                    answer_response_prefix=args.answer_prefix,
                    prompt_suffix=args.prompt_suffix,
                    max_prompt_tokens=args.max_prompt_tokens,
                    max_answer_tokens=args.max_answer_tokens,
                    device=device,
                )
            )
        except InvalidProcessBranchExample:
            rejected += 1
    return examples, rejected


def _example_loss(backbone, emitter, example, cycles: int, *, source_override=None):
    return continuous_thought_answer_loss(
        backbone,
        emitter,
        prompt_ids=example.prompt_ids.unsqueeze(0),
        response_prefix_ids=example.answer_response_prefix_ids.unsqueeze(0),
        target_ids=example.final_answer_ids.unsqueeze(0),
        cycles=cycles,
        source_override=source_override,
    )


@torch.no_grad()
def _prompt_source(backbone, example):
    hidden, _ = backbone.forward_chunk(
        example.prompt_ids.unsqueeze(0),
        0,
        None,
        logits=False,
        cache_format="windowed",
    )
    return hidden[:, -1].detach()


@torch.no_grad()
def _heldout_metrics(backbone, emitter, examples, cycles: int, autocast_context) -> dict[str, float | int]:
    if not examples:
        return {"rows": 0}
    emitter.eval()
    sources = []
    for example in examples:
        with autocast_context():
            sources.append(_prompt_source(backbone, example))
    matched: list[float] = []
    shuffled: list[float] = []
    skip: list[float] = []
    for index, example in enumerate(examples):
        donor_source = sources[(index + 1) % len(sources)]
        with autocast_context():
            matched.append(float(_example_loss(backbone, emitter, example, cycles).loss.float()))
            shuffled.append(
                float(
                    _example_loss(
                        backbone,
                        emitter,
                        example,
                        cycles,
                        source_override=donor_source,
                    ).loss.float()
                )
            )
            skip.append(float(_example_loss(backbone, emitter, example, 0).loss.float()))
    emitter.train()
    matched_mean = sum(matched) / len(matched)
    shuffled_mean = sum(shuffled) / len(shuffled)
    skip_mean = sum(skip) / len(skip)
    return {
        "rows": len(examples),
        "k_loss": matched_mean,
        "shuffled_source_loss": shuffled_mean,
        "k0_skip_loss": skip_mean,
        "k_minus_k0": matched_mean - skip_mean,
        "matched_minus_shuffled": matched_mean - shuffled_mean,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    _validate_args(args)
    torch.manual_seed(args.seed)
    device = _device(args.device)
    loaded_model, model_config, backbone_state = load_checkpoint(args.backbone_checkpoint, device=device)
    backbone = frozen_qwen_backbone(loaded_model)
    if args.dtype == "float32":
        backbone.float()
    elif args.dtype == "bfloat16":
        backbone.to(dtype=torch.bfloat16)
    emitter = ContinuousThoughtEmitter(
        ContinuousThoughtEmitterConfig(
            model_width=backbone.config.hidden_size,
            processor_width=args.processor_width,
            latent_tokens_per_cycle=args.latent_tokens_per_cycle,
        )
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(emitter.parameters(), lr=args.lr)

    tokenizer = get_tokenizer(f"hf:{args.tokenizer}")
    if tokenizer.get_vocab_size() > backbone.config.vocab_size:
        raise ValueError("tokenizer vocabulary exceeds backbone vocabulary")
    records = read_answer_only_records(args.data)
    needed = args.train_rows + args.heldout_rows
    if len(records) < needed:
        raise ValueError(f"data has {len(records)} rows but train/heldout split requests {needed}")
    train_records = records[: args.train_rows]
    heldout_records = records[args.train_rows : needed]
    train_examples, train_rejected = _tokenize_rows(train_records, tokenizer, args, device)
    heldout_examples, heldout_rejected = _tokenize_rows(heldout_records, tokenizer, args, device)
    if not train_examples:
        raise ValueError("no token-bounded training rows")

    use_bfloat16 = backbone.embed_tokens.weight.dtype == torch.bfloat16 and device.type == "cuda"

    def autocast_context():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bfloat16 else nullcontext()

    args.output.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output / "metrics.jsonl"
    started = time.monotonic()
    last_loss = 0.0
    last_objective = 0.0
    max_backbone_grad_tensors = 0
    with metrics_path.open("x", encoding="utf-8") as metrics:
        for step in range(args.steps):
            example = train_examples[step % len(train_examples)]
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                result = _example_loss(backbone, emitter, example, args.k)
                shuffled_train_loss = None
                contrastive_penalty = result.loss.new_zeros(())
                objective = result.loss
                if args.shuffle_contrastive_weight > 0:
                    donor_example = train_examples[(step + 1) % len(train_examples)]
                    donor_source = _prompt_source(backbone, donor_example)
                    shuffled_result = _example_loss(
                        backbone,
                        emitter,
                        example,
                        args.k,
                        source_override=donor_source,
                    )
                    shuffled_train_loss = shuffled_result.loss
                    contrastive_penalty = torch.relu(
                        args.shuffle_contrastive_margin + result.loss - shuffled_result.loss
                    )
                    objective = result.loss + args.shuffle_contrastive_weight * contrastive_penalty
            if objective.requires_grad:
                objective.backward()
                backbone_grad_tensors = sum(parameter.grad is not None for parameter in backbone.parameters())
                grad_norm = torch.nn.utils.clip_grad_norm_(emitter.parameters(), 1.0)
                optimizer.step()
            else:
                # K=0 deliberately bypasses the emitter, so a fully frozen
                # backbone leaves no autograd graph and the optimizer is a no-op.
                backbone_grad_tensors = 0
                grad_norm = result.loss.new_zeros(())
            frozen_model_grad_tensors = sum(parameter.grad is not None for parameter in loaded_model.parameters())
            max_backbone_grad_tensors = max(max_backbone_grad_tensors, frozen_model_grad_tensors)
            last_loss = float(result.loss.detach().float())
            last_objective = float(objective.detach().float())
            payload = {
                "step": step + 1,
                "loss": last_loss,
                "objective_loss": last_objective,
                "shuffled_train_loss": (
                    None if shuffled_train_loss is None else float(shuffled_train_loss.detach().float())
                ),
                "contrastive_penalty": float(contrastive_penalty.detach().float()),
                "grad_norm": float(grad_norm.detach().float()),
                "k": args.k,
                "added_positions": result.trace.added_positions,
                "answer_start_position": result.answer_start_position,
                "backbone_grad_tensors": backbone_grad_tensors,
                "frozen_model_grad_tensors": frozen_model_grad_tensors,
                "elapsed_seconds": time.monotonic() - started,
            }
            metrics.write(json.dumps(payload, sort_keys=True) + "\n")
            metrics.flush()

    heldout = _heldout_metrics(backbone, emitter, heldout_examples, args.k, autocast_context)
    summary: dict[str, object] = {
        "kind": "gcw.continuous_thought.exploration.v1",
        "model": asdict(model_config),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_train_state": backbone_state,
        "emitter": asdict(emitter.config),
        "k": args.k,
        "steps": args.steps,
        "learning_rate": args.lr,
        "shuffle_contrastive_weight": args.shuffle_contrastive_weight,
        "shuffle_contrastive_margin": args.shuffle_contrastive_margin,
        "train_rows_requested": args.train_rows,
        "train_rows_valid": len(train_examples),
        "train_rows_rejected": train_rejected,
        "heldout_rows_requested": args.heldout_rows,
        "heldout_rows_valid": len(heldout_examples),
        "heldout_rows_rejected": heldout_rejected,
        "last_train_loss": last_loss,
        "last_objective_loss": last_objective,
        "heldout": heldout,
        "backbone_parameters_trainable": sum(parameter.requires_grad for parameter in backbone.parameters()),
        "frozen_model_parameters_trainable": sum(parameter.requires_grad for parameter in loaded_model.parameters()),
        "max_frozen_model_grad_tensors": max_backbone_grad_tensors,
        "emitter_trainable_parameters": sum(parameter.numel() for parameter in emitter.parameters()),
        "dtype": str(backbone.embed_tokens.weight.dtype).removeprefix("torch."),
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
    }
    save_emitter_checkpoint(args.output / "emitter-checkpoint", emitter, train_state=summary)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
