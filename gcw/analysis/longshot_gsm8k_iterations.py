from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import random
import re
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from gcw.checkpoint import load_checkpoint
from gcw.eval import _stop_token_sequences, _tokenizer_for, _trim_generated_token_ids, _truncate_context_continuation
from gcw.models.cache_state import CacheState, StaticCacheState


DEFAULT_CHECKPOINT = "/mnt/jfs-01/shared/gcw-best_longshot"
DEFAULT_OUT_DIR = "/mnt/jfs-01/shared/gcw/results/longshot_gsm8k_iterations"
UNTIL = ("</s>", "<|end_of_text|>", "<|eot_id|>", "\n\nQuestion:", "\nQuestion:")


@dataclass(frozen=True)
class Gsm8kExample:
    index: int
    prompt: str
    answer: str
    target: str


@dataclass(frozen=True)
class IterationExampleRow:
    iteration: int
    example_id: int
    split_id: int
    synergy_raw: float
    synergy_norm: float
    mean_full_nll: float
    mean_base_nll: float
    n_tokens: int


@dataclass(frozen=True)
class GenerationRow:
    iteration: int
    example_id: int
    target: str
    prediction: str
    correct: bool
    generation: str


@dataclass(frozen=True)
class SummaryRow:
    iteration: int
    accuracy: float
    correct: int
    n_examples: int
    mean_synergy_raw: float
    raw_ci_low: float
    raw_ci_high: float
    mean_synergy_norm: float
    norm_ci_low: float
    norm_ci_high: float
    mean_full_nll: float
    mean_base_nll: float
    n_tokens: int


@dataclass(frozen=True)
class PreparedAnswer:
    prefix_len: int
    prefix_cache: CacheState
    generation_input_ids: torch.Tensor
    score_input_ids: torch.Tensor
    score_target_ids: torch.Tensor
    target_token_ids: list[int]


class HeadMeanAblator:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.backbone = model.backbone
        self.num_heads = int(self.backbone.config.num_heads)
        self.head_dim = int(self.backbone.config.head_dim)
        self.layer_indices = list(range(len(self.backbone.layers)))
        self.head_means: dict[int, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    @torch.no_grad()
    def capture(self, batches: Iterable[torch.Tensor]) -> None:
        sums = {idx: torch.zeros(self.num_heads, self.head_dim, dtype=torch.float32) for idx in self.layer_indices}
        counts = {idx: 0 for idx in self.layer_indices}

        def make_hook(layer_idx: int):
            def hook(_module, inputs):
                x = inputs[0].detach().float().cpu()
                x = x.reshape(-1, self.num_heads, self.head_dim)
                sums[layer_idx] += x.sum(dim=0)
                counts[layer_idx] += x.shape[0]

            return hook

        handles = [
            self.backbone.layers[idx].self_attn.o_proj.register_forward_pre_hook(make_hook(idx))
            for idx in self.layer_indices
        ]
        try:
            for input_ids in batches:
                self.backbone.forward_full(input_ids)
        finally:
            for handle in handles:
                handle.remove()

        device = self.backbone.get_device()
        for idx in self.layer_indices:
            if counts[idx] == 0:
                raise ValueError(f"no calibration tokens reached layer {idx}")
            self.head_means[idx] = (sums[idx] / counts[idx]).to(device)

    @contextmanager
    def ablated(self, masks: dict[int, torch.Tensor] | None):
        if masks is None:
            self.remove()
            yield
            return
        self.apply(masks)
        try:
            yield
        finally:
            self.remove()

    def apply(self, masks: dict[int, torch.Tensor]) -> None:
        self.remove()

        def make_hook(layer_idx: int):
            def hook(_module, inputs):
                x = inputs[0]
                batch, seq_len, width = x.shape
                heads = x.reshape(batch, seq_len, self.num_heads, self.head_dim)
                mask = masks[layer_idx].to(device=x.device, dtype=x.dtype).view(1, 1, self.num_heads, 1)
                means = self.head_means[layer_idx].to(device=x.device, dtype=x.dtype).view(1, 1, self.num_heads, self.head_dim)
                mixed = heads * mask + means * (1.0 - mask)
                return (mixed.reshape(batch, seq_len, width),)

            return hook

        self.hooks = [
            self.backbone.layers[idx].self_attn.o_proj.register_forward_pre_hook(make_hook(idx))
            for idx in self.layer_indices
        ]

    def remove(self) -> None:
        for handle in self.hooks:
            handle.remove()
        self.hooks = []


class UnigramBaseline:
    def __init__(self, vocab_size: int, epsilon: float = 1e-8):
        self.vocab_size = vocab_size
        self.epsilon = epsilon
        self.counts: Counter[int] = Counter()
        self.total = 0

    def update(self, token_ids: Iterable[int]) -> None:
        ids = [int(token_id) for token_id in token_ids if 0 <= int(token_id) < self.vocab_size]
        self.counts.update(ids)
        self.total += len(ids)

    def nll(self, token_ids: list[int]) -> torch.Tensor:
        denom = max(self.total + self.vocab_size * self.epsilon, self.vocab_size * self.epsilon)
        return torch.tensor(
            [-math.log((self.counts.get(int(token_id), 0) + self.epsilon) / denom) for token_id in token_ids],
            dtype=torch.float32,
        )


@torch.no_grad()
def run_analysis(args: argparse.Namespace) -> None:
    _configure_environment()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(out_dir, args)
    _seed_everything(args.seed)

    model, cfg, state = load_checkpoint(args.checkpoint, device=args.device)
    if not hasattr(model, "cache_processor"):
        raise ValueError(f"expected a cache-processor checkpoint, got variant={cfg.variant!r}")
    model.eval()
    tokenizer = _tokenizer_for(cfg, state)

    measure_iterations = _iteration_values(args)
    examples = select_shard(
        load_gsm8k_examples(args.num_examples, seed=args.sample_seed, random_sample=args.sample_mode == "random"),
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    calibration = load_gsm8k_examples(args.num_calibration, split="train")
    baseline = fit_unigram_baseline(tokenizer, calibration, cfg.vocab_size, target_prefix=args.target_prefix)
    ablator = HeadMeanAblator(model)
    ablator.capture(calibration_batches(model, tokenizer, calibration, args.max_length, args.max_target_tokens, args.target_prefix))

    detail_rows: list[IterationExampleRow] = []
    generation_rows: list[GenerationRow] = []
    progress_path = out_dir / "progress.json"
    start_time = time.time()
    skipped = 0
    for example_idx, example in enumerate(examples):
        prepared = prepare_answer(
            model,
            tokenizer,
            example,
            max_length=args.max_length,
            max_prompt_tokens=args.max_prompt_tokens,
            max_target_tokens=args.max_target_tokens,
            target_prefix=args.target_prefix,
            prompt_style=args.prompt_style,
            prefill_mode=args.prefill_mode,
        )
        if prepared is None:
            skipped += 1
            _write_progress(progress_path, example_idx + 1, len(examples), start_time, skipped=skipped)
            continue
        cache = prepared.prefix_cache
        base = baseline.nll(prepared.target_token_ids)
        max_iteration = max(measure_iterations)
        for iteration in range(max_iteration + 1):
            if iteration in measure_iterations:
                generation_rows.append(
                    generate_for_iteration(
                        model,
                        tokenizer,
                        prepared,
                        cache,
                        example=example,
                        iteration=iteration,
                        max_new_tokens=args.max_gen_toks,
                    )
                )
                full = score_answer_nll(model, prepared, cache)
                for split_id in range(args.num_splits):
                    mask_a, mask_b = deterministic_masks(
                        ablator,
                        split_id=split_id,
                        seed=args.seed + example.index * 31,
                        group_fraction=args.group_fraction,
                    )
                    loss_a = score_answer_nll(model, prepared, cache, ablator=ablator, masks=mask_a)
                    loss_b = score_answer_nll(model, prepared, cache, ablator=ablator, masks=mask_b)
                    raw, norm = synergy_from_losses(full, loss_a, loss_b, base)
                    detail_rows.append(
                        IterationExampleRow(
                            iteration=iteration,
                            example_id=example.index,
                            split_id=split_id,
                            synergy_raw=raw,
                            synergy_norm=norm,
                            mean_full_nll=float(full.mean().item()),
                            mean_base_nll=float(base.mean().item()),
                            n_tokens=int(full.numel()),
                        )
                    )
            if iteration < max_iteration:
                cache = model.cache_processor(cache, select_all=True)
        _write_progress(progress_path, example_idx + 1, len(examples), start_time, skipped=skipped)
        if (example_idx + 1) % args.flush_every == 0:
            write_outputs(out_dir, detail_rows, generation_rows, args.bootstrap_samples, args.seed)

    write_outputs(out_dir, detail_rows, generation_rows, args.bootstrap_samples, args.seed)
    _write_progress(progress_path, len(examples), len(examples), start_time, status="completed", skipped=skipped)


def load_gsm8k_examples(
    limit: int | None,
    *,
    split: str = "test",
    seed: int = 0,
    random_sample: bool = False,
) -> list[Gsm8kExample]:
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=split)
    indices = list(range(len(data)))
    if random_sample:
        rng = random.Random(seed)
        rng.shuffle(indices)
    if limit is not None:
        indices = indices[:limit]
    examples = []
    for idx in indices:
        row = data[idx]
        answer = str(row["answer"])
        examples.append(Gsm8kExample(idx, str(row["question"]), answer, target_answer(answer) or ""))
    return examples


def select_shard(examples: list[Gsm8kExample], *, num_shards: int, shard_index: int) -> list[Gsm8kExample]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return [example for offset, example in enumerate(examples) if offset % num_shards == shard_index]


def fit_unigram_baseline(
    tokenizer,
    examples: list[Gsm8kExample],
    vocab_size: int,
    *,
    target_prefix: str,
) -> UnigramBaseline:
    baseline = UnigramBaseline(vocab_size)
    for example in examples:
        baseline.update(_encode(tokenizer, f"{example.prompt}{target_prefix}{example.answer}", add_special_tokens=False))
    return baseline


def calibration_batches(
    model: torch.nn.Module,
    tokenizer,
    examples: list[Gsm8kExample],
    max_length: int,
    max_target_tokens: int,
    target_prefix: str,
) -> Iterable[torch.Tensor]:
    device = model.get_device()
    add_special = _analysis_add_special_tokens(model)
    for example in examples:
        prompt_ids = _encode(tokenizer, example.prompt, add_special_tokens=add_special)
        target_ids = _encode(tokenizer, target_prefix + example.answer, add_special_tokens=False)[:max_target_tokens]
        yield torch.tensor((prompt_ids + target_ids)[-max_length:], dtype=torch.long, device=device).unsqueeze(0)


def prepare_answer(
    model: torch.nn.Module,
    tokenizer,
    example: Gsm8kExample,
    *,
    max_length: int,
    max_prompt_tokens: int | None,
    max_target_tokens: int,
    target_prefix: str,
    prompt_style: str,
    prefill_mode: str,
) -> PreparedAnswer | None:
    device = model.get_device()
    add_special = _analysis_add_special_tokens(model)
    prompt = format_prompt(tokenizer, example.prompt, prompt_style=prompt_style)
    prompt_ids = _encode(tokenizer, prompt, add_special_tokens=add_special and prompt_style != "legacy")
    target_ids = _encode(tokenizer, target_prefix + example.answer, add_special_tokens=False)[:max_target_tokens]
    if max_prompt_tokens is not None and len(prompt_ids) > max_prompt_tokens:
        return None
    prompt_ids, target_ids = _truncate_context_continuation(prompt_ids, target_ids, max_length)
    if len(prompt_ids) < 2 or len(target_ids) < 2:
        raise ValueError("GSM8K prompt must leave at least one prefix token and one trigger token")
    answer_ids = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)
    if prefill_mode == "prefix_trigger":
        prefix_ids = torch.tensor(prompt_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        generation_input_ids = torch.tensor(prompt_ids[-1:], dtype=torch.long, device=device).unsqueeze(0)
        score_input_ids = torch.cat([generation_input_ids, answer_ids[:, :-1]], dim=1)
        score_target_ids = answer_ids
        target_token_ids = [int(token_id) for token_id in target_ids]
    elif prefill_mode == "old_full_prompt":
        prefix_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        generation_input_ids = torch.tensor(prompt_ids[-1:], dtype=torch.long, device=device).unsqueeze(0)
        score_input_ids = answer_ids
        score_target_ids = answer_ids[:, 1:]
        target_token_ids = [int(token_id) for token_id in target_ids[1:]]
    else:
        raise ValueError(f"unsupported prefill_mode: {prefill_mode}")
    empty_scores = prefix_ids.new_empty(1, 0, dtype=torch.float32)
    _, prefix_cache = model.backbone.forward_chunk(prefix_ids, 0, CacheState([], [], empty_scores))
    return PreparedAnswer(
        prefix_len=prefix_ids.shape[1],
        prefix_cache=prefix_cache,
        generation_input_ids=generation_input_ids,
        score_input_ids=score_input_ids,
        score_target_ids=score_target_ids,
        target_token_ids=target_token_ids,
    )


def deterministic_masks(
    ablator: HeadMeanAblator,
    *,
    split_id: int,
    seed: int,
    group_fraction: float,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    rng = random.Random(seed + split_id * 1009)
    group_size = max(1, min(ablator.num_heads - 1, round(ablator.num_heads * group_fraction)))
    a_masks = {}
    b_masks = {}
    for layer_idx in ablator.layer_indices:
        heads = list(range(ablator.num_heads))
        rng.shuffle(heads)
        a = torch.zeros(ablator.num_heads)
        b = torch.zeros(ablator.num_heads)
        a[heads[:group_size]] = 1.0
        b[heads[group_size : group_size * 2]] = 1.0
        if b.sum() == 0:
            b[heads[group_size:]] = 1.0
        a_masks[layer_idx] = a
        b_masks[layer_idx] = b
    return a_masks, b_masks


@torch.no_grad()
def score_answer_nll(
    model: torch.nn.Module,
    prepared: PreparedAnswer,
    cache: CacheState,
    *,
    ablator: HeadMeanAblator | None = None,
    masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    context = ablator.ablated(masks) if ablator is not None else _null_context()
    with context:
        logits, _ = model.backbone.forward_chunk(prepared.score_input_ids, prepared.prefix_len, cache)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    target = prepared.score_target_ids[0]
    if logp.shape[0] != target.shape[0]:
        logp = logp[-target.shape[0] :]
    return -logp.gather(-1, target[:, None]).squeeze(-1).detach().cpu()


@torch.no_grad()
def generate_for_iteration(
    model: torch.nn.Module,
    tokenizer,
    prepared: PreparedAnswer,
    cache: CacheState,
    *,
    example: Gsm8kExample,
    iteration: int,
    max_new_tokens: int,
) -> GenerationRow:
    generation_cache = static_cache_from_cache(model, cache, extra_tokens=prepared.generation_input_ids.shape[1] + max_new_tokens)
    logits, generation_cache = model.backbone.forward_chunk_static(prepared.generation_input_ids, prepared.prefix_len, generation_cache)
    pos = prepared.prefix_len + prepared.generation_input_ids.shape[1]
    generated: list[int] = []
    stop_sequences = _stop_token_sequences(tokenizer, UNTIL)
    eos_token_id = getattr(tokenizer, "eot_id", None)
    for _ in range(max_new_tokens):
        next_id = int(logits[:, -1].argmax(dim=-1).item())
        generated.append(next_id)
        trimmed = _trim_generated_token_ids(generated, stop_sequences, eos_token_id)
        if len(trimmed) < len(generated):
            generated = trimmed
            break
        next_tensor = torch.tensor([[next_id]], dtype=prepared.generation_input_ids.dtype, device=prepared.generation_input_ids.device)
        logits, generation_cache = model.backbone.forward_chunk_static(next_tensor, pos, generation_cache)
        pos += 1
    text = tokenizer.decode(generated)
    prediction = extract_flexible(text) or ""
    return GenerationRow(
        iteration=iteration,
        example_id=example.index,
        target=example.target,
        prediction=prediction,
        correct=answers_match(prediction, example.target),
        generation=text,
    )


def static_cache_from_cache(model: torch.nn.Module, cache: CacheState, *, extra_tokens: int) -> StaticCacheState:
    static = StaticCacheState.allocate(
        batch_size=cache.batch_size,
        num_layers=cache.num_layers,
        num_kv_heads=cache.keys[0].shape[1],
        max_seq_len=cache.seq_len + extra_tokens,
        head_dim=cache.keys[0].shape[-1],
        device=cache.device,
        dtype=model.backbone.embed_tokens.weight.dtype,
    )
    for layer_idx, (keys, values) in enumerate(zip(cache.keys, cache.values, strict=True)):
        static.write_layer(layer_idx, 0, keys, values)
    static.write_scores(0, cache.scores)
    return static


def synergy_from_losses(
    full: torch.Tensor,
    group_a: torch.Tensor,
    group_b: torch.Tensor,
    baseline: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[float, float]:
    raw = (group_a + group_b) - (full + baseline)
    raw_mean = float(raw.mean().item())
    useful = float((baseline - full).mean().item())
    norm = raw_mean / useful if useful > eps else float("nan")
    return raw_mean, norm


def summarize_rows(
    detail_rows: list[IterationExampleRow],
    generation_rows: list[GenerationRow],
    *,
    bootstrap_samples: int,
    seed: int,
) -> list[SummaryRow]:
    generations_by_iteration: dict[int, list[GenerationRow]] = {}
    for row in generation_rows:
        generations_by_iteration.setdefault(row.iteration, []).append(row)

    examples_by_iteration: dict[int, dict[int, list[IterationExampleRow]]] = {}
    for row in detail_rows:
        examples_by_iteration.setdefault(row.iteration, {}).setdefault(row.example_id, []).append(row)

    summaries = []
    for iteration in sorted(examples_by_iteration):
        collapsed = []
        for rows in examples_by_iteration[iteration].values():
            first = rows[0]
            collapsed.append(
                {
                    "raw": _mean([row.synergy_raw for row in rows]),
                    "norm": _mean([row.synergy_norm for row in rows]),
                    "full": first.mean_full_nll,
                    "base": first.mean_base_nll,
                    "tokens": first.n_tokens,
                }
            )
        generations = generations_by_iteration.get(iteration, [])
        correct = sum(int(row.correct) for row in generations)
        raw_values = [row["raw"] for row in collapsed]
        norm_values = [row["norm"] for row in collapsed]
        raw_ci = _bootstrap_ci(raw_values, bootstrap_samples=bootstrap_samples, seed=seed + iteration * 17)
        norm_ci = _bootstrap_ci(norm_values, bootstrap_samples=bootstrap_samples, seed=seed + iteration * 19)
        summaries.append(
            SummaryRow(
                iteration=iteration,
                accuracy=correct / len(generations) if generations else float("nan"),
                correct=correct,
                n_examples=len(generations),
                mean_synergy_raw=_mean(raw_values),
                raw_ci_low=raw_ci[0],
                raw_ci_high=raw_ci[1],
                mean_synergy_norm=_mean(norm_values),
                norm_ci_low=norm_ci[0],
                norm_ci_high=norm_ci[1],
                mean_full_nll=_mean([row["full"] for row in collapsed]),
                mean_base_nll=_mean([row["base"] for row in collapsed]),
                n_tokens=sum(int(row["tokens"]) for row in collapsed),
            )
        )
    return summaries


def write_outputs(
    out_dir: Path,
    detail_rows: list[IterationExampleRow],
    generation_rows: list[GenerationRow],
    bootstrap_samples: int,
    seed: int,
) -> None:
    _write_csv(out_dir / "iteration_details.csv", IterationExampleRow, detail_rows)
    _write_csv(out_dir / "generations.csv", GenerationRow, generation_rows)
    summary = summarize_rows(detail_rows, generation_rows, bootstrap_samples=bootstrap_samples, seed=seed)
    _write_csv(out_dir / "summary.csv", SummaryRow, summary)
    (out_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    _write_plot(out_dir / "summary.png", summary)
    _write_plot_svg(out_dir / "summary.svg", summary)


def merge_shard_outputs(
    shard_dirs: list[str | Path],
    out_dir: str | Path,
    *,
    bootstrap_samples: int,
    seed: int,
) -> None:
    detail_rows: list[IterationExampleRow] = []
    generation_rows: list[GenerationRow] = []
    for shard_dir in shard_dirs:
        shard = Path(shard_dir)
        detail_rows.extend(_read_iteration_rows(shard / "iteration_details.csv"))
        generation_rows.extend(_read_generation_rows(shard / "generations.csv"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(out, detail_rows, generation_rows, bootstrap_samples, seed)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "merged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "shards": [str(Path(path)) for path in shard_dirs],
                "bootstrap_samples": bootstrap_samples,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def summary_markdown(rows: list[SummaryRow]) -> str:
    lines = [
        "| Iterations | GSM8K acc | Raw synergy, nats/token [95% CI] | Full NLL | n |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.iteration} | {100.0 * row.accuracy:.2f} | "
            f"{row.mean_synergy_raw:.4f} [{row.raw_ci_low:.4f}, {row.raw_ci_high:.4f}] | "
            f"{row.mean_full_nll:.4f} | {row.n_examples} |"
        )
    return "\n".join(lines) + "\n"


def extract_flexible(text: str) -> str | None:
    for extractor in (_extract_boxed_answer, _extract_hash_answer, _extract_answer_is):
        value = extractor(text)
        if value is not None:
            return _normalize_answer(value)
    matches = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", text)
    values = [a or b for a, b in matches]
    return _normalize_answer(values[-1]) if values else None


def answers_match(prediction: str | None, target: str | None) -> bool:
    if not prediction or not target:
        return False
    try:
        return abs(float(prediction) - float(target)) < 1e-5
    except ValueError:
        return _normalize_answer(prediction) == _normalize_answer(target)


def target_answer(text: str) -> str | None:
    if "####" in text:
        text = text.rsplit("####", 1)[-1]
    return extract_flexible(text)


def format_prompt(tokenizer, question: str, *, prompt_style: str) -> str:
    if prompt_style == "plain":
        return question
    if prompt_style == "legacy":
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"Question: {question}\nAnswer:"
    raise ValueError(f"unsupported prompt_style: {prompt_style}")


def _iteration_values(args: argparse.Namespace) -> set[int]:
    if args.iteration_list:
        values = {int(item) for item in args.iteration_list.split(",") if item.strip()}
    else:
        values = set(range(args.max_iterations + 1))
    if not values or min(values) < 0:
        raise ValueError("iteration values must be non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep extra Longshot cache iterations on GSM8K accuracy and head synergy.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--num-calibration", type=int, default=256)
    parser.add_argument("--num-splits", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--iteration-list", default="")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--group-fraction", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--max-target-tokens", type=int, default=512)
    parser.add_argument("--max-gen-toks", type=int, default=512)
    parser.add_argument("--target-prefix", default=" ")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-mode", choices=["first", "random"], default="first")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--prompt-style", choices=["plain", "legacy"], default="plain")
    parser.add_argument("--prefill-mode", choices=["prefix_trigger", "old_full_prompt"], default="prefix_trigger")
    parser.add_argument("--old-bottleneck-protocol", action="store_true")
    parser.add_argument("--flush-every", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-shards", nargs="*", default=None)
    parser.add_argument("--merge-out-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.old_bottleneck_protocol:
        args.prompt_style = "legacy"
        args.prefill_mode = "old_full_prompt"
        args.sample_mode = "random"
        args.sample_seed = 0
        args.num_examples = 200
        args.max_prompt_tokens = 384
        args.max_target_tokens = 128
        args.max_gen_toks = 256
        args.target_prefix = ""
        args.iteration_list = args.iteration_list or "0,1,2,4,8,16,32"
    if args.merge_shards is not None:
        if not args.merge_out_dir:
            raise SystemExit("--merge-out-dir is required with --merge-shards")
        merge_shard_outputs(args.merge_shards, args.merge_out_dir, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        return
    if not args.out_dir:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        args.out_dir = str(Path(DEFAULT_OUT_DIR) / run_id)
    run_analysis(args)


def _analysis_add_special_tokens(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "legacy_bottleneck_generate", False) and getattr(config, "family", None) in {"llama", "qwen3"})


def _encode(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


@contextmanager
def _null_context():
    yield


def _write_csv(path: Path, row_type, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(row_type.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _read_iteration_rows(path: Path) -> list[IterationExampleRow]:
    return [
        IterationExampleRow(
            iteration=int(row["iteration"]),
            example_id=int(row["example_id"]),
            split_id=int(row["split_id"]),
            synergy_raw=float(row["synergy_raw"]),
            synergy_norm=float(row["synergy_norm"]),
            mean_full_nll=float(row["mean_full_nll"]),
            mean_base_nll=float(row["mean_base_nll"]),
            n_tokens=int(row["n_tokens"]),
        )
        for row in csv.DictReader(io.StringIO(_read_text_with_retry(path)))
    ]


def _read_generation_rows(path: Path) -> list[GenerationRow]:
    return [
        GenerationRow(
            iteration=int(row["iteration"]),
            example_id=int(row["example_id"]),
            target=row["target"],
            prediction=row["prediction"],
            correct=row["correct"] == "True",
            generation=row["generation"],
        )
        for row in csv.DictReader(io.StringIO(_read_text_with_retry(path)))
    ]


def _read_text_with_retry(path: Path, attempts: int = 20, delay_sec: float = 2.0) -> str:
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            last_error = exc
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error
    return path.read_text(encoding="utf-8")


def _write_plot(path: Path, rows: list[SummaryRow]) -> None:
    if not rows:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    x = [row.iteration for row in rows]
    acc = [100.0 * row.accuracy for row in rows]
    raw = [row.mean_synergy_raw for row in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(x, acc, marker="o", color="#1f77b4", label="GSM8K accuracy")
    ax1.set_xlabel("Extra Longshot cache iterations")
    ax1.set_ylabel("GSM8K accuracy (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(x, raw, marker="s", color="#d62728", label="Raw head synergy")
    ax2.set_ylabel("Raw head synergy (nats/token)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_plot_svg(path: Path, rows: list[SummaryRow]) -> None:
    if not rows:
        return
    width, height = 760, 430
    left, right, top, bottom = 70, 80, 28, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [row.iteration for row in rows]
    acc = [100.0 * row.accuracy for row in rows]
    raw = [row.mean_synergy_raw for row in rows]
    x_min, x_max = min(xs), max(xs)
    raw_min, raw_max = min(raw), max(raw)
    raw_pad = max(0.25, 0.05 * (raw_max - raw_min))
    raw_min -= raw_pad
    raw_max += raw_pad

    def x_pos(value: float) -> float:
        return left + (value - x_min) / max(1, x_max - x_min) * plot_w

    def y_acc(value: float) -> float:
        return top + (100.0 - value) / 100.0 * plot_h

    def y_raw(value: float) -> float:
        return top + (raw_max - value) / max(1e-8, raw_max - raw_min) * plot_h

    acc_points = " ".join(f"{x_pos(x):.1f},{y_acc(y):.1f}" for x, y in zip(xs, acc, strict=True))
    raw_points = " ".join(f"{x_pos(x):.1f},{y_raw(y):.1f}" for x, y in zip(xs, raw, strict=True))
    x_ticks = "\n".join(
        f'<text x="{x_pos(x):.1f}" y="{height - 34}" text-anchor="middle" font-size="12">{x}</text>'
        f'<line x1="{x_pos(x):.1f}" y1="{height - bottom}" x2="{x_pos(x):.1f}" y2="{height - bottom + 5}" stroke="#333"/>'
        for x in xs
    )
    acc_ticks = "\n".join(
        f'<text x="{left - 10}" y="{y_acc(y) + 4:.1f}" text-anchor="end" font-size="12" fill="#1f77b4">{y}</text>'
        f'<line x1="{left - 5}" y1="{y_acc(y):.1f}" x2="{left}" y2="{y_acc(y):.1f}" stroke="#333"/>'
        for y in (0, 25, 50, 75, 100)
    )
    raw_ticks_values = [raw_min + i * (raw_max - raw_min) / 4 for i in range(5)]
    raw_ticks = "\n".join(
        f'<text x="{width - right + 10}" y="{y_raw(y) + 4:.1f}" text-anchor="start" font-size="12" fill="#d62728">{y:.1f}</text>'
        f'<line x1="{width - right}" y1="{y_raw(y):.1f}" x2="{width - right + 5}" y2="{y_raw(y):.1f}" stroke="#333"/>'
        for y in raw_ticks_values
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2:.1f}" y="20" text-anchor="middle" font-family="sans-serif" font-size="16">Longshot GSM8K Iteration Sweep</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333"/>
<line x1="{width - right}" y1="{top}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333"/>
{x_ticks}
{acc_ticks}
{raw_ticks}
<polyline points="{acc_points}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
<polyline points="{raw_points}" fill="none" stroke="#d62728" stroke-width="2.5"/>
<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="sans-serif" font-size="13">Extra Longshot cache iterations</text>
<text x="18" y="{height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#1f77b4" transform="rotate(-90 18 {height / 2:.1f})">GSM8K accuracy (%)</text>
<text x="{width - 18}" y="{height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#d62728" transform="rotate(90 {width - 18} {height / 2:.1f})">Raw head synergy (nats/token)</text>
<circle cx="{left + 20}" cy="{top + 18}" r="4" fill="#1f77b4"/><text x="{left + 30}" y="{top + 22}" font-family="sans-serif" font-size="12">Accuracy</text>
<circle cx="{left + 110}" cy="{top + 18}" r="4" fill="#d62728"/><text x="{left + 120}" y="{top + 22}" font-family="sans-serif" font-size="12">Raw synergy</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def _write_manifest(out_dir: Path, args: argparse.Namespace) -> None:
    payload = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "GSM8K plain-question prefix prefill, repeated cache-processor iterations, final prompt token as trigger.",
        "accuracy": "GSM8K boxed answer if present, then #### answer, then answer-is phrase, then last numeric fallback.",
        "synergy_units": "mean answer-token NLL difference in nats/token",
        "ci_method": "bootstrap 95% CI over example means after averaging random head partitions",
        "settings": vars(args),
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_progress(path: Path, completed: int, total: int, start_time: float, *, status: str = "running", skipped: int = 0) -> None:
    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta = (total - completed) / rate if rate > 0 and completed < total else 0.0
    path.write_text(
        json.dumps(
            {
                "status": status,
                "completed_examples": completed,
                "total_examples": total,
                "skipped_examples": skipped,
                "elapsed_sec": elapsed,
                "eta_sec": eta,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_environment() -> None:
    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def _bootstrap_ci(values: list[float], *, bootstrap_samples: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan"), float("nan")
    if bootstrap_samples <= 0:
        mean = _mean(clean)
        return mean, mean
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        sample = [clean[rng.randrange(len(clean))] for _ in clean]
        estimates.append(_mean(sample))
    estimates.sort()
    low = estimates[min(len(estimates) - 1, max(0, int((alpha / 2) * len(estimates))))]
    high = estimates[min(len(estimates) - 1, max(0, int((1 - alpha / 2) * len(estimates))))]
    return low, high


def _normalize_answer(value: str) -> str:
    value = value.lower().strip()
    for old in (",", "$", " "):
        value = value.replace(old, "")
    while value.endswith("."):
        value = value[:-1]
    return value.strip()


def _extract_boxed_answer(text: str) -> str | None:
    marker = r"\boxed"
    start = text.rfind(marker)
    if start < 0:
        return None
    pos = start + len(marker)
    while pos < len(text) and text[pos].isspace():
        pos += 1
    if pos >= len(text):
        return None
    if text[pos] != "{":
        match = re.match(r"(-?[$0-9.,]+)", text[pos:])
        return match.group(1) if match else None

    depth = 0
    body_start = pos + 1
    for idx in range(pos, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[body_start:idx]
    return None


def _extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    tail = text.rsplit("####", 1)[-1]
    match = re.search(r"-?[$0-9][0-9,]*(?:\.[0-9]+)?", tail)
    return match.group(0) if match else None


def _extract_answer_is(text: str) -> str | None:
    matches = re.findall(
        r"(?:answer\s+is|answer:|therefore[:,]?)\s*(?:\\boxed\{)?\s*(-?[$0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    return matches[-1] if matches else None


if __name__ == "__main__":
    main()
