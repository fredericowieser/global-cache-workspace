from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch

from gcw.analysis.ablation import HeadMeanAblator, UnigramBaseline, deterministic_masks, synergy_from_losses
from gcw.checkpoint import load_checkpoint
from gcw.data.sft import iter_jsonl
from gcw.eval_support import tokenizer_for_checkpoint, truncate_context_continuation
from gcw.models.cache_state import CacheState


DEFAULT_SHARED_ROOT = Path("/mnt/jfs-01/shared/gcw")
DEFAULT_TASKS = ("gsm8k_openmath", "svamp", "gsm8k_hard", "logiqa", "agieval_gaokao_mathqa")
DEFAULT_FAMILIES = ("llama32_1b", "llama32_3b", "llama31_8b", "qwen3_06b")
DEFAULT_VARIANTS = ("sft", "bt", "bgt")
TASK_LABELS = {
    "gsm8k_openmath": "GSM8K",
    "svamp": "SVAMP",
    "gsm8k_hard": "GSM-Hd",
    "logiqa": "LogiQA",
    "agieval_gaokao_mathqa": "Gaokao",
}


@dataclass(frozen=True)
class CheckpointRecord:
    family: str
    variant: str
    path: str

    @property
    def name(self) -> str:
        return f"{self.family}_{self.variant}"


@dataclass(frozen=True)
class AnalysisExample:
    task: str
    index: int
    prompt: str
    target: str


@dataclass(frozen=True)
class ExampleResult:
    family: str
    variant: str
    checkpoint: str
    task: str
    example_id: int
    split_id: int
    mean_synergy_raw: float
    mean_synergy_norm: float
    mean_full_nll: float
    mean_baseline_nll: float
    n_tokens: int


@dataclass(frozen=True)
class SummaryRow:
    family: str
    variant: str
    task: str
    mean_synergy_raw: float
    std_synergy_raw: float
    raw_ci_low: float
    raw_ci_high: float
    mean_synergy_norm: float
    norm_ci_low: float
    norm_ci_high: float
    n_examples: int
    n_tokens: int
    avg_mean_synergy_raw: float | None = None
    avg_raw_ci_low: float | None = None
    avg_raw_ci_high: float | None = None
    avg_mean_synergy_norm: float | None = None
    avg_norm_ci_low: float | None = None
    avg_norm_ci_high: float | None = None


@dataclass(frozen=True)
class PreparedContinuation:
    ctx_len: int
    cont_ids: torch.Tensor
    ctx_logits: torch.Tensor
    cache: CacheState


def select_checkpoints(
    root: str | Path = DEFAULT_SHARED_ROOT,
    families: Iterable[str] = DEFAULT_FAMILIES,
    variants: Iterable[str] = DEFAULT_VARIANTS,
) -> list[CheckpointRecord]:
    root = Path(root)
    records = []
    for family in families:
        for variant in variants:
            if variant == "sft":
                path = root / "checkpoints" / "sft_backbones" / family / "tf"
            else:
                path = root / "checkpoints" / "bottleneck_llm" / family / variant / "final"
            records.append(CheckpointRecord(str(family), str(variant), str(path)))
    return records


def aggregate_normalized_synergy(raw_values: list[float], useful_values: list[float], eps: float = 1e-6) -> float:
    useful = _mean(useful_values)
    if useful <= eps:
        return float("nan")
    return _mean(raw_values) / useful


@torch.no_grad()
def prepare_continuation(
    model: torch.nn.Module,
    tokenizer,
    example: AnalysisExample,
    *,
    max_length: int,
    max_target_tokens: int,
) -> tuple[PreparedContinuation, list[int]]:
    backbone = _backbone(model)
    device = backbone.get_device()
    ctx, cont = encode_context_continuation(model, tokenizer, example.prompt, example.target, max_length, max_target_tokens)
    ctx_ids = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
    cont_ids = torch.tensor(cont, dtype=torch.long, device=device).unsqueeze(0)

    cache = CacheState([], [], ctx_ids.new_empty(1, 0, dtype=torch.float32))
    ctx_logits, cache = backbone.forward_chunk(ctx_ids, 0, cache)
    if _uses_legacy_boundary_processor(model):
        cache = model.cache_processor(cache, select_all=True)
    return PreparedContinuation(len(ctx), cont_ids, ctx_logits, cache), cont


@torch.no_grad()
def score_prepared_continuation_nll(
    model: torch.nn.Module,
    prepared: PreparedContinuation,
    *,
    ablator: HeadMeanAblator | None = None,
    masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    backbone = _backbone(model)
    if ablator is not None:
        ablator.remove()

    context = ablator.ablated(masks) if ablator is not None else _null_context()
    with context:
        cont_logits, _ = backbone.forward_chunk(prepared.cont_ids, prepared.ctx_len, prepared.cache)

    logits = torch.cat([prepared.ctx_logits[:, -1:, :], cont_logits[:, :-1, :]], dim=1)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    target = prepared.cont_ids[0]
    return -logp.gather(-1, target[:, None]).squeeze(-1).detach().cpu()


@torch.no_grad()
def score_continuation_nll(
    model: torch.nn.Module,
    tokenizer,
    example: AnalysisExample,
    *,
    max_length: int,
    max_target_tokens: int,
    ablator: HeadMeanAblator | None = None,
    masks: dict[int, torch.Tensor] | None = None,
) -> torch.Tensor:
    prepared, _ = prepare_continuation(
        model,
        tokenizer,
        example,
        max_length=max_length,
        max_target_tokens=max_target_tokens,
    )
    return score_prepared_continuation_nll(model, prepared, ablator=ablator, masks=masks)


def encode_context_continuation(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    target: str,
    max_length: int,
    max_target_tokens: int,
) -> tuple[list[int], list[int]]:
    prompt_ids = _encode(tokenizer, prompt, add_special_tokens=_analysis_add_special_tokens(model))
    target_ids = _encode(tokenizer, target, add_special_tokens=False)[:max_target_tokens]
    if not target_ids:
        raise ValueError("target produced no tokens")
    if not prompt_ids:
        prompt_ids = [int(getattr(tokenizer, "eot_id", 0))]
    return truncate_context_continuation(prompt_ids, target_ids, max_length)


def analyze_checkpoint(
    record: CheckpointRecord,
    examples_by_task: dict[str, list[AnalysisExample]],
    calibration_by_task: dict[str, list[AnalysisExample]],
    *,
    out_dir: Path,
    num_calibration: int,
    num_splits: int,
    group_fraction: float,
    max_length: int,
    max_target_tokens: int,
    seed: int,
    device: str,
) -> list[ExampleResult]:
    model, cfg, state = load_checkpoint(record.path, device=device)
    model.eval()
    tokenizer = tokenizer_for_checkpoint(cfg, state)
    ablator = HeadMeanAblator(model)
    calibration = _flatten_examples(calibration_by_task, num_calibration)
    baseline = fit_baseline(tokenizer, calibration, cfg.vocab_size)
    ablator.capture(calibration_batches(model, tokenizer, calibration, max_length=max_length, max_target_tokens=max_target_tokens))

    rows: list[ExampleResult] = []
    checkpoint_dir = out_dir / record.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    examples_path = checkpoint_dir / "examples.jsonl"
    with examples_path.open("w", encoding="utf-8") as f:
        for task, examples in examples_by_task.items():
            for example in examples:
                prepared, cont = prepare_continuation(
                    model,
                    tokenizer,
                    example,
                    max_length=max_length,
                    max_target_tokens=max_target_tokens,
                )
                full = score_prepared_continuation_nll(model, prepared)
                base = baseline.nll(cont)
                for split_id in range(num_splits):
                    mask_a, mask_b = deterministic_masks(
                        ablator,
                        split_id=split_id,
                        seed=seed + example.index * 31,
                        group_fraction=group_fraction,
                    )
                    loss_a = score_prepared_continuation_nll(
                        model,
                        prepared,
                        ablator=ablator,
                        masks=mask_a,
                    )
                    loss_b = score_prepared_continuation_nll(
                        model,
                        prepared,
                        ablator=ablator,
                        masks=mask_b,
                    )
                    raw, norm = synergy_from_losses(full, loss_a, loss_b, base)
                    row = ExampleResult(
                        family=record.family,
                        variant=record.variant,
                        checkpoint=record.path,
                        task=task,
                        example_id=example.index,
                        split_id=split_id,
                        mean_synergy_raw=raw,
                        mean_synergy_norm=norm,
                        mean_full_nll=float(full.mean().item()),
                        mean_baseline_nll=float(base.mean().item()),
                        n_tokens=int(full.numel()),
                    )
                    rows.append(row)
                    f.write(json.dumps(asdict(row)) + "\n")
    summaries = summarize(rows)
    write_summary_files(checkpoint_dir, summaries)
    return rows


def fit_baseline(tokenizer, examples: list[AnalysisExample], vocab_size: int) -> UnigramBaseline:
    baseline = UnigramBaseline(vocab_size)
    for example in examples:
        baseline.update(_encode(tokenizer, f"{example.prompt}\n{example.target}", add_special_tokens=False))
    return baseline


def calibration_batches(
    model: torch.nn.Module,
    tokenizer,
    examples: list[AnalysisExample],
    *,
    max_length: int,
    max_target_tokens: int,
) -> Iterable[torch.Tensor]:
    backbone = _backbone(model)
    device = backbone.get_device()
    for example in examples:
        ctx, cont = encode_context_continuation(model, tokenizer, example.prompt, example.target, max_length, max_target_tokens)
        yield torch.tensor((ctx + cont)[-max_length:], dtype=torch.long, device=device).unsqueeze(0)


def summarize(rows: list[ExampleResult], *, bootstrap_samples: int = 1000, seed: int = 0, norm_eps: float = 1e-6) -> list[SummaryRow]:
    grouped: dict[tuple[str, str, str], list[ExampleResult]] = {}
    for row in rows:
        grouped.setdefault((row.family, row.variant, row.task), []).append(row)

    summaries = []
    by_model: dict[tuple[str, str], list[SummaryRow]] = {}
    for (family, variant, task), task_rows in sorted(grouped.items()):
        example_rows = _collapse_example_rows(task_rows)
        raw = [row["raw"] for row in example_rows]
        useful = [row["useful"] for row in example_rows]
        n_tokens = sum(int(row["n_tokens"]) for row in example_rows)
        raw_ci = _bootstrap_ci(raw, bootstrap_samples=bootstrap_samples, seed=_stable_seed(seed, family, variant, task))
        norm_ci = _bootstrap_ci(
            example_rows,
            bootstrap_samples=bootstrap_samples,
            seed=_stable_seed(seed + 17, family, variant, task),
            statistic=lambda sample: aggregate_normalized_synergy(
                [float(row["raw"]) for row in sample],
                [float(row["useful"]) for row in sample],
                eps=norm_eps,
            ),
        )
        summary = SummaryRow(
            family=family,
            variant=variant,
            task=task,
            mean_synergy_raw=_mean(raw),
            std_synergy_raw=_std(raw),
            raw_ci_low=raw_ci[0],
            raw_ci_high=raw_ci[1],
            mean_synergy_norm=aggregate_normalized_synergy(raw, useful, eps=norm_eps),
            norm_ci_low=norm_ci[0],
            norm_ci_high=norm_ci[1],
            n_examples=len(example_rows),
            n_tokens=n_tokens,
        )
        summaries.append(summary)
        by_model.setdefault((family, variant), []).append(summary)

    out = []
    for summary in summaries:
        model_rows = by_model[(summary.family, summary.variant)]
        task_raw_means = [row.mean_synergy_raw for row in model_rows]
        task_means = [row.mean_synergy_norm for row in model_rows]
        out.append(
            SummaryRow(
                **{
                    **asdict(summary),
                    "avg_mean_synergy_raw": _mean(task_raw_means),
                    "avg_raw_ci_low": _mean([row.raw_ci_low for row in model_rows]),
                    "avg_raw_ci_high": _mean([row.raw_ci_high for row in model_rows]),
                    "avg_mean_synergy_norm": _mean(task_means),
                    "avg_norm_ci_low": _mean([row.norm_ci_low for row in model_rows]),
                    "avg_norm_ci_high": _mean([row.norm_ci_high for row in model_rows]),
                }
            )
        )
    return out


def write_summary_files(path: Path, rows: list[SummaryRow]) -> None:
    write_summary_csv(path / "summary.csv", rows)
    (path / "summary.md").write_text(summary_markdown(rows), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(SummaryRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summary_markdown(rows: list[SummaryRow]) -> str:
    header = "| Family | Variant | Task | Raw mean [95% CI] | Agg norm [95% CI] | n | tokens | AVG raw [95% CI] | AVG norm [95% CI] |"
    sep = "|---|---|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        task = TASK_LABELS.get(row.task, row.task)
        lines.append(
            f"| {row.family} | {row.variant} | {task} | "
            f"{row.mean_synergy_raw:.4f} [{row.raw_ci_low:.4f}, {row.raw_ci_high:.4f}] | "
            f"{_fmt_optional(row.mean_synergy_norm)} [{_fmt_optional(row.norm_ci_low)}, {_fmt_optional(row.norm_ci_high)}] | "
            f"{row.n_examples} | {row.n_tokens} | "
            f"{_fmt_optional(row.avg_mean_synergy_raw)} [{_fmt_optional(row.avg_raw_ci_low)}, {_fmt_optional(row.avg_raw_ci_high)}] | "
            f"{_fmt_optional(row.avg_mean_synergy_norm)} [{_fmt_optional(row.avg_norm_ci_low)}, {_fmt_optional(row.avg_norm_ci_high)}] |"
        )
    return "\n".join(lines) + "\n"


def load_examples_for_tasks(tasks: list[str], *, limit: int | None, calibration_limit: int) -> tuple[dict[str, list[AnalysisExample]], dict[str, list[AnalysisExample]]]:
    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR")
    task_dict = get_task_dict(tasks, manager)
    examples = {}
    calibration = {}
    for task_name in tasks:
        task = task_dict[task_name]
        examples[task_name] = _task_examples(task_name, task, "test", limit)
        calibration[task_name] = _task_examples(task_name, task, "calibration", calibration_limit)
    return examples, calibration


def load_examples_jsonl(path: str | Path, tasks: list[str], limit: int | None) -> tuple[dict[str, list[AnalysisExample]], dict[str, list[AnalysisExample]]]:
    grouped: dict[str, list[AnalysisExample]] = {task: [] for task in tasks}
    for idx, row in enumerate(iter_jsonl(path)):
        task = str(row.get("task", tasks[0]))
        if task not in grouped:
            continue
        grouped[task].append(AnalysisExample(task, int(row.get("index", idx)), str(row["prompt"]), str(row["target"])))
    if limit is not None:
        grouped = {task: rows[:limit] for task, rows in grouped.items()}
    return grouped, grouped


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = select_checkpoints(args.root, args.families, args.variants)
    if args.limit_checkpoints is not None:
        checkpoints = checkpoints[: args.limit_checkpoints]
    if args.dry_run:
        payload = {"checkpoints": [asdict(record) for record in checkpoints], "tasks": args.tasks}
        print(json.dumps(payload, indent=2))
        return

    if args.examples_jsonl:
        examples_by_task, calibration_by_task = load_examples_jsonl(args.examples_jsonl, args.tasks, args.num_examples)
    else:
        examples_by_task, calibration_by_task = load_examples_for_tasks(
            args.tasks,
            limit=args.num_examples,
            calibration_limit=args.num_calibration,
        )

    manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(args.root),
        "tasks": args.tasks,
        "checkpoints": [asdict(record) for record in checkpoints],
        "settings": {key: value for key, value in vars(args).items() if key not in {"dry_run"}},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_rows: list[ExampleResult] = []
    for idx, record in enumerate(checkpoints):
        _write_progress(out_dir, checkpoints, idx, record, "running")
        rows = analyze_checkpoint(
            record,
            examples_by_task,
            calibration_by_task,
            out_dir=out_dir,
            num_calibration=args.num_calibration,
            num_splits=args.num_splits,
            group_fraction=args.group_fraction,
            max_length=args.max_length,
            max_target_tokens=args.max_target_tokens,
            seed=args.seed,
            device=args.device,
        )
        all_rows.extend(rows)
        combined = summarize(all_rows, bootstrap_samples=args.bootstrap_samples, seed=args.seed, norm_eps=args.norm_eps)
        write_summary_csv(out_dir / "combined_summary.csv", combined)
        (out_dir / "combined_summary.md").write_text(summary_markdown(combined), encoding="utf-8")
        _write_progress(out_dir, checkpoints, idx, record, "completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure attention-head synergy for GCW checkpoints.")
    parser.add_argument("--root", default=str(DEFAULT_SHARED_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--families", nargs="+", default=list(DEFAULT_FAMILIES))
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--num-examples", type=int, default=256)
    parser.add_argument("--num-calibration", type=int, default=512)
    parser.add_argument("--num-splits", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--norm-eps", type=float, default=1e-6)
    parser.add_argument("--group-fraction", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-target-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-checkpoints", type=int, default=None)
    parser.add_argument("--examples-jsonl", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.out_dir:
        run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        args.out_dir = str(DEFAULT_SHARED_ROOT / "results" / "head_synergy" / run_id)
    run(args)


def _task_examples(task_name: str, task, split: str, limit: int | None) -> list[AnalysisExample]:
    docs = _docs_for_split(task, split)
    examples = []
    for idx, doc in enumerate(docs):
        if limit is not None and idx >= limit:
            break
        prompt, target = _prompt_target(task, doc)
        if prompt and target:
            examples.append(AnalysisExample(task_name, idx, prompt, target))
    return examples


def _docs_for_split(task, split: str):
    candidates = ("training_docs", "validation_docs", "test_docs") if split == "calibration" else ("test_docs", "validation_docs")
    for name in candidates:
        fn = getattr(task, name, None)
        if fn is None:
            continue
        try:
            docs = fn()
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        if docs is not None:
            return docs
    raise ValueError(f"no usable {split} docs for task {getattr(task, 'name', task)}")


def _prompt_target(task, doc) -> tuple[str, str]:
    prompt = str(task.doc_to_text(doc))
    choices_fn = getattr(task, "doc_to_choice", None)
    output_type = getattr(task, "OUTPUT_TYPE", None) or getattr(task, "output_type", None)
    if output_type == "multiple_choice" and choices_fn is not None:
        try:
            choices = choices_fn(doc)
            target = task.doc_to_target(doc)
            return prompt, str(_choice_text(choices, target))
        except (AssertionError, KeyError, TypeError, ValueError, IndexError):
            pass
    return prompt, str(task.doc_to_target(doc))


def _choice_text(choices, target) -> str:
    if isinstance(target, int):
        return choices[target]
    if isinstance(target, str) and target.isdigit():
        return choices[int(target)]
    if isinstance(target, str) and len(target) == 1 and target.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return choices[ord(target.upper()) - ord("A")]
    return str(target)


def _flatten_examples(examples_by_task: dict[str, list[AnalysisExample]], limit: int) -> list[AnalysisExample]:
    out = []
    for task in examples_by_task:
        out.extend(examples_by_task[task])
        if len(out) >= limit:
            return out[:limit]
    return out


def _write_progress(out_dir: Path, records: list[CheckpointRecord], index: int, record: CheckpointRecord, status: str) -> None:
    payload = {
        "status": status,
        "checkpoint_index": index,
        "checkpoint_count": len(records),
        "checkpoint": asdict(record),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _uses_legacy_boundary_processor(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "variant", None) in {"bt", "bgt"}
        and hasattr(model, "backbone")
        and hasattr(model, "cache_processor")
    )


def _analysis_add_special_tokens(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    return bool(getattr(config, "legacy_bottleneck_generate", False) and getattr(config, "family", None) in {"llama", "qwen3"})


def _encode(tokenizer, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
    except TypeError:
        return list(tokenizer.encode(text))


def _backbone(model: torch.nn.Module):
    return model.backbone if hasattr(model, "backbone") else model


@contextmanager
def _null_context():
    yield


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return float(sum(clean) / len(clean)) if clean else float("nan")


def _std(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    if len(clean) < 2:
        return 0.0
    mean = _mean(clean)
    return float(math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)))


def _fmt_optional(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.4f}"


def _collapse_example_rows(rows: list[ExampleResult]) -> list[dict[str, float]]:
    grouped: dict[int, list[ExampleResult]] = {}
    for row in rows:
        grouped.setdefault(row.example_id, []).append(row)

    collapsed = []
    for example_id in sorted(grouped):
        example_rows = grouped[example_id]
        first = example_rows[0]
        collapsed.append(
            {
                "raw": _mean([row.mean_synergy_raw for row in example_rows]),
                "useful": float(first.mean_baseline_nll - first.mean_full_nll),
                "n_tokens": float(first.n_tokens),
            }
        )
    return collapsed


def _bootstrap_ci(
    values,
    *,
    bootstrap_samples: int,
    seed: int,
    statistic=None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if bootstrap_samples <= 0:
        stat = _mean(values) if statistic is None else statistic(values)
        return stat, stat

    rng = random.Random(seed)
    n = len(values)
    statistic = statistic or _mean
    estimates = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimate = statistic(sample)
        if math.isfinite(estimate):
            estimates.append(float(estimate))
    if not estimates:
        return float("nan"), float("nan")
    estimates.sort()
    lo = estimates[min(len(estimates) - 1, max(0, int((alpha / 2) * len(estimates))))]
    hi = estimates[min(len(estimates) - 1, max(0, int((1 - alpha / 2) * len(estimates))))]
    return lo, hi


def _stable_seed(seed: int, *parts: str) -> int:
    out = seed
    for part in parts:
        for char in part:
            out = (out * 33 + ord(char)) % 2_147_483_647
    return out


if __name__ == "__main__":
    main()
