from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def convert_hf(*args: Any, **kwargs: Any) -> Any:
    from gcw.convert import convert_hf as implementation

    return implementation(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    """Build the GCW command-line parser without importing execution systems."""
    parser = argparse.ArgumentParser(prog="gcw")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare_sub = prepare.add_subparsers(dest="kind", required=True)
    fineweb = prepare_sub.add_parser("fineweb")
    fineweb.add_argument("--out", required=True)
    fineweb.add_argument("--shard-tokens", type=int, default=100_000_000)
    fineweb.add_argument("--limit", type=int, default=None)
    openmath = prepare_sub.add_parser("openmath")
    openmath.add_argument("--source", required=True)
    openmath.add_argument("--out", required=True)
    openmath.add_argument("--limit", type=int, default=None)

    train = sub.add_parser("train")
    train.add_argument("--config", required=True)

    eval_p = sub.add_parser("eval")
    eval_sub = eval_p.add_subparsers(dest="suite", required=True)
    for suite in ("paper7", "paper7-legacy"):
        legacy = eval_sub.add_parser(suite)
        legacy.add_argument("--checkpoint", required=True)
        legacy.add_argument("--limit", type=int, default=None)
        legacy.add_argument("--batch-size", type=int, default=8)
        legacy.add_argument(
            "--dtype",
            choices=["auto", "bfloat16", "float16", "float32"],
            default="auto",
        )
        legacy.add_argument("--out", default=None)
    for suite in ("all", "math", "general", "tasks"):
        command = eval_sub.add_parser(suite)
        target = command.add_mutually_exclusive_group(required=True)
        target.add_argument("--checkpoint")
        target.add_argument("--hf-model")
        command.add_argument("--processor", choices=["on", "off"], default="on")
        command.add_argument("--limit", type=int, default=None)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument(
            "--prompt-mode",
            choices=["auto", "plain", "thinking_off", "thinking_on"],
            default="auto",
        )
        command.add_argument(
            "--dtype",
            choices=["auto", "bfloat16", "float16", "float32"],
            default="auto",
        )
        if suite == "tasks":
            command.add_argument("--tasks", nargs="+", required=True)
        command.add_argument("--out", default=None)
    chat = eval_sub.add_parser("chat")
    target = chat.add_mutually_exclusive_group(required=True)
    target.add_argument("--checkpoint")
    target.add_argument("--hf-model")
    chat.add_argument("--tokenizer-path")
    chat.add_argument("--processor", choices=["on", "off"], default="on")
    chat.add_argument("--tasks", nargs="+", required=True)
    chat.add_argument(
        "--mode", choices=["thinking_off", "thinking_on"], required=True
    )
    chat.add_argument("--batch-size", type=int, default=8)
    chat.add_argument("--max-new-tokens", type=int, default=2048)
    chat.add_argument("--out", default=None)
    fineweb = eval_sub.add_parser("fineweb")
    fineweb.add_argument("--checkpoint", required=True)
    fineweb.add_argument("--data-dir", default=None)
    fineweb.add_argument("--eval-tokens", type=int, default=10 * 524288)
    fineweb.add_argument("--batch-size", type=int, default=16)
    fineweb.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    fineweb.add_argument("--out", default=None)
    convert = sub.add_parser("convert")
    convert_sub = convert.add_subparsers(dest="source", required=True)
    hf = convert_sub.add_parser("hf")
    hf.add_argument(
        "--family",
        choices=["llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"],
        required=True,
    )
    hf.add_argument("--model", required=True)
    hf.add_argument(
        "--variant",
        choices=["tf", "bt", "bgt", "icarus", "longshot"],
        default="tf",
    )
    hf.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    hf.add_argument("--out", required=True)
    params = sub.add_parser("params")
    params_sub = params.add_subparsers(dest="action", required=True)
    make = params_sub.add_parser("make")
    make.add_argument("--family", default="smollm360m")
    make.add_argument("--processor-fraction", type=float, default=0.05)
    make.add_argument("--out", default="configs/model/generated")
    make.add_argument("--target-non-embedding", type=int, default=None)
    make.add_argument(
        "--regime",
        choices=["added_on_top", "matched_total"],
        default="added_on_top",
    )

    return parser


def dispatch(args: argparse.Namespace) -> None:
    """Execute a parsed GCW command."""
    from gcw.data.fineweb import prepare_fineweb
    from gcw.data.sft import prepare_openmath
    from gcw.evaluation.core import (
        is_main_process,
        run_all,
        run_fineweb,
        run_general,
        run_math,
    )
    from gcw.params import write_family_configs
    from gcw.training.core import train_from_config

    if args.cmd == "prepare":
        if args.kind == "fineweb":
            prepare_fineweb(
                args.out, shard_tokens=args.shard_tokens, limit=args.limit
            )
        else:
            rows = prepare_openmath(args.source, args.out, limit=args.limit)
            print(f"Wrote {rows} rows to {args.out}")
    elif args.cmd == "train":
        train_from_config(args.config)
    elif args.cmd == "eval":
        if args.suite == "fineweb":
            results = run_fineweb(
                args.checkpoint,
                data_dir=args.data_dir,
                eval_tokens=args.eval_tokens,
                batch_size=args.batch_size,
                dtype=args.dtype,
            )
        elif args.suite == "chat":
            from gcw.evaluation.chat_benchmarks import run_chat_benchmarks

            results = run_chat_benchmarks(
                tasks=args.tasks,
                mode=args.mode,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                checkpoint=args.checkpoint,
                hf_model=args.hf_model,
                tokenizer_path=args.tokenizer_path,
                processor_enabled=args.processor == "on",
            )
        elif args.suite in {"paper7", "paper7-legacy"}:
            from gcw.evaluation.legacy import run_paper7_legacy

            results = run_paper7_legacy(
                args.checkpoint,
                limit=args.limit,
                batch_size=args.batch_size,
                dtype=args.dtype,
            )
        else:
            from gcw.evaluation.core import run_tasks

            runners = {"all": run_all, "math": run_math, "general": run_general}
            common = {
                "checkpoint": args.checkpoint,
                "limit": args.limit,
                "batch_size": args.batch_size,
                "prompt_mode": args.prompt_mode,
                "hf_model": args.hf_model,
                "processor_enabled": args.processor == "on",
                "dtype": args.dtype,
            }
            results = (
                run_tasks(args.tasks, **common)
                if args.suite == "tasks"
                else runners[args.suite](**common)
            )
        if not is_main_process():
            return
        text = json.dumps(results, indent=2, default=str)
        if args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        else:
            print(text)
        from gcw.runs import RunRecord

        record = RunRecord.create(
            kind="evaluation",
            experiment_name=f"eval-{args.suite}",
            config=vars(args),
            state_path=None,
        )
        record.append_log(f"completed evaluation suite {args.suite}")
        summary = json.loads(text)
        if args.suite == "chat":
            summary = {
                key: value for key, value in summary.items() if key != "rows"
            }
        record.finish(summary=summary)
    elif args.cmd == "convert":
        if args.source == "hf":
            convert_hf(
                args.model,
                args.out,
                family=args.family,
                variant=args.variant,
                dtype=args.dtype,
            )
    elif args.cmd == "params":
        print(
            json.dumps(
                write_family_configs(
                    args.out,
                    family=args.family,
                    fraction=args.processor_fraction,
                    regime=args.regime,
                    target_non_embedding=args.target_non_embedding,
                ),
                indent=2,
            )
        )


def main(argv: list[str] | None = None) -> None:
    """Parse and execute a GCW command."""
    dispatch(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
