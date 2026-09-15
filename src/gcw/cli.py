from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal


def convert_hf(*args: Any, **kwargs: Any) -> Any:
    from gcw.convert import convert_hf as implementation

    return implementation(*args, **kwargs)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _eval_batch_size(value: str) -> int | Literal["auto"]:
    if value == "auto":
        return value
    return _positive_int(value)


def _boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise argparse.ArgumentTypeError("value must be true or false")
    return normalized == "true"


def _resolve_enable_thinking(
    value: bool | None,
    *,
    checkpoint: str | Path | None,
    apply_chat_template: bool,
) -> bool:
    if value is not None:
        return value
    if not apply_chat_template or checkpoint is None:
        return False
    root = Path(checkpoint)
    candidates = (
        (root / "manifest.json", "training_fingerprint"),
        (root / "run_config.json", None),
    )
    for path, nested_key in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid evaluation provenance file {path}"
            ) from exc
        source = payload.get(nested_key) if nested_key is not None else payload
        data = source.get("data") if isinstance(source, dict) else None
        if not isinstance(data, dict) or "sft_enable_thinking" not in data:
            continue
        enable_thinking = data["sft_enable_thinking"]
        if type(enable_thinking) is not bool:
            raise ValueError(f"{path} data.sft_enable_thinking must be boolean")
        return enable_thinking
    return False


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
        legacy.add_argument("--batch-size", type=_eval_batch_size, default=8)
        legacy.add_argument("--max-batch-size", type=_positive_int, default=64)
        legacy.add_argument(
            "--apply-chat-template",
            nargs="?",
            const=True,
            default=True,
            type=_boolean,
        )
        legacy.add_argument(
            "--enable-thinking",
            nargs="?",
            const=True,
            default=None,
            type=_boolean,
        )
        legacy.add_argument(
            "--dtype",
            choices=["auto", "bfloat16", "float16", "float32"],
            default="auto",
        )
        legacy.add_argument(
            "--generation-process-every", type=_positive_int, default=None
        )
        legacy.add_argument(
            "--head-selection-method", choices=["random"], default=None
        )
        legacy.add_argument("--out", default=None)
    for suite in ("all", "math", "general", "tasks"):
        command = eval_sub.add_parser(suite)
        target = command.add_mutually_exclusive_group(required=True)
        target.add_argument("--checkpoint")
        target.add_argument("--hf-model")
        command.add_argument("--processor", choices=["on", "off"], default="on")
        command.add_argument(
            "--diffusion-eval-solver-steps", type=_positive_int
        )
        command.add_argument(
            "--generation-process-every", type=_positive_int, default=None
        )
        command.add_argument("--limit", type=int, default=None)
        command.add_argument(
            "--batch-size", type=_eval_batch_size, default=8
        )
        command.add_argument(
            "--max-batch-size", type=_positive_int, default=64
        )
        command.add_argument(
            "--apply-chat-template",
            nargs="?",
            const=True,
            default=False,
            type=_boolean,
        )
        command.add_argument(
            "--enable-thinking",
            nargs="?",
            const=True,
            default=None,
            type=_boolean,
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
    chat.add_argument("--diffusion-eval-solver-steps", type=_positive_int)
    chat.add_argument(
        "--generation-process-every", type=_positive_int, default=None
    )
    chat.add_argument("--tasks", nargs="+", required=True)
    chat.add_argument(
        "--enable-thinking",
        nargs="?",
        const=True,
        default=None,
        type=_boolean,
    )
    chat.add_argument("--batch-size", type=int, default=8)
    chat.add_argument("--max-new-tokens", type=int, default=32768)
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
    gsm8k = eval_sub.add_parser("gsm8k-reproduction")
    gsm8k.add_argument("--model-path", required=True)
    gsm8k.add_argument("--target-kind", choices=("hf", "gcw"), default="hf")
    gsm8k.add_argument("--tokenizer-path")
    gsm8k.add_argument("--processor", choices=("on", "off"), default="on")
    gsm8k.add_argument(
        "--scoring-protocol",
        choices=("numeric_last_answer", "lm_eval_flexible_exact"),
        default="lm_eval_flexible_exact",
    )
    gsm8k.add_argument("--output-dir", required=True)
    gsm8k.add_argument("--batch-size", type=int, default=64)
    gsm8k.add_argument("--limit", type=int)
    gsm8k.add_argument(
        "--enable-thinking",
        nargs="?",
        const=True,
        default=None,
        type=_boolean,
    )
    gsm8k.set_defaults(out=None)
    answer_nll = eval_sub.add_parser("answer-nll")
    answer_nll.add_argument("--checkpoint", type=Path, required=True)
    answer_nll.add_argument("--expected-manifest-sha256", required=True)
    answer_nll.add_argument("--data", type=Path, required=True)
    answer_nll.add_argument("--expected-data-sha256", required=True)
    answer_nll.add_argument("--expected-count", type=int, required=True)
    answer_nll.add_argument("--out", type=Path, required=True)
    answer_nll.add_argument("--device", default="cpu")
    answer_nll.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    answer_nll.add_argument("--expected-step", type=int)
    answer_nll.add_argument("--expected-tokenizer-sha256")
    answer_nll.add_argument("--expected-loaded-state-sha256")
    answer_nll.add_argument("--solver-steps", type=int)
    answer_nll.add_argument("--sequence-len", type=int)
    answer_nll.add_argument(
        "--tokenizer", help="Native name: int, cosmo2, or hf:SOURCE"
    )
    answer_nll.add_argument("--response-prefix", default="")
    answer_nll.add_argument("--prompt-suffix", default="")
    answer_nll.add_argument(
        "--apply-chat-template",
        nargs="?",
        const=True,
        default=None,
        type=_boolean,
    )
    answer_nll.add_argument(
        "--enable-thinking",
        nargs="?",
        const=True,
        default=None,
        type=_boolean,
    )
    answer_nll.add_argument(
        "--disjoint-from",
        nargs=3,
        action="append",
        default=[],
        metavar=("JSONL", "SHA256", "COUNT"),
    )
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
    from gcw.config import EvalConfig
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
                config=EvalConfig(batch_size=args.batch_size),
                dtype=args.dtype,
            )
        elif args.suite == "chat":
            from gcw.evaluation.chat_benchmarks import run_chat_benchmarks

            args.enable_thinking = _resolve_enable_thinking(
                args.enable_thinking,
                checkpoint=args.checkpoint,
                apply_chat_template=True,
            )
            results = run_chat_benchmarks(
                tasks=args.tasks,
                config=EvalConfig(
                    batch_size=args.batch_size,
                    apply_chat_template=True,
                    enable_thinking=args.enable_thinking,
                ),
                max_new_tokens=args.max_new_tokens,
                checkpoint=args.checkpoint,
                hf_model=args.hf_model,
                tokenizer_path=args.tokenizer_path,
                processor_enabled=args.processor == "on",
                generation_process_every=args.generation_process_every,
                diffusion_eval_solver_steps=args.diffusion_eval_solver_steps,
            )
        elif args.suite in {"paper7", "paper7-legacy"}:
            from gcw.evaluation.legacy import run_paper7_legacy
            from gcw.evaluation.paper7 import run_paper7

            args.enable_thinking = _resolve_enable_thinking(
                args.enable_thinking,
                checkpoint=args.checkpoint,
                apply_chat_template=args.apply_chat_template,
            )
            eval_config = EvalConfig(
                batch_size=args.batch_size,
                max_batch_size=args.max_batch_size,
                limit=args.limit,
                apply_chat_template=args.apply_chat_template,
                enable_thinking=args.enable_thinking,
            )
            runner = (
                run_paper7_legacy
                if args.suite == "paper7-legacy"
                else run_paper7
            )
            results = runner(
                args.checkpoint,
                config=eval_config,
                dtype=args.dtype,
                generation_process_every=args.generation_process_every,
                head_selection_method=args.head_selection_method,
            )
            args.apply_chat_template = eval_config.apply_chat_template
            args.enable_thinking = eval_config.enable_thinking
        elif args.suite == "gsm8k-reproduction":
            from gcw.evaluation.benchmarks import (
                BenchmarkConfig,
                run_gsm8k_reproduction,
            )

            args.enable_thinking = _resolve_enable_thinking(
                args.enable_thinking,
                checkpoint=(
                    args.model_path if args.target_kind == "gcw" else None
                ),
                apply_chat_template=True,
            )
            results = run_gsm8k_reproduction(
                BenchmarkConfig(
                    model_path=args.model_path,
                    output_dir=args.output_dir,
                    target_kind=args.target_kind,
                    tokenizer_path=args.tokenizer_path,
                    processor_enabled=args.processor == "on",
                    scoring_protocol=args.scoring_protocol,
                    eval_config=EvalConfig(
                        batch_size=args.batch_size,
                        limit=args.limit,
                        apply_chat_template=True,
                        enable_thinking=args.enable_thinking,
                    ),
                )
            )
        elif args.suite == "answer-nll":
            from gcw.evaluation.answer_nll import (
                AnswerNLLData,
                AnswerNLLDataOptions,
                evaluate_answer_nll,
            )

            options = None
            if args.sequence_len is not None or args.tokenizer is not None:
                if args.sequence_len is None or args.tokenizer is None:
                    raise ValueError(
                        "explicit data policy requires both sequence_len and tokenizer"
                    )
                options = AnswerNLLDataOptions(
                    sequence_len=args.sequence_len,
                    tokenizer=args.tokenizer,
                    response_prefix=args.response_prefix,
                    prompt_suffix=args.prompt_suffix,
                )
            elif args.response_prefix or args.prompt_suffix:
                raise ValueError(
                    "data option overrides require sequence_len and tokenizer"
                )
            eval_config = None
            if (
                args.apply_chat_template is not None
                or args.enable_thinking is not None
            ):
                eval_config = EvalConfig(
                    batch_size=1,
                    apply_chat_template=bool(args.apply_chat_template),
                    enable_thinking=bool(args.enable_thinking),
                )
            results = evaluate_answer_nll(
                checkpoint=args.checkpoint,
                expected_manifest_sha256=args.expected_manifest_sha256,
                data=AnswerNLLData(
                    args.data, args.expected_data_sha256, args.expected_count
                ),
                out=args.out,
                options=options,
                config=eval_config,
                device=args.device,
                dtype=args.dtype,
                expected_step=args.expected_step,
                expected_tokenizer_sha256=args.expected_tokenizer_sha256,
                expected_loaded_state_sha256=args.expected_loaded_state_sha256,
                solver_steps=args.solver_steps,
                disjoint_from=[
                    AnswerNLLData(Path(path), digest, int(count))
                    for path, digest, count in args.disjoint_from
                ],
            )
            args.apply_chat_template = results["data_options"][
                "apply_chat_template"
            ]
            args.enable_thinking = results["data_options"]["enable_thinking"]
        else:
            from gcw.evaluation.core import run_tasks

            args.enable_thinking = _resolve_enable_thinking(
                args.enable_thinking,
                checkpoint=args.checkpoint,
                apply_chat_template=args.apply_chat_template,
            )
            runners = {"all": run_all, "math": run_math, "general": run_general}
            common = {
                "checkpoint": args.checkpoint,
                "config": EvalConfig(
                    batch_size=args.batch_size,
                    max_batch_size=args.max_batch_size,
                    limit=args.limit,
                    apply_chat_template=args.apply_chat_template,
                    enable_thinking=args.enable_thinking,
                ),
                "hf_model": args.hf_model,
                "processor_enabled": args.processor == "on",
                "dtype": args.dtype,
                "generation_process_every": args.generation_process_every,
                "diffusion_eval_solver_steps": args.diffusion_eval_solver_steps,
            }
            results = (
                run_tasks(args.tasks, **common)
                if args.suite == "tasks"
                else runners[args.suite](**common)
            )
        if not is_main_process():
            return
        text = json.dumps(results, indent=2, default=str)
        if args.suite == "answer-nll":
            print(json.dumps(results["result"], allow_nan=False))
        elif args.out:
            output = Path(args.out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
        else:
            print(text)
        from gcw.runs import RunRecord

        record = RunRecord.create(
            kind="evaluation",
            experiment_name=f"eval-{args.suite}",
            config=json.loads(json.dumps(vars(args), default=str)),
            state_path=None,
        )
        record.append_log(f"completed evaluation suite {args.suite}")
        summary = json.loads(text)
        if args.suite in {"chat", "gsm8k-reproduction", "answer-nll"}:
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
