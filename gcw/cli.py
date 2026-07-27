from __future__ import annotations

import argparse
import json

from gcw.convert import convert_bottleneck, convert_hf, convert_ppt
from gcw.data.fineweb import prepare_fineweb
from gcw.data.sft import prepare_openmath
from pathlib import Path

from gcw.eval import is_main_process, run_core, run_fineweb, run_math, run_paper7, run_ppt
from gcw.params import write_family_configs
from gcw.train import train_from_config


def main(argv: list[str] | None = None) -> None:
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
    core = eval_sub.add_parser("core")
    core.add_argument("--checkpoint", required=True)
    core.add_argument("--limit", type=int, default=None)
    core.add_argument("--batch-size", type=int, default=8)
    core.add_argument("--out", default=None)
    math = eval_sub.add_parser("math")
    math.add_argument("--checkpoint", required=True)
    math.add_argument("--limit", type=int, default=None)
    math.add_argument("--batch-size", type=int, default=8)
    math.add_argument("--out", default=None)
    ppt = eval_sub.add_parser("ppt")
    ppt.add_argument("--checkpoint", required=True)
    ppt.add_argument("--limit", type=int, default=None)
    ppt.add_argument("--batch-size", type=int, default=8)
    ppt.add_argument("--out", default=None)
    paper7 = eval_sub.add_parser("paper7")
    paper7.add_argument("--checkpoint", required=True)
    paper7.add_argument("--limit", type=int, default=None)
    paper7.add_argument("--batch-size", type=int, default=8)
    paper7.add_argument("--out", default=None)
    fineweb = eval_sub.add_parser("fineweb")
    fineweb.add_argument("--checkpoint", required=True)
    fineweb.add_argument("--data-dir", default=None)
    fineweb.add_argument("--eval-tokens", type=int, default=10 * 524288)
    fineweb.add_argument("--batch-size", type=int, default=16)
    fineweb.add_argument("--out", default=None)

    convert = sub.add_parser("convert")
    convert_sub = convert.add_subparsers(dest="source", required=True)
    hf = convert_sub.add_parser("hf")
    hf.add_argument("--family", choices=["llama", "qwen3", "qwen3_5", "qwen3_5_moe"], required=True)
    hf.add_argument("--model", required=True)
    hf.add_argument("--variant", choices=["tf", "bt", "bgt", "icarus", "longshot"], default="tf")
    hf.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    hf.add_argument("--out", required=True)
    ppt = convert_sub.add_parser("ppt")
    ppt.add_argument("--checkpoint", required=True)
    ppt.add_argument("--meta", required=True)
    ppt.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    ppt.add_argument("--out", required=True)
    bottleneck = convert_sub.add_parser("bottleneck")
    bottleneck.add_argument("--checkpoint", required=True)
    bottleneck.add_argument("--variant", choices=["bt", "bgt"], required=True)
    bottleneck.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    bottleneck.add_argument("--source", default=None)
    bottleneck.add_argument("--tokenizer-source", default=None)
    bottleneck.add_argument("--backbone", choices=["llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "qwen3-0.6b"], default="llama-3.2-1b")
    bottleneck.add_argument("--sequence-len", type=int, default=8192)
    bottleneck.add_argument("--legacy-generate", action="store_true")
    bottleneck.add_argument("--out", required=True)

    params = sub.add_parser("params")
    params_sub = params.add_subparsers(dest="action", required=True)
    make = params_sub.add_parser("make")
    make.add_argument("--family", default="smollm360m")
    make.add_argument("--processor-fraction", type=float, default=0.05)
    make.add_argument("--out", default="configs/model/generated")
    make.add_argument("--target-non-embedding", type=int, default=None)
    make.add_argument("--regime", choices=["added_on_top", "matched_total"], default="added_on_top")

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        if args.kind == "fineweb":
            prepare_fineweb(args.out, shard_tokens=args.shard_tokens, limit=args.limit)
        else:
            rows = prepare_openmath(args.source, args.out, limit=args.limit)
            print(f"Wrote {rows} rows to {args.out}")
    elif args.cmd == "train":
        train_from_config(args.config)
    elif args.cmd == "eval":
        if args.suite == "fineweb":
            results = run_fineweb(args.checkpoint, data_dir=args.data_dir, eval_tokens=args.eval_tokens, batch_size=args.batch_size)
        else:
            runners = {"core": run_core, "math": run_math, "paper7": run_paper7, "ppt": run_ppt}
            results = runners[args.suite](args.checkpoint, limit=args.limit, batch_size=args.batch_size)
        if not is_main_process():
            return
        text = json.dumps(results, indent=2, default=str)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text)
        else:
            print(text)
    elif args.cmd == "convert":
        if args.source == "hf":
            convert_hf(
                args.model,
                args.out,
                family=args.family,
                variant=args.variant,
                dtype=args.dtype,
            )
        elif args.source == "ppt":
            convert_ppt(args.checkpoint, args.meta, args.out, dtype=args.dtype)
        else:
            convert_bottleneck(
                args.checkpoint,
                args.out,
                variant=args.variant,
                dtype=args.dtype,
                source=args.source,
                tokenizer_source=args.tokenizer_source,
                backbone=args.backbone,
                sequence_len=args.sequence_len,
                legacy_generate=args.legacy_generate,
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


if __name__ == "__main__":
    main()
