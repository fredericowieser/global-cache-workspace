from __future__ import annotations

import argparse
import json

from gcw.evaluation.benchmarks import (
    MAX_NEW_TOKENS,
    BenchmarkConfig,
    run_benchmark_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an authenticated Qwen3 GSM8K benchmark matrix"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-kind", choices=("hf", "gcw"), default="hf")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--processor", choices=("on", "off"), default="on")
    parser.add_argument(
        "--scoring-protocol",
        choices=("numeric_last_answer", "lm_eval_flexible_exact"),
        default="lm_eval_flexible_exact",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("thinking_off", "thinking_on"),
        default=("thinking_off", "thinking_on"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_benchmark_matrix(
        BenchmarkConfig(
            model_path=args.model_path,
            output_dir=args.output_dir,
            target_kind=args.target_kind,
            tokenizer_path=args.tokenizer_path,
            processor_enabled=args.processor == "on",
            scoring_protocol=args.scoring_protocol,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            modes=tuple(args.modes),
        )
    )
    print(
        json.dumps(
            {
                mode: {
                    key: cell[key] for key in ("correct", "total", "accuracy")
                }
                for mode, cell in result["cells"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
