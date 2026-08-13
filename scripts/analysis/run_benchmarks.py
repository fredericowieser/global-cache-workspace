from __future__ import annotations

import argparse
import json

from gcw.evaluation.benchmarks import MAX_NEW_TOKENS, BenchmarkConfig, run_benchmark_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Qwen3-0.6B GSM8K thinking-off/on baselines")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
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
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            modes=tuple(args.modes),
        )
    )
    print(json.dumps({mode: {key: cell[key] for key in ("correct", "total", "accuracy")} for mode, cell in result["cells"].items()}, indent=2))


if __name__ == "__main__":
    main()
