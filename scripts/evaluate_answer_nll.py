"""Evaluate native answer-only NLL without generation or campaign policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gcw.evaluation.answer_nll import (
    AnswerNLLData,
    AnswerNLLDataOptions,
    evaluate_answer_nll,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--expected-tokenizer-sha256")
    parser.add_argument("--expected-loaded-state-sha256")
    parser.add_argument("--solver-steps", type=int)
    parser.add_argument("--sequence-len", type=int)
    parser.add_argument(
        "--tokenizer", help="Native name: int, cosmo2, or hf:SOURCE"
    )
    parser.add_argument("--response-prefix", default="")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument(
        "--disjoint-from",
        nargs=3,
        action="append",
        default=[],
        metavar=("JSONL", "SHA256", "COUNT"),
    )
    args = parser.parse_args()
    options = None
    if args.sequence_len is not None or args.tokenizer is not None:
        if args.sequence_len is None or args.tokenizer is None:
            parser.error(
                "explicit data policy requires both --sequence-len and --tokenizer"
            )
        options = AnswerNLLDataOptions(
            sequence_len=args.sequence_len,
            tokenizer=args.tokenizer,
            response_prefix=args.response_prefix,
            prompt_suffix=args.prompt_suffix,
            chat_template=args.chat_template,
            enable_thinking=args.enable_thinking,
        )
    elif (
        args.response_prefix
        or args.prompt_suffix
        or args.chat_template
        or args.enable_thinking
    ):
        parser.error(
            "data option overrides require --sequence-len and --tokenizer"
        )
    receipt = evaluate_answer_nll(
        checkpoint=args.checkpoint,
        expected_manifest_sha256=args.expected_manifest_sha256,
        data=AnswerNLLData(
            args.data, args.expected_data_sha256, args.expected_count
        ),
        out=args.out,
        options=options,
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
    print(json.dumps(receipt["result"], allow_nan=False))


if __name__ == "__main__":
    main()
