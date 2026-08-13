from __future__ import annotations

import argparse
import json
import string
from pathlib import Path

import torch

from gcw.checkpoint import load_checkpoint
from gcw.eval import (
    _encode_generate_prompt,
    _legacy_generate_add_special_tokens,
    _stop_token_sequences,
    _tokenizer_for,
    _trim_generated_token_ids,
    _trim_until,
)

DEFAULT_PROMPTS = [
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. "
    "She sells the rest for $2 per egg. How much money does she make per day?\nAnswer:",
    "Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts are needed total?\nAnswer:",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a cheap generation smoke test for a GCW checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-nonspace-chars", type=int, default=16)
    parser.add_argument("--max-control-ratio", type=float, default=0.02)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, state = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tokenizer = _tokenizer_for(cfg, state)
    add_special_tokens = _legacy_generate_add_special_tokens(model)
    stop_sequences = _stop_token_sequences(
        tokenizer,
        (
            "</s>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "\n\nQuestion:",
            "\nQuestion:",
        ),
    )
    prompts = args.prompt or DEFAULT_PROMPTS

    rows = []
    failures = []
    for prompt in prompts:
        ids = _encode_generate_prompt(
            tokenizer,
            prompt,
            cfg.sequence_len,
            add_special_tokens=add_special_tokens,
        )
        input_ids = torch.tensor(
            ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        if hasattr(model, "generate_fast"):
            generated = model.generate_fast(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eot_id,
                stop_sequences=stop_sequences,
                pad_token_id=tokenizer.eot_id,
            )[0].tolist()
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eot_id,
            )[0].tolist()
        new_tokens = _trim_generated_token_ids(
            generated[len(ids) :], stop_sequences, tokenizer.eot_id
        )
        text = _trim_until(tokenizer.decode(new_tokens), tuple())
        report = _quality_report(text)
        row = {"prompt": prompt, "generation": text, **report}
        rows.append(row)
        if report["nonspace_chars"] < args.min_nonspace_chars:
            failures.append("too few non-space characters")
        if report["control_ratio"] > args.max_control_ratio:
            failures.append("too many control characters")
        if report["repeat_ratio"] > args.max_repeat_ratio:
            failures.append("degenerate repeated-character output")

    output = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output + "\n", encoding="utf-8")
    print(output)
    if failures:
        unique = ", ".join(sorted(set(failures)))
        raise SystemExit(f"generation smoke failed: {unique}")


def _quality_report(text: str) -> dict[str, float | int]:
    printable = set(string.printable)
    nonspace = [ch for ch in text if not ch.isspace()]
    control = [ch for ch in text if ch not in printable and ord(ch) < 0x20]
    if nonspace:
        counts = {ch: nonspace.count(ch) for ch in set(nonspace)}
        repeat_ratio = max(counts.values()) / len(nonspace)
    else:
        repeat_ratio = 1.0
    return {
        "chars": len(text),
        "nonspace_chars": len(nonspace),
        "control_ratio": len(control) / max(1, len(text)),
        "repeat_ratio": repeat_ratio,
    }


if __name__ == "__main__":
    main()
