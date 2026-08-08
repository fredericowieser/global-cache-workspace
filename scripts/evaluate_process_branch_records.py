from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Sequence

import torch

from gcw.checkpoint import load_checkpoint
from gcw.data.tokenizers import get_tokenizer
from gcw.models.longshot import Longshot
from gcw.process_branch import generate_process_branch
from gcw.process_data import read_process_branch_records
from gcw.process_eval import numeric_answers_match


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate answer-only process-branch generation on process records.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--skip-rows", required=True, type=int)
    parser.add_argument("--take-rows", required=True, type=int)
    parser.add_argument("--prompt-suffix", required=True)
    parser.add_argument("--answer-prefix", default="\n\n")
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--max-new-tokens", default=32, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.out.exists():
        raise FileExistsError(args.out)
    if args.take_rows <= 0:
        raise ValueError("take-rows must be positive")
    if args.cycles < 0:
        raise ValueError("cycles must be non-negative")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, train_state = load_checkpoint(args.checkpoint, device=device)
    if not isinstance(model, Longshot):
        raise TypeError("process-branch evaluation requires a Longshot checkpoint")
    model.eval()
    tokenizer = get_tokenizer(args.tokenizer)
    records = read_process_branch_records(
        args.records,
        skip_rows=args.skip_rows,
        take_rows=args.take_rows,
    )
    if len(records) != args.take_rows:
        raise ValueError(f"requested {args.take_rows} records but loaded {len(records)}")
    answer_prefix_ids = torch.tensor(
        tokenizer.encode(args.answer_prefix),
        dtype=torch.long,
        device=device,
    )
    if answer_prefix_ids.numel() == 0:
        raise ValueError("answer prefix tokenised to an empty sequence")

    rows = []
    correct = 0
    generated_tokens = 0
    truncated = 0
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        for record in records:
            prompt = record.prompt
            if args.prompt_suffix.strip():
                prompt = f"{prompt}\n\n{args.prompt_suffix.strip()}"
            prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
            generated = generate_process_branch(
                model,
                prompt_ids=prompt_ids,
                answer_prefix_ids=answer_prefix_ids,
                cycles=args.cycles,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=int(tokenizer.eot_id),
                valid_vocab_size=tokenizer.get_vocab_size(),
            )
            prediction = tokenizer.decode(generated.tolist())
            is_correct = numeric_answers_match(prediction, record.final_answer)
            is_truncated = (
                generated.numel() == args.max_new_tokens
                and int(generated[-1]) != int(tokenizer.eot_id)
            )
            correct += int(is_correct)
            generated_tokens += generated.numel()
            truncated += int(is_truncated)
            rows.append(
                {
                    "source_id": record.source_id,
                    "target": record.final_answer,
                    "prediction": prediction,
                    "correct": is_correct,
                    "generated_tokens": generated.numel(),
                    "truncated": is_truncated,
                }
            )

    payload = {
        "schema": "gcw.process_branch.record_eval.v1",
        "checkpoint": str(args.checkpoint),
        "processor_sha256": train_state.get("final_processor_sha256"),
        "records": str(args.records),
        "skip_rows": args.skip_rows,
        "take_rows": args.take_rows,
        "cycles": args.cycles,
        "max_new_tokens": args.max_new_tokens,
        "correct": correct,
        "accuracy": correct / len(records),
        "mean_generated_tokens": generated_tokens / len(records),
        "truncated": truncated,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
