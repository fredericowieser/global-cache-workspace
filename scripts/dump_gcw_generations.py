from __future__ import annotations

import argparse
import json
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
from paper7_generation_common import load_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GCW raw generations for paper7 parity debugging.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default="gsm8k_openmath,svamp")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--mode", choices=["reference", "fast"], default="fast")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, state = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tokenizer = _tokenizer_for(cfg, state)
    add_special_tokens = _legacy_generate_add_special_tokens(model)
    examples = load_examples([task.strip() for task in args.tasks.split(",") if task.strip()], limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop_sequences = _stop_token_sequences(tokenizer, ("</s>", "<|end_of_text|>", "<|eot_id|>", "\n\nQuestion:", "\nQuestion:"))

    with out.open("w", encoding="utf-8") as f:
        for example in examples:
            ids = _encode_generate_prompt(tokenizer, example.prompt, cfg.sequence_len, add_special_tokens=add_special_tokens)
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            if args.mode == "fast":
                generated = model.generate_fast(
                    input_ids,
                    max_new_tokens=example.max_gen_toks,
                    eos_token_id=tokenizer.eot_id,
                    stop_sequences=stop_sequences,
                    pad_token_id=tokenizer.eot_id,
                )[0].tolist()
            else:
                generated = model.generate(
                    input_ids,
                    max_new_tokens=example.max_gen_toks,
                    eos_token_id=tokenizer.eot_id,
                )[0].tolist()
            new_tokens = _trim_generated_token_ids(generated[len(ids) :], stop_sequences, tokenizer.eot_id)
            text = _trim_until(tokenizer.decode(new_tokens), tuple())
            row = {
                "runner": f"gcw:{args.mode}",
                "task": example.task,
                "index": example.index,
                "prompt": example.prompt,
                "target": example.target,
                "generation": text,
                "new_token_count": len(new_tokens),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
