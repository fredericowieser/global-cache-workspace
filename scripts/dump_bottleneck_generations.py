from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from paper7_generation_common import load_examples, request_kwargs


class _Request:
    def __init__(self, prompt: str, kwargs: dict):
        self.args = (prompt, kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump legacy bottleneck_llm raw generations for parity debugging."
    )
    parser.add_argument("--repo", default="/home/f50056118/bottleneck_llm")
    parser.add_argument("--config", default="bgt/Llama3.2-1B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tasks", default="gsm8k_openmath,svamp")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    repo = Path(args.repo)
    out = Path(args.out).resolve()
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import lm_eval.models  # noqa: F401
    import src.model.lm_eval_wrapper  # noqa: F401
    from src.model.lm_eval_wrapper import BGTLM

    model = BGTLM(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        eval_recurrence_steps=1,
    )
    examples = load_examples(
        [task.strip() for task in args.tasks.split(",") if task.strip()],
        limit=args.limit,
    )
    requests = [
        _Request(example.prompt, request_kwargs(example))
        for example in examples
    ]
    generations = model.generate_until(requests)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for example, generation in zip(examples, generations):
            row = {
                "runner": f"bottleneck:bs{args.batch_size}",
                "task": example.task,
                "index": example.index,
                "prompt": example.prompt,
                "target": example.target,
                "generation": generation,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
