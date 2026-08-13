from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper7_generation_common import (
    extract_boxed,
    extract_flexible,
    target_answer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two paper7 generation JSONL dumps."
    )
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    left = _load(Path(args.left))
    right = _load(Path(args.right))
    keys = sorted(set(left) & set(right))
    if not keys:
        raise SystemExit("no overlapping task/index pairs")

    text_same = 0
    flexible_same = 0
    boxed_same = 0
    left_correct = 0
    right_correct = 0
    divergences = []
    for key in keys:
        a, b = left[key], right[key]
        gen_a = a["generation"]
        gen_b = b["generation"]
        target = target_answer(a["target"])
        flex_a = extract_flexible(gen_a)
        flex_b = extract_flexible(gen_b)
        boxed_a = extract_boxed(gen_a)
        boxed_b = extract_boxed(gen_b)
        same_text = gen_a == gen_b
        same_flex = flex_a == flex_b
        same_boxed = boxed_a == boxed_b
        text_same += int(same_text)
        flexible_same += int(same_flex)
        boxed_same += int(same_boxed)
        left_correct += int(flex_a is not None and flex_a == target)
        right_correct += int(flex_b is not None and flex_b == target)
        if (not same_text or not same_flex) and len(
            divergences
        ) < args.examples:
            divergences.append(
                {
                    "task": key[0],
                    "index": key[1],
                    "target": target,
                    "left_flexible": flex_a,
                    "right_flexible": flex_b,
                    "left_boxed": boxed_a,
                    "right_boxed": boxed_b,
                    "left_generation": gen_a[:800],
                    "right_generation": gen_b[:800],
                }
            )

    print(f"Compared: {len(keys)}")
    print(
        f"Text identical: {text_same}/{len(keys)} ({text_same / len(keys):.2%})"
    )
    print(
        f"Flexible answer identical: {flexible_same}/{len(keys)} ({flexible_same / len(keys):.2%})"
    )
    print(
        f"Boxed answer identical: {boxed_same}/{len(keys)} ({boxed_same / len(keys):.2%})"
    )
    print(
        f"Left flexible accuracy: {left_correct}/{len(keys)} ({left_correct / len(keys):.2%})"
    )
    print(
        f"Right flexible accuracy: {right_correct}/{len(keys)} ({right_correct / len(keys):.2%})"
    )
    if divergences:
        print("\nDivergences:")
        for item in divergences:
            print(json.dumps(item, indent=2, ensure_ascii=False))


def _load(path: Path) -> dict[tuple[str, int], dict]:
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[(row["task"], int(row["index"]))] = row
    return rows


if __name__ == "__main__":
    main()
