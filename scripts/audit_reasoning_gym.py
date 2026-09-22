from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from gcw.data.sft import sft_batch
from gcw.data.tokenizers import get_tokenizer


def length_statistics(values):
    result = dict(
        count=int(values.size), mean=None, std=None, median=None, p95=None
    )
    if values.size:
        result.update(
            mean=float(values.mean()),
            std=float(values.std()),
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            min=int(values.min()),
            max=int(values.max()),
            total=int(values.sum()),
        )
    return result


def truncation_statistics(counts, cap):
    prompts, answers = counts[:, 0], counts[:, 1]
    totals = prompts + answers
    truncated = totals > cap
    retained = np.minimum(answers, np.maximum(cap - prompts, 0))
    removed = answers - retained
    return {
        "sequence_len": cap,
        "rows": len(counts),
        "truncated_rows": int(truncated.sum()),
        "truncated_percent": float(100 * truncated.mean()),
        "prompt_only_rows": int((prompts >= cap).sum()),
        "prompt_only_percent_all_rows": float(100 * (prompts >= cap).mean()),
        "eos_only_truncated_rows": int((totals == cap + 1).sum()),
        "original_total_tokens_truncated_rows": length_statistics(
            totals[truncated]
        ),
        "removed_total_tokens_truncated_rows": length_statistics(
            totals[truncated] - cap
        ),
        "retained_answer_tokens_truncated_rows": length_statistics(
            retained[truncated]
        ),
        "removed_answer_tokens_truncated_rows": length_statistics(
            removed[truncated]
        ),
        "lost_answer_token_percent": float(100 * removed.sum() / answers.sum()),
    }


def audit_dataset(source: Path, caps: list[int], split: str = "train"):
    if not caps or min(caps) <= 0:
        raise ValueError("Sequence lengths must be positive")
    manifest = json.loads((source / "manifest.json").read_bytes())
    required_format = {
        "sft_chat_template": True,
        "sft_enable_thinking": False,
        "sft_loss_scope": "answer_only",
        "response_prefix": "",
        "prompt_suffix": "",
    }
    if any(
        manifest["format"].get(key) != value
        for key, value in required_format.items()
    ):
        raise ValueError("Audit requires the prepared Reasoning Gym SFT format")
    expected = manifest["files"][split]
    tasks = sorted(manifest["tasks"])
    task_indices = {task: index for index, task in enumerate(tasks)}
    lengths = []
    examples = {}
    digest = hashlib.sha256()
    aggregates = defaultdict(Counter)
    quality = defaultdict(Counter)
    with (source / expected["file"]).open("rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            row = json.loads(line)
            task = row["task"]
            prompt, target = row["prompt_tokens"], row["target_tokens"]
            total = prompt + target
            if (
                row["split"] != split
                or total != row["sequence_tokens"]
                or min(prompt, target) <= 0
            ):
                raise ValueError("Invalid row split or token counts")
            lengths.append((prompt, target, task_indices[task]))
            aggregates[task].update(
                rows=1, sequence_tokens=total, target_tokens=target
            )
            quality[task]["rows"] += 1
            for prefix in (24, 384, 768):
                quality[task][f"rows_in_first_{prefix}"] += index < prefix
            for cap in caps:
                examples.setdefault((task, cap, total > cap), row)
            if task == "gcd":
                quality[task][f"answer_{row['final_answer']}"] += 1
            elif task == "manipulate_matrix":
                cells = [int(value) for value in row["final_answer"].split()]
                for limit in (1, 2, 4):
                    quality[task][f"final_cells_le_{limit}"] += (
                        len(cells) <= limit
                    )
                quality[task]["all_zero"] += not any(cells)
                quality[task]["operations"] += row["rendered_steps"]
                quality[task]["unchanged_operations"] += (
                    row["rendered_steps"] - row["trace_steps"]
                )
            elif task == "string_splitting":
                for name, value in zip(
                    "ABCXYZ", row["final_answer"].split(), strict=True
                ):
                    quality[task][f"{name}_zero"] += value == "0"
                quality[task]["rule_applications"] += row["trace_steps"]
                quality[task]["rendered_groups"] += row["rendered_steps"]
            elif task == "color_cube_rotation":
                initial = {
                    side: color
                    for color, side in re.findall(
                        r"- a (\w+) (\w+) side", row["prompt"]
                    )
                }
                side = re.search(r"color of the (\w+) side", row["prompt"])[1]
                quality[task]["unchanged_queried_colour"] += (
                    row["final_answer"] == initial[side]
                )
    if (
        digest.hexdigest() != expected["sha256"]
        or len(lengths) != expected["rows"]
    ):
        raise ValueError("Source does not match its manifest checksum/count")
    for task, aggregate in aggregates.items():
        recorded = manifest["tasks"][task][split]
        if aggregate["rows"] != recorded["retained"] or any(
            aggregate[field] != recorded[field]["total"]
            for field in ("sequence_tokens", "target_tokens")
        ):
            raise ValueError(f"Stored counts disagree with manifest for {task}")
    tokenizer = get_tokenizer("hf:" + manifest["tokenizer"])
    if tokenizer.identity_sha256() != manifest["tokenizer_identity_sha256"]:
        raise ValueError("Tokenizer identity differs from preparation")
    for (_, cap, _), row in examples.items():
        x, y = sft_batch(
            [row],
            tokenizer,
            max_length=cap,
            chat_template=True,
            enable_thinking=False,
        )
        if x.shape[1] != min(cap, row["sequence_tokens"]) or (
            y != -100
        ).sum().item() != min(
            row["target_tokens"], max(cap - row["prompt_tokens"], 0)
        ):
            raise ValueError("Stored admission counts disagree with sft_batch")
    counts = np.asarray(lengths, dtype=np.int64)
    report = {
        "dataset_sha256": digest.hexdigest(),
        "rows": len(lengths),
        "tokenizer": manifest["tokenizer"],
        "tokenizer_identity_sha256": manifest["tokenizer_identity_sha256"],
        "format": manifest["format"],
        "standard_deviation": "Population (ddof=0), conditional on truncation at each cap",
        "method": "Full scan of stored admission token counts, authenticated against dataset checksum and aggregate manifest statistics; cross-checked real sft_batch on each available task/cap/outcome stratum",
        "sft_batch_checks": len(examples),
        "results": [truncation_statistics(counts, cap) for cap in caps],
        "by_task": {
            task: [
                truncation_statistics(counts[counts[:, 2] == index], cap)
                for cap in caps
            ]
            for task, index in task_indices.items()
        },
        "row_lengths_columns": [
            "prompt_tokens",
            "answer_tokens_including_eos",
            "task_index",
        ],
        "task_index_names": tasks,
    }
    quality_report = {
        "dataset_sha256": digest.hexdigest(),
        "split": split,
        "rows": len(lengths),
        "method": "All checksum-verified rows; exact final-answer counts and stored trace statistics, without fresh generator sampling",
        "tasks": dict(quality),
    }
    return report, quality_report


def write_reports(output: Path, report: dict, quality: dict):
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in (("report.json", report), ("quality.json", quality)):
        (output / name).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    lines = [
        "# Reasoning Gym truncation audit",
        "",
        f"All {report['rows']:,} {quality['split']} rows, authenticated against the source manifest. Token counts include EOS; standard deviations are population values conditional on truncation. "
        f"Verified {report['sft_batch_checks']} task/cap/outcome strata with the pinned tokenizer and actual `sft_batch`.",
        "",
        "Reproduce the training audit: `python scripts/audit_reasoning_gym.py`. Use `--split validation` or `--split test` with a separate `--out-dir` for holdouts. This reads the prepared data without changing it.",
        "",
        "| Cap | Truncated rows | Original tokens, mean +/- std | Removed tokens, mean +/- std | Prompt-only rows | Answer-token mass lost |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        original = result["original_total_tokens_truncated_rows"]
        removed = result["removed_total_tokens_truncated_rows"]
        before = (
            f"{original['mean']:.1f} +/- {original['std']:.1f}"
            if original["count"]
            else "N/A"
        )
        after = (
            f"{removed['mean']:.1f} +/- {removed['std']:.1f}"
            if removed["count"]
            else "N/A"
        )
        lines.append(
            f"| {result['sequence_len']} | {result['truncated_rows']:,} ({result['truncated_percent']:.3f}%) | {before} | {after} | {result['prompt_only_rows']} | {result['lost_answer_token_percent']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "| Task | "
            + " | ".join(
                f"At {row['sequence_len']}" for row in report["results"]
            )
            + " |",
            "|---|" + "---:|" * len(report["results"]),
        ]
    )
    for task, results in report["by_task"].items():
        lines.append(
            f"| {task} | "
            + " | ".join(f"{row['truncated_percent']:.3f}%" for row in results)
            + " |"
        )
    lines.extend(
        [
            "",
            "[Detailed statistics](report.json) include retained targets, EOS-only truncations, and per-task results. [Distribution counts](quality.json) include matrix degeneracy, GCD labels, dismantling outputs, and the cube initial-colour baseline.",
            "",
            f"Source SHA-256: `{report['dataset_sha256']}`.",
            "",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("data/train-data/reasoning-gym")
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/analysis/reasoning-gym-truncation"),
    )
    parser.add_argument(
        "--sequence-lens",
        nargs="+",
        type=int,
        default=[512, 1024, 1536, 2048, 4096],
    )
    parser.add_argument(
        "--split", choices=["train", "validation", "test"], default="train"
    )
    args = parser.parse_args()
    report, quality = audit_dataset(args.source, args.sequence_lens, args.split)
    write_reports(args.out_dir, report, quality)
    print(
        f"Audited {report['rows']:,} rows and {report['sft_batch_checks']} SFT strata; wrote {args.out_dir}"
    )


if __name__ == "__main__":
    main()
