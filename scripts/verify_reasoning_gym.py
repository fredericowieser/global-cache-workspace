import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from gcw.data.reasoning_gym import (
    _generate_batch,
    VENDOR,
    VENDOR_SOURCES_SHA256,
)
from gcw.data.reasoning_gym_traces import extract_final_answer
from gcw.data.sft import sft_batch
from gcw.data.tokenizers import get_tokenizer


def verify_dataset(root: Path):
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["source_manifest_sha256"] != VENDOR_SOURCES_SHA256
        or hashlib.sha256((VENDOR / "SOURCES.json").read_bytes()).hexdigest()
        != VENDOR_SOURCES_SHA256
    ):
        raise ValueError(
            "Prepared source manifest differs from the pinned implementation"
        )
    tokenizer = get_tokenizer("hf:" + manifest["tokenizer"])
    if not (
        tokenizer.identity_sha256() == manifest["tokenizer_identity_sha256"]
    ):
        raise ValueError(
            "Verification failed: tokenizer.identity_sha256() == manifest['tokenizer_identity_sha256']"
        )
    prompts, problems, seeds = set(), set(), set()
    examples = {}
    report = {
        "schema": manifest["schema"],
        "source_revision": manifest["source"]["revision"],
        "tokenizer_repository": "Qwen/Qwen3.5-2B",
        "tokenizer_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "tokenizer_identity_sha256": manifest["tokenizer_identity_sha256"],
        "tokenizer_file_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(manifest["tokenizer"]).glob("*")
            if p.is_file() and p.suffix in (".json", ".jinja")
        },
        "preparation_command": "python scripts/prepare_reasoning_gym.py --tokenizer artifacts/tokenizers/Qwen3.5-2B --workers 4 --replace",
        "manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "implementation_sha256": manifest["implementation_sha256"],
        "files": {},
        "task_statistics": manifest["tasks"],
        "audit": {
            "all_file_checksums": True,
            "all_answer_labels": True,
            "globally_unique_prompts_inputs_and_effective_seeds": True,
            "reconstructed_and_training_batch_checked_per_task_per_split": 100,
        },
    }
    report["admission_policy"] = manifest["admission_policy"]
    report["source_manifest_sha256"] = manifest["source_manifest_sha256"]
    for name, expected in manifest["implementation_sha256"].items():
        if not (
            hashlib.sha256(
                (Path("src/gcw/data") / name).read_bytes()
            ).hexdigest()
            == expected
        ):
            raise ValueError(
                "Verification failed: hashlib.sha256((Path('src/gcw/data') / name).read_bytes()).hexdigest() == expected"
            )
    for split in ("train", "validation", "test"):
        counts, checked = Counter(), Counter()
        lengths = Counter()
        target_tokens = Counter()
        digest, ids = hashlib.sha256(), hashlib.sha256()
        with (root / (split + ".jsonl")).open("rb") as handle:
            for index, line in enumerate(handle):
                digest.update(line)
                row = json.loads(line)
                task = row["task"]
                if split == "train":
                    examples.setdefault(task, row)
                ids.update(
                    ("\n" if index else "").encode() + row["source_id"].encode()
                )
                if not (row["experiment_order"] == index):
                    raise ValueError(
                        "Verification failed: row['experiment_order'] == index"
                    )
                if not (row["split"] == split):
                    raise ValueError(
                        "Verification failed: row['split'] == split"
                    )
                if not (
                    row["prompt_sha256"]
                    == hashlib.sha256(
                        " ".join(row["prompt"].split()).encode()
                    ).hexdigest()
                ):
                    raise ValueError(
                        "Verification failed: row['prompt_sha256'] == hashlib.sha256(' '.join(row['prompt'].split()).encode()).hexdigest()"
                    )
                if not (row["prompt_sha256"] not in prompts):
                    raise ValueError(
                        "Verification failed: row['prompt_sha256'] not in prompts"
                    )
                if not (row["problem_sha256"] not in problems):
                    raise ValueError(
                        "Verification failed: row['problem_sha256'] not in problems"
                    )
                if not (row["effective_seed"] not in seeds):
                    raise ValueError(
                        "Verification failed: row['effective_seed'] not in seeds"
                    )
                if not (
                    row["effective_seed"]
                    == row["generator_seed"] + row["source_index"]
                ):
                    raise ValueError(
                        "Verification failed: row['effective_seed'] == row['generator_seed'] + row['source_index']"
                    )
                prompts.add(row["prompt_sha256"])
                problems.add(row["problem_sha256"])
                seeds.add(row["effective_seed"])
                if not (
                    extract_final_answer(row["answer"]) == row["final_answer"]
                ):
                    raise ValueError(
                        "Verification failed: extract_final_answer(row['answer']) == row['final_answer']"
                    )
                if not (row["trace_steps"] >= manifest["min_trace_steps"]):
                    raise ValueError(
                        "Verification failed: row['trace_steps'] >= manifest['min_trace_steps']"
                    )
                if not (row["sequence_tokens"] <= manifest["sequence_length"]):
                    raise ValueError(
                        "Verification failed: row['sequence_tokens'] <= manifest['sequence_length']"
                    )
                if not (
                    row["sequence_tokens"]
                    == row["prompt_tokens"] + row["target_tokens"]
                ):
                    raise ValueError(
                        "Verification failed: row['sequence_tokens'] == row['prompt_tokens'] + row['target_tokens']"
                    )
                counts[task] += 1
                lengths[row["sequence_tokens"]] += 1
                target_tokens[task] += row["target_tokens"]
                if checked[task] < 100:
                    generated, rejected = _generate_batch(
                        (
                            task,
                            split,
                            row["generator_seed"],
                            row["source_index"],
                            1,
                            manifest["min_trace_steps"],
                        )
                    )
                    if not (not rejected and len(generated) == 1):
                        raise ValueError(
                            "Verification failed: not rejected and len(generated) == 1"
                        )
                    for key, value in generated[0].items():
                        if not (row[key] == value):
                            raise ValueError((task, key))
                    x, y = sft_batch(
                        [row],
                        tokenizer,
                        max_length=manifest["sequence_length"],
                        chat_template=True,
                    )
                    if not (x.shape[1] == row["sequence_tokens"]):
                        raise ValueError(
                            "Verification failed: x.shape[1] == row['sequence_tokens']"
                        )
                    if not ((y != -100).sum().item() == row["target_tokens"]):
                        raise ValueError(
                            "Verification failed: (y != -100).sum().item() == row['target_tokens']"
                        )
                    if not (x[0, -1].item() == tokenizer.eot_id):
                        raise ValueError(
                            "Verification failed: x[0, -1].item() == tokenizer.eot_id"
                        )
                    checked[task] += 1
                if (index + 1) % 100000 == 0:
                    print(f"{split}: audited {index + 1:,}", flush=True)
        expected = manifest["files"][split]
        if not (sum(counts.values()) == expected["rows"]):
            raise ValueError(
                "Verification failed: sum(counts.values()) == expected['rows']"
            )
        if not (digest.hexdigest() == expected["sha256"]):
            raise ValueError(
                "Verification failed: digest.hexdigest() == expected['sha256']"
            )
        if not (ids.hexdigest() == expected["source_ids_sha256"]):
            raise ValueError(
                "Verification failed: ids.hexdigest() == expected['source_ids_sha256']"
            )
        for task, count in counts.items():
            if not (count == manifest["tasks"][task][split]["retained"]):
                raise ValueError(
                    "Verification failed: count == manifest['tasks'][task][split]['retained']"
                )
            if not (
                target_tokens[task]
                == manifest["tasks"][task][split]["target_tokens"]["total"]
            ):
                raise ValueError(
                    "Verification failed: target_tokens[task] == manifest['tasks'][task][split]['target_tokens']['total']"
                )
            if not (checked[task] == 100):
                raise ValueError("Verification failed: checked[task] == 100")
        report["files"][split] = {
            **expected,
            "tasks": dict(counts),
            "supervised_tokens": sum(target_tokens.values()),
            "max_sequence_tokens": max(lengths),
            "mean_sequence_tokens": sum(k * v for k, v in lengths.items())
            / sum(lengths.values()),
        }
        print(f"{split}: verified {dict(counts)}", flush=True)
    return report, [examples[task] for task in sorted(examples)]


def main():
    parser = argparse.ArgumentParser(
        description="Verify prepared rows, regenerate trace samples, and write the preparation receipt"
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/train-data/reasoning-gym")
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path("data/train-data/reasoning-gym-preparation.json"),
    )
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train-data/reasoning-gym-examples.jsonl"),
    )
    args = parser.parse_args()
    report, examples = verify_dataset(args.source)
    args.receipt.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.examples.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in examples
        ),
        encoding="utf-8",
        newline="\n",
    )
    print("Preparation verification passed")


if __name__ == "__main__":
    main()
