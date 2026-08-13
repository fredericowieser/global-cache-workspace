from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable, Mapping

from datasets import load_dataset

from gcw.data.sft import iter_jsonl
from gcw.data.tokenizers import get_tokenizer


TRACE_RE = re.compile(r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*(?P<answer>.+?)\s*$", re.DOTALL)
DEFAULT_CAPS = (1024, 2048, 4096, 8192)
BENCHMARK_DATASETS = (
    ("openai/gsm8k", "main", "test"),
    ("reasoning-machines/gsm-hard", "default", "train"),
    ("ChilleD/SVAMP", "default", "test"),
    ("EleutherAI/logiqa", "default", "test"),
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_prompt(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def _prompt_hash(value: object) -> str:
    return _sha256_text(_normalize_prompt(value))


def _first_bool(value: object) -> bool | None:
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, bool) else None


def _answer_tokens(tokenizer: Any, answer: str) -> list[int]:
    return tokenizer.encode(answer) + [tokenizer.eot_id]


def _render_length(tokenizer: Any, prompt: str, answer: str, *, thinking: bool) -> tuple[int, int, int]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    answer_ids = _answer_tokens(tokenizer, answer)
    return len(prompt_ids), len(answer_ids), len(prompt_ids) + len(answer_ids)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
            handle.write(line)
            digest.update(line)
    temporary.replace(path)
    return digest.hexdigest()


def _load_excluded_hashes(paths: list[str]) -> set[str]:
    excluded: set[str] = set()
    for value in paths:
        path = Path(value)
        for row in iter_jsonl(path):
            prompt = row.get("prompt") or row.get("question") or row.get("problem")
            if prompt:
                excluded.add(_prompt_hash(prompt))
    return excluded


def _load_benchmark_hashes() -> tuple[set[str], dict[str, int]]:
    excluded: set[str] = set()
    counts: dict[str, int] = {}
    for repo, config, split in BENCHMARK_DATASETS:
        try:
            dataset = load_dataset(repo, config, split=split)
        except (FileNotFoundError, ValueError):
            dataset = load_dataset(repo, split=split)
        count = 0
        for row in dataset:
            prompt = row.get("question") or row.get("Body") or row.get("prompt") or row.get("problem")
            if prompt:
                excluded.add(_prompt_hash(prompt))
                count += 1
        counts[f"{repo}:{config}:{split}"] = count
    return excluded, counts


def _openmath_rows(path: Path) -> Iterable[tuple[str, str, str]]:
    for index, row in enumerate(iter_jsonl(path)):
        prompt = str(row.get("prompt") or row.get("problem") or "").strip()
        answer = str(row.get("answer") or row.get("generated_solution") or "").strip()
        if prompt and answer:
            yield prompt, answer, str(row.get("problem_sha256") or f"openmath:{index}")


def _verified_openr1_rows(repo: str, config: str, revision: str) -> Iterable[tuple[str, str, str]]:
    dataset = load_dataset(repo, config, split="train", streaming=True, revision=revision)
    for row in dataset:
        prompt = str(row.get("problem") or "").strip()
        messages = row.get("messages")
        assistant = messages[-1] if isinstance(messages, list) and messages else None
        answer = str(assistant.get("content") or "") if isinstance(assistant, Mapping) else ""
        match = TRACE_RE.match(answer)
        if (
            prompt
            and match is not None
            and match.group("reasoning").strip()
            and match.group("answer").strip()
            and _first_bool(row.get("is_reasoning_complete")) is True
            and _first_bool(row.get("correctness_math_verify")) is True
        ):
            yield prompt, answer, str(row.get("uuid") or _prompt_hash(prompt))


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out_dir).expanduser()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    tokenizer = get_tokenizer(f"hf:{args.tokenizer}")
    benchmark_hashes, benchmark_counts = _load_benchmark_hashes()
    excluded = benchmark_hashes | _load_excluded_hashes(args.exclude_jsonl)
    caps = tuple(sorted(set(args.caps)))
    if args.sequence_length not in caps:
        caps = tuple(sorted((*caps, args.sequence_length)))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejection: Counter[str] = Counter()
    audit: dict[int, Counter[str]] = {cap: Counter() for cap in caps}
    sources = (
        ("openmath", False, _openmath_rows(Path(args.openmath))),
        (
            "openr1",
            True,
            _verified_openr1_rows(args.openr1_repo, args.openr1_config, args.openr1_revision),
        ),
    )
    for source, thinking, source_rows in sources:
        for prompt, answer, source_id in source_rows:
            prompt_hash = _prompt_hash(prompt)
            if prompt_hash in excluded:
                rejection[f"{source}:benchmark_overlap"] += 1
                continue
            if prompt_hash in seen:
                rejection[f"{source}:duplicate"] += 1
                continue
            prompt_tokens, target_tokens, total_tokens = _render_length(
                tokenizer, prompt, answer, thinking=thinking
            )
            for cap in caps:
                audit[cap][f"{source}:seen_rows"] += 1
                audit[cap][f"{source}:target_tokens"] += target_tokens
                if total_tokens <= cap:
                    audit[cap][f"{source}:admitted_rows"] += 1
                    audit[cap][f"{source}:admitted_target_tokens"] += target_tokens
            if total_tokens > args.sequence_length:
                rejection[f"{source}:over_context"] += 1
                continue
            seen.add(prompt_hash)
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ],
                    "prompt": prompt,
                    "answer": answer,
                    "enable_thinking": thinking,
                    "source_arm": source,
                    "source_id": source_id,
                    "source_prompt_hash": prompt_hash,
                    "prompt_tokens": prompt_tokens,
                    "target_tokens": target_tokens,
                    "sequence_tokens": total_tokens,
                }
            )

    random.Random(args.seed).shuffle(rows)
    for index, row in enumerate(rows):
        row["campaign_order"] = index
    dataset_path = output / "train.jsonl"
    dataset_sha256 = _atomic_jsonl(dataset_path, rows)
    source_ids_sha256 = hashlib.sha256(
        "\n".join(str(row["source_id"]) for row in rows).encode()
    ).hexdigest()
    source_counts = Counter(str(row["source_arm"]) for row in rows)
    manifest = {
        "schema": "gcw.math_campaign_data.v1",
        "sequence_length": args.sequence_length,
        "seed": args.seed,
        "rows": len(rows),
        "source_counts": dict(source_counts),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "source_ids_sha256": source_ids_sha256,
        "tokenizer": args.tokenizer,
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "openmath": str(Path(args.openmath).resolve()),
        "openr1": {
            "repository": args.openr1_repo,
            "config": args.openr1_config,
            "revision": args.openr1_revision,
        },
        "benchmark_exclusion_counts": benchmark_counts,
        "excluded_prompt_hashes": len(excluded),
        "rejection_counts": dict(rejection),
        "retention": {str(cap): dict(values) for cap, values in audit.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the authenticated Qwen math campaign dataset")
    parser.add_argument("--openmath", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--caps", type=int, nargs="+", default=DEFAULT_CAPS)
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--exclude-jsonl", action="append", default=[])
    parser.add_argument("--openr1-repo", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--openr1-config", default="all")
    parser.add_argument("--openr1-revision", default="e4e141ec9dea9f8326f4d347be56105859b2bd68")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
