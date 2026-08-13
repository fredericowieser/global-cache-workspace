from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import os
import tempfile
from typing import Any, Iterable, Mapping

from datasets import load_dataset

from gcw.data.sft import iter_jsonl
from gcw.data.tokenizers import get_tokenizer


TRACE_RE = re.compile(r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*(?P<answer>.+?)\s*$", re.DOTALL)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_prompt(value: object) -> str:
    return str(value).strip()


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _jsonl_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_flag(value: object) -> bool | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return value if isinstance(value, bool) else None


def _load_exclusion_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    payload = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
    hashes: set[str] = set()
    if isinstance(payload, dict):
        for row in payload.get("sample_manifest", []):
            if isinstance(row, Mapping) and isinstance(row.get("prompt_hash"), str):
                hashes.add(row["prompt_hash"])
        for value in payload.get("prompt_hashes", []):
            if isinstance(value, str):
                hashes.add(value)
        return hashes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            hashes.add(line.strip())
            continue
        if isinstance(row, str):
            hashes.add(row)
        elif isinstance(row, Mapping):
            for key in ("prompt_hash", "normalized_prompt_text_sha256", "problem_sha256"):
                value = row.get(key)
                if isinstance(value, str):
                    hashes.add(value)
    return hashes


def _render_lengths(tokenizer: Any, prompt: str, answer: str, *, enable_thinking: bool) -> tuple[int, int, int]:
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    answer_ids = tokenizer.encode(answer) + [tokenizer.eot_id]
    return len(prompt_ids), len(answer_ids), len(prompt_ids) + len(answer_ids)


def _batch_encode(tokenizer: Any, texts: list[str], *, batch_size: int = 512) -> list[list[int]]:
    encoder = getattr(tokenizer, "enc", None)
    backend = getattr(encoder, "backend_tokenizer", None)
    encoded: list[list[int]] = []
    if backend is not None:
        for start in range(0, len(texts), batch_size):
            values = backend.encode_batch(texts[start : start + batch_size], add_special_tokens=False)
            encoded.extend([list(item.ids) for item in values])
        return encoded
    if callable(encoder):
        for start in range(0, len(texts), batch_size):
            result = encoder(
                texts[start : start + batch_size],
                add_special_tokens=False,
                return_attention_mask=False,
            )
            values = result["input_ids"] if isinstance(result, Mapping) else result
            encoded.extend([list(ids) for ids in values])
        return encoded
    return [tokenizer.encode(text) for text in texts]


def _openmath_records(
    path: Path,
    tokenizer: Any,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], set[str], dict[str, int]]:
    records: list[dict[str, Any]] = []
    hashes: set[str] = set()
    counts: Counter[str] = Counter()
    raw_records: list[tuple[int, Mapping[str, Any], str, str, str]] = []
    for index, row in enumerate(iter_jsonl(path)):
        if index >= limit:
            break
        prompt = canonical_prompt(row.get("prompt") or row.get("problem"))
        answer = str(row.get("answer") or row.get("generated_solution") or "").strip()
        if not prompt or not answer:
            counts["missing_prompt_or_answer"] += 1
            continue
        prompt_hash = sha256_text(prompt)
        hashes.add(prompt_hash)
        raw_records.append((index, row, prompt, answer, prompt_hash))
    answer_token_ids = _batch_encode(tokenizer, [item[3] for item in raw_records])
    for (index, row, prompt, answer, prompt_hash), answer_ids in zip(raw_records, answer_token_ids):
        answer_tokens = len(answer_ids) + 1
        record = dict(row)
        record.update(
            {
                "prompt": prompt,
                "answer": answer,
                "enable_thinking": False,
                "source_arm": "openmath",
                "thinking_mode": "thinking_off",
                "source_id": str(row.get("problem_sha256") or prompt_hash),
                "source_prompt_hash": prompt_hash,
                "prompt_tokens": None,
                "target_tokens": answer_tokens,
                "sequence_tokens": None,
                "source_index": index,
                "admission": "existing_clean_openmath_slice",
            }
        )
        counts["admitted"] += 1
        records.append(record)
    return records, hashes, dict(counts)


def _trace_record(row: Mapping[str, Any]) -> tuple[str, str, str] | tuple[None, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None, "missing_messages", ""
    user = messages[-2] if isinstance(messages[-2], Mapping) else None
    assistant = messages[-1] if isinstance(messages[-1], Mapping) else None
    if not user or user.get("role") != "user" or not assistant or assistant.get("role") != "assistant":
        return None, "invalid_message_roles", ""
    content = str(assistant.get("content") or "")
    match = TRACE_RE.match(content)
    if match is None:
        return None, "invalid_reasoning_delimiters_or_empty_answer", content
    if not match.group("reasoning").strip():
        return None, "empty_reasoning", content
    if not match.group("answer").strip():
        return None, "empty_visible_answer", content
    if _first_flag(row.get("is_reasoning_complete")) is not True:
        return None, "source_reasoning_incomplete", content
    if _first_flag(row.get("correctness_math_verify")) is not True:
        return None, "source_math_verify_false_or_missing", content
    return content, "source_is_reasoning_complete_and_math_verified", content


def _iter_openr1(
    *,
    repo: str,
    config: str,
    revision: str,
) -> Iterable[Mapping[str, Any]]:
    dataset = load_dataset(repo, config, split="train", streaming=True, revision=revision)
    yield from dataset


def _interleave_by_token_mass(
    direct: list[dict[str, Any]], cot: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Greedily order both arms so each prefix stays near token-balanced."""
    output: list[dict[str, Any]] = []
    direct_index = cot_index = 0
    direct_mass = cot_mass = 0
    while direct_index < len(direct) or cot_index < len(cot):
        if direct_index >= len(direct):
            choose_direct = False
        elif cot_index >= len(cot):
            choose_direct = True
        else:
            choose_direct = direct_mass <= cot_mass
        row = direct[direct_index] if choose_direct else cot[cot_index]
        if choose_direct:
            direct_index += 1
            direct_mass += int(row["target_tokens"])
        else:
            cot_index += 1
            cot_mass += int(row["target_tokens"])
        row = dict(row)
        row["mixture_order"] = len(output)
        row["prefix_direct_target_tokens"] = direct_mass
        row["prefix_cot_target_tokens"] = cot_mass
        output.append(row)
    return output


def _admission_policy(cot_target_mass: int, direct_target_mass: int, *, allow_underfilled: bool) -> str:
    if cot_target_mass >= direct_target_mass:
        return "balanced"
    if allow_underfilled and cot_target_mass > 0:
        return "underfilled_openr1_cap"
    raise RuntimeError(
        f"OpenR1 admitted token mass {cot_target_mass} below direct target {direct_target_mass}; refusing launch"
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(f"hf:{Path(args.tokenizer).expanduser()}")

    direct, direct_hashes, direct_counts = _openmath_records(
        Path(args.openmath), tokenizer, limit=args.openmath_rows
    )
    excluded = direct_hashes | _load_exclusion_hashes(
        Path(args.exclude_prompt_hashes) if args.exclude_prompt_hashes else None
    )
    if len(direct) != args.openmath_rows:
        raise RuntimeError(f"OpenMath admitted {len(direct)} rows, expected exact slice {args.openmath_rows}")

    direct_target_mass = sum(int(row["target_tokens"]) for row in direct)
    cot_path = out_dir / "openr1_admitted.jsonl"
    cot: list[dict[str, Any]] = []
    cot_hashes: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    scanned = 0
    cot_target_mass = 0
    for row in _iter_openr1(repo=args.openr1_repo, config=args.openr1_config, revision=args.openr1_revision):
        scanned += 1
        prompt = canonical_prompt(row.get("problem"))
        prompt_hash = sha256_text(prompt) if prompt else ""
        if not prompt:
            rejection_counts["missing_problem"] += 1
            continue
        if prompt_hash in excluded or prompt_hash in cot_hashes:
            rejection_counts["duplicate_or_excluded_prompt"] += 1
            continue
        answer, reason, raw_assistant = _trace_record(row)
        if answer is None:
            rejection_counts[reason] += 1
            continue
        prompt_tokens, target_tokens, sequence_tokens = _render_lengths(
            tokenizer, prompt, answer, enable_thinking=True
        )
        if sequence_tokens > args.max_length:
            rejection_counts["over_context_cap"] += 1
            continue
        record = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "prompt": prompt,
            "answer": answer,
            "enable_thinking": True,
            "source_arm": "openr1",
            "thinking_mode": "thinking_on",
            "source_id": str(row.get("uuid") or prompt_hash),
            "source_prompt_hash": prompt_hash,
            "source": row.get("source"),
            "problem_type": row.get("problem_type"),
            "question_type": row.get("question_type"),
            "source_is_reasoning_complete": row.get("is_reasoning_complete"),
            "source_correctness_math_verify": row.get("correctness_math_verify"),
            "source_correctness_llama": row.get("correctness_llama"),
            "source_finish_reasons": row.get("finish_reasons"),
            "terminal_reason": reason,
            "raw_assistant_sha256": sha256_text(raw_assistant),
            "prompt_tokens": prompt_tokens,
            "target_tokens": target_tokens,
            "sequence_tokens": sequence_tokens,
            "source_index": scanned - 1,
            "admission": "complete_math_verified_first_generation",
        }
        cot.append(record)
        cot_hashes.add(prompt_hash)
        cot_target_mass += target_tokens
        if cot_target_mass >= direct_target_mass:
            break
    mixture_policy = _admission_policy(
        cot_target_mass,
        direct_target_mass,
        allow_underfilled=args.allow_underfilled_openr1,
    )

    combined = _interleave_by_token_mass(direct, cot)
    combined_path = out_dir / "combined_train.jsonl"
    training_path = out_dir / f"training_{args.training_rows_for_audit}.jsonl"
    direct_path = out_dir / "openmath_slice.jsonl"
    _atomic_replace(
        direct_path,
        ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in direct)).encode("utf-8"),
    )
    _atomic_replace(
        cot_path,
        ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cot)).encode("utf-8"),
    )
    _atomic_replace(
        combined_path,
        ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in combined)).encode("utf-8"),
    )

    if len(combined) < args.training_rows_for_audit:
        raise RuntimeError(
            f"combined mixture has {len(combined)} rows, below required training prefix "
            f"{args.training_rows_for_audit}"
        )
    train_prefix_rows = args.training_rows_for_audit
    prefix = combined[:train_prefix_rows]
    _atomic_replace(
        training_path,
        ("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prefix)).encode("utf-8"),
    )
    prefix_counts = Counter(row["source_arm"] for row in prefix)
    prefix_mass = Counter()
    for row in prefix:
        prefix_mass[row["source_arm"]] += int(row["target_tokens"])
    manifest = {
        "schema": "gcw.openr1_bgt_mixture.v1",
        "openr1_repo": args.openr1_repo,
        "openr1_config": args.openr1_config,
        "openr1_revision": args.openr1_revision,
        "tokenizer": str(Path(args.tokenizer).expanduser().resolve()),
        "max_length": args.max_length,
        "mixture_policy": mixture_policy,
        "balance_target_met": mixture_policy == "balanced",
        "objective": "response_only_nll",
        "direct_source": str(Path(args.openmath).expanduser().resolve()),
        "direct_source_sha256": _jsonl_digest(Path(args.openmath)),
        "direct_slice_rows": len(direct),
        "direct_slice_counts": direct_counts,
        "direct_target_token_mass": direct_target_mass,
        "openr1_scanned_rows": scanned,
        "openr1_rejection_counts": dict(rejection_counts),
        "openr1_admitted_rows": len(cot),
        "openr1_target_token_mass": cot_target_mass,
        "openr1_admitted_sha256": _jsonl_digest(cot_path),
        "combined_rows": len(combined),
        "combined_sha256": _jsonl_digest(combined_path),
        "training_path": str(training_path),
        "training_rows": train_prefix_rows,
        "training_sha256": _jsonl_digest(training_path),
        "training_rows_for_audit": train_prefix_rows,
        "training_prefix_source_counts": dict(prefix_counts),
        "training_prefix_target_token_mass": dict(prefix_mass),
        "excluded_prompt_hash_count": len(excluded),
        "source_id_policy": "uuid_or_sha256_prompt",
        "external_cot_not_native_qwen": True,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_replace(out_dir / "manifest.json", manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openmath", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--openmath-rows", type=int, default=64_730)
    parser.add_argument("--training-rows-for-audit", type=int, default=64_730)
    parser.add_argument("--allow-underfilled-openr1", action="store_true")
    parser.add_argument("--max-length", type=int, default=1_024)
    parser.add_argument("--exclude-prompt-hashes", default=None)
    parser.add_argument("--openr1-repo", default="open-r1/OpenR1-Math-220k")
    parser.add_argument("--openr1-config", default="all")
    parser.add_argument("--openr1-revision", default="e4e141ec9dea9f8326f4d347be56105859b2bd68")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
