from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import torch

from gcw.distill.prompt import ChatPrompt, format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession, RolloutResult
from gcw.models.bgt import BGT


@dataclass(frozen=True)
class OpsdEvalCondition:
    enable_thinking: bool
    processor_enabled: bool
    process_every: int = 64
    select_all: bool = False
    reasoning_open_text: str = "<think>\n"
    max_response_tokens: int = 512

    @property
    def name(self) -> str:
        thinking = "cot-on" if self.enable_thinking else "cot-off"
        processor = "cp-on" if self.processor_enabled else "cp-off"
        return f"{thinking}__{processor}"


def build_opsd_eval_prompt(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
    reasoning_open_text: str = "<think>\n",
) -> ChatPrompt:
    if enable_thinking:
        return format_teacher_prompt(tokenizer, prompt, reasoning_open_text=reasoning_open_text)
    return format_chat_prompt(tokenizer, prompt, enable_thinking=False)


def greedy_opsd_response(
    model: BGT,
    tokenizer: Any,
    prompt: str,
    *,
    max_response_tokens: int,
    condition: OpsdEvalCondition,
    stop_sequences: tuple[tuple[int, ...], ...] = (),
) -> RolloutResult:
    chat_prompt = build_opsd_eval_prompt(
        tokenizer,
        prompt,
        enable_thinking=condition.enable_thinking,
        reasoning_open_text=condition.reasoning_open_text,
    )
    session = BGTDecodeSession(
        model,
        process_every=condition.process_every,
        select_all=condition.select_all,
        valid_vocab_size=_tokenizer_vocab_size(tokenizer),
        processor_enabled=condition.processor_enabled,
    )
    device = model.get_device()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with context:
        return session.rollout(
            chat_prompt,
            max_response_tokens=max_response_tokens,
            eos_token_id=int(tokenizer.eot_id),
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            stop_sequences=stop_sequences,
            greedy=True,
        )


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    inner = getattr(tokenizer, "tokenizer", tokenizer)
    try:
        size = len(inner)
    except TypeError:
        getter = getattr(tokenizer, "get_vocab_size", None)
        if getter is None:
            raise TypeError("tokenizer must expose __len__ or get_vocab_size") from None
        size = int(getter())
    if size <= 0:
        raise ValueError("tokenizer vocabulary must be positive")
    return int(size)


OPSD_SCREEN_CONDITIONS = (
    "cot-off__cp-off",
    "cot-off__cp-on",
    "cot-on__cp-off",
    "cot-on__cp-on",
)


def aggregate_opsd_screen(documents: list[dict[str, Any]]) -> dict[str, Any]:
    by_condition = {document["condition"]["name"]: document for document in documents}
    if set(by_condition) != set(OPSD_SCREEN_CONDITIONS) or len(documents) != len(by_condition):
        raise ValueError("OPSD screen requires exactly four unique canonical conditions")

    ordered = [by_condition[name] for name in OPSD_SCREEN_CONDITIONS]
    for name, document in zip(OPSD_SCREEN_CONDITIONS, ordered, strict=True):
        condition = document["condition"]
        if bool(condition["enable_thinking"]) != name.startswith("cot-on"):
            raise ValueError("OPSD screen condition name disagrees with thinking flag")
        if bool(condition["processor_enabled"]) != name.endswith("cp-on"):
            raise ValueError("OPSD screen condition name disagrees with processor flag")
    shared_keys = (
        "source_revision",
        "checkpoint_sha256",
        "checkpoint_manifest",
        "tokenizer_source",
        "tokenizer_identity_sha256",
        "datasets_cache",
        "tasks",
        "task_config_sha256",
        "dataset_files",
        "limit_per_task",
    )
    for key in shared_keys:
        if any(document[key] != ordered[0][key] for document in ordered[1:]):
            raise ValueError(f"OPSD screen conditions differ in {key}")

    sample_manifest = ordered[0]["sample_manifest"]
    if not sample_manifest or any(
        document["sample_manifest"] != sample_manifest for document in ordered[1:]
    ):
        raise ValueError("OPSD screen conditions do not share one sample/target manifest")
    prompt_hashes = [record["prompt_sha256"] for record in ordered[0]["generation_records"]]
    if not prompt_hashes or any(
        [record["prompt_sha256"] for record in document["generation_records"]] != prompt_hashes
        for document in ordered[1:]
    ):
        raise ValueError("OPSD screen conditions do not share one ordered prompt manifest")
    base_hashes = [
        [record["base_token_ids_sha256"] for record in document["generation_records"]]
        for document in ordered
    ]
    if any(hashes != base_hashes[0] for hashes in base_hashes[1:]):
        raise ValueError("OPSD screen conditions differ in user/system base token IDs")
    for left, right in ((ordered[0], ordered[1]), (ordered[2], ordered[3])):
        left_hashes = [record["full_prompt_token_ids_sha256"] for record in left["generation_records"]]
        right_hashes = [record["full_prompt_token_ids_sha256"] for record in right["generation_records"]]
        if left_hashes != right_hashes:
            raise ValueError("OPSD screen processor cells differ in rendered prompt token IDs")

    for document in ordered:
        enabled = bool(document["condition"]["processor_enabled"])
        bad_schedule = False
        for record in document["generation_records"]:
            expected = (
                [0, *range(64, int(record["raw_response_tokens"]), 64)] if enabled else []
            )
            expected_lengths = []
            if enabled:
                expected_lengths = [int(record["base_tokens"])] + [
                    int(record["base_tokens"])
                    + int(record["trigger_tokens"])
                    + position
                    for position in expected[1:]
                ]
            if (
                record["cache_event_positions"] != expected
                or record["cache_event_lengths"] != expected_lengths
            ):
                bad_schedule = True
                break
        if bad_schedule:
            raise ValueError("OPSD screen cache-event evidence differs from processor condition")

    metrics = {name: _condition_metrics(by_condition[name]) for name in OPSD_SCREEN_CONDITIONS}
    contrasts = {
        "cp_effect_cot_off": _metric_delta(metrics["cot-off__cp-on"], metrics["cot-off__cp-off"]),
        "cp_effect_cot_on": _metric_delta(metrics["cot-on__cp-on"], metrics["cot-on__cp-off"]),
        "cot_effect_cp_off": _metric_delta(metrics["cot-on__cp-off"], metrics["cot-off__cp-off"]),
        "cot_effect_cp_on": _metric_delta(metrics["cot-on__cp-on"], metrics["cot-off__cp-on"]),
    }
    contrasts["interaction"] = _metric_delta(
        contrasts["cp_effect_cot_on"], contrasts["cp_effect_cot_off"]
    )
    prompt_manifest_sha256 = hashlib.sha256(
        json.dumps(prompt_hashes, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    sample_manifest_sha256 = hashlib.sha256(
        json.dumps(sample_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "opsd_quick_screen_aggregate",
        **{key: ordered[0][key] for key in shared_keys},
        "conditions": list(OPSD_SCREEN_CONDITIONS),
        "prompt_count": len(prompt_hashes),
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "sample_manifest_sha256": sample_manifest_sha256,
        "metrics": metrics,
        "contrasts": contrasts,
    }


def _condition_metrics(document: dict[str, Any]) -> dict[str, Any]:
    task_results = document["lm_eval"]["results"]
    metrics: dict[str, Any] = {}
    for task in document["tasks"]:
        row = task_results[task]
        metrics[task] = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and key.startswith("exact_match,")
        }
    records = document["generation_records"]
    metrics["generation"] = {
        "mean_response_tokens": sum(record["response_tokens"] for record in records) / len(records),
        "truncated_fraction": sum(
            record["response_tokens"] >= record["max_response_tokens"] for record in records
        )
        / len(records),
    }
    return metrics


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in left.keys() & right.keys():
        if not isinstance(left[group], dict) or not isinstance(right[group], dict):
            continue
        common = left[group].keys() & right[group].keys()
        output[group] = {
            key: float(left[group][key]) - float(right[group][key]) for key in common
        }
    return output
