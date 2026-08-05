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


def aggregate_opsd_three_way(
    documents: dict[str, dict[str, Any]],
    *,
    bootstrap_seed: int = 240805,
    bootstrap_samples: int = 10_000,
    expected_tasks: tuple[str, ...] = (
        "gsm8k_openmath",
        "svamp",
        "gsm8k_hard",
    ),
    expected_limit: int = 64,
    expected_processor_seed: int = 24,
) -> dict[str, Any]:
    """Aggregate paired CoT-off CP-off/untrained/trained OPSD controls."""
    import numpy as np

    labels = ("off", "untrained", "trained")
    if set(documents) != set(labels):
        raise ValueError("OPSD three-way aggregate requires off, untrained, and trained")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    ordered = [documents[label] for label in labels]
    shared_keys = (
        "source_revision",
        "tokenizer_source",
        "tokenizer_identity_sha256",
        "datasets_cache",
        "tasks",
        "task_config_sha256",
        "dataset_files",
        "limit_per_task",
        "model_config",
    )
    for key in shared_keys:
        if any(document[key] != ordered[0][key] for document in ordered[1:]):
            raise ValueError(f"OPSD three-way controls differ in {key}")
    if tuple(ordered[0]["tasks"]) != expected_tasks:
        raise ValueError("OPSD three-way tasks differ from the canonical protocol")
    if int(ordered[0]["limit_per_task"]) != expected_limit:
        raise ValueError("OPSD three-way limit differs from the canonical protocol")
    expected_generation_policy = {
        "do_sample": False,
        "temperature": 0.0,
        "top_k": 1,
        "fewshot": 0,
        "apply_chat_template_in_lm_eval": False,
    }
    if any(
        document.get("generation_policy") != expected_generation_policy
        for document in ordered
    ):
        raise ValueError("OPSD three-way generation policy is not canonical greedy decoding")

    expected_conditions = {
        "off": ("cot-off__cp-off", False),
        "untrained": ("cot-off__cp-on", True),
        "trained": ("cot-off__cp-on", True),
    }
    for label, document in documents.items():
        name, processor_enabled = expected_conditions[label]
        condition = document["condition"]
        if (
            condition["name"] != name
            or bool(condition["enable_thinking"])
            or bool(condition["processor_enabled"]) != processor_enabled
        ):
            raise ValueError(f"OPSD three-way {label} condition is not canonical CoT-off")

    off_manifest = documents["off"]["checkpoint_manifest"]
    trained_manifest = documents["trained"]["checkpoint_manifest"]
    untrained_manifest = documents["untrained"]["checkpoint_manifest"]
    if off_manifest != trained_manifest:
        raise ValueError("off and trained controls must use the same trained checkpoint")
    config_hashes = {
        document["checkpoint_manifest"]["files"]["config.json"]["sha256"]
        for document in ordered
    }
    if len(config_hashes) != 1:
        raise ValueError("three-way checkpoint model configs differ")
    if documents["untrained"]["checkpoint_sha256"] == documents["trained"]["checkpoint_sha256"]:
        raise ValueError("untrained and trained checkpoint weights are identical")
    provenance = documents["untrained"].get("train_state", {})
    if (
        provenance.get("kind") != "derived_untrained_fp32_cache_processor_control"
        or provenance.get("processor_storage_dtype") != "float32"
        or provenance.get("processor_initialization_seed") != expected_processor_seed
        or documents["trained"].get("train_state", {}).get("training_seed")
        != expected_processor_seed
        or provenance.get("dtype") != "mixed_bfloat16_backbone_float32_processor"
        or provenance.get("source_revision") != documents["trained"]["source_revision"]
        or provenance.get("source_model_sha256")
        != documents["trained"]["checkpoint_sha256"]
        or provenance.get("tokenizer_identity_sha256")
        != documents["trained"]["tokenizer_identity_sha256"]
        or provenance.get("trained_processor_sha256")
        != documents["trained"].get("train_state", {}).get(
            "post_load_processor_sha256"
        )
        or not provenance.get("initial_processor_sha256")
        or provenance.get("source_backbone_sha256")
        != provenance.get("control_backbone_sha256")
        or set(provenance.get("checkpoint_dtype_counts", {}))
        != {"torch.bfloat16", "torch.float32"}
    ):
        raise ValueError("three-way comparison lacks valid FP32 untrained processor provenance")
    if untrained_manifest["files"]["train_state.json"]["sha256"] == trained_manifest[
        "files"
    ]["train_state.json"]["sha256"]:
        raise ValueError("untrained control metadata is identical to trained metadata")

    sample_manifest = documents["off"]["sample_manifest"]
    if not sample_manifest or any(
        document["sample_manifest"] != sample_manifest for document in ordered[1:]
    ):
        raise ValueError("three-way controls do not share one sample/target manifest")
    record_keys = (
        "prompt_sha256",
        "prompt_text_sha256",
        "normalized_prompt_text_sha256",
        "base_token_ids_sha256",
        "full_prompt_token_ids_sha256",
    )
    for key in record_keys:
        reference = [record[key] for record in documents["off"]["generation_records"]]
        if not reference or any(
            [record[key] for record in document["generation_records"]] != reference
            for document in ordered[1:]
        ):
            raise ValueError(f"three-way controls differ in {key}")
    for label, document in documents.items():
        enabled = label != "off"
        for record in document["generation_records"]:
            expected_positions = (
                [0, *range(64, int(record["raw_response_tokens"]), 64)] if enabled else []
            )
            expected_lengths = (
                [int(record["base_tokens"])]
                + [
                    int(record["base_tokens"])
                    + int(record["trigger_tokens"])
                    + position
                    for position in expected_positions[1:]
                ]
                if enabled
                else []
            )
            if (
                record["cache_event_positions"] != expected_positions
                or record["cache_event_lengths"] != expected_lengths
            ):
                raise ValueError(f"three-way {label} cache-event evidence is invalid")

    tasks = tuple(documents["off"]["tasks"])
    limit = int(documents["off"]["limit_per_task"])

    def sample_rows(document: dict[str, Any], task: str) -> dict[int, dict[str, Any]]:
        samples = document.get("lm_eval", {}).get("samples", {}).get(task)
        if not isinstance(samples, list):
            raise ValueError(f"three-way sample logs are missing task {task}")
        filtered = [row for row in samples if row.get("filter") == "flexible"]
        result = {int(row["doc_id"]): row for row in filtered}
        if len(result) != limit or len(filtered) != len(result):
            raise ValueError(f"expected {limit} unique flexible rows for {task}")
        return result

    by_task = {
        label: {task: sample_rows(document, task) for task in tasks}
        for label, document in documents.items()
    }
    for task in tasks:
        reference_ids = list(by_task["off"][task])
        if any(list(by_task[label][task]) != reference_ids for label in labels[1:]):
            raise ValueError(f"three-way paired document IDs differ for {task}")

    values_by_task: dict[str, dict[str, Any]] = {label: {} for label in labels}
    accuracy: dict[str, dict[str, float]] = {label: {} for label in labels}
    for label, document in documents.items():
        for task in tasks:
            ids = list(by_task[label][task])
            values = np.asarray(
                [float(by_task[label][task][doc_id]["exact_match"]) for doc_id in ids]
            )
            if not np.all((values == 0) | (values == 1)):
                raise ValueError(f"three-way exact-match samples are not binary for {label}/{task}")
            aggregate_score = document["lm_eval"]["results"][task].get(
                "exact_match,flexible"
            )
            if aggregate_score is None or not math.isclose(
                float(values.mean()), float(aggregate_score), abs_tol=1e-12
            ):
                raise ValueError(
                    f"three-way sample accuracy disagrees with lm-eval for {label}/{task}"
                )
            values_by_task[label][task] = values
            accuracy[label][task] = float(values.mean())

    contrasts = {
        "untrained_minus_off": ("off", "untrained"),
        "trained_minus_off": ("off", "trained"),
        "trained_minus_untrained": ("untrained", "trained"),
    }
    rng = np.random.default_rng(bootstrap_seed)
    draws_by_task = {
        task: rng.integers(0, limit, size=(bootstrap_samples, limit)) for task in tasks
    }
    draws_digest = hashlib.sha256()
    for task in tasks:
        draws_digest.update(task.encode("utf-8"))
        draws_digest.update(draws_by_task[task].astype("<i8", copy=False).tobytes())

    def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
        discordant = wrong_to_right + right_to_wrong
        if discordant == 0:
            return 1.0
        tail = sum(
            math.comb(discordant, value)
            for value in range(min(wrong_to_right, right_to_wrong) + 1)
        ) / (2**discordant)
        return min(1.0, 2.0 * tail)

    paired: dict[str, Any] = {}
    for contrast, (left, right) in contrasts.items():
        task_results = {}
        task_bootstraps = []
        for task in tasks:
            left_values = values_by_task[left][task]
            right_values = values_by_task[right][task]
            difference = right_values - left_values
            bootstrap = difference[draws_by_task[task]].mean(axis=1)
            task_bootstraps.append(bootstrap)
            wrong_to_right = int(np.sum((left_values == 0) & (right_values == 1)))
            right_to_wrong = int(np.sum((left_values == 1) & (right_values == 0)))
            task_results[task] = {
                "delta": float(difference.mean()),
                "bootstrap_95_ci": [
                    float(value)
                    for value in np.quantile(
                        bootstrap, [0.025, 0.975], method="linear"
                    )
                ],
                "wrong_to_right": wrong_to_right,
                "right_to_wrong": right_to_wrong,
                "both_correct": int(np.sum((left_values == 1) & (right_values == 1))),
                "both_wrong": int(np.sum((left_values == 0) & (right_values == 0))),
                "exact_mcnemar_p": exact_mcnemar_p(wrong_to_right, right_to_wrong),
            }
        macro_bootstrap = np.stack(task_bootstraps, axis=1).mean(axis=1)
        paired[contrast] = {
            "tasks": task_results,
            "three_task_macro_is_descriptive": True,
            "three_task_macro_delta": float(
                np.mean([task_results[task]["delta"] for task in tasks])
            ),
            "three_task_macro_bootstrap_95_ci": [
                float(value)
                for value in np.quantile(
                    macro_bootstrap, [0.025, 0.975], method="linear"
                )
            ],
        }

    def response(row: dict[str, Any]) -> str:
        value = row["resps"][0]
        return str(value[0] if isinstance(value, list) else value)

    response_identity = {}
    for left, right in (("off", "untrained"), ("untrained", "trained"), ("off", "trained")):
        response_identity[f"{left}_equals_{right}"] = {
            task: sum(
                response(by_task[left][task][doc_id])
                == response(by_task[right][task][doc_id])
                for doc_id in by_task[left][task]
            )
            for task in tasks
        }

    diagnostics = {}
    for label, document in documents.items():
        grouped: dict[str, list[dict[str, Any]]] = {task: [] for task in tasks}
        for identity, record in zip(
            document["sample_manifest"], document["generation_records"], strict=True
        ):
            grouped[identity["task"]].append(record)
        diagnostics[label] = {
            task: {
                "mean_response_tokens": sum(
                    int(record["response_tokens"]) for record in records
                )
                / len(records),
                "reached_token_cap_fraction": sum(
                    int(record["response_tokens"]) >= int(record["max_response_tokens"])
                    for record in records
                )
                / len(records),
                "mean_cache_events": sum(
                    len(record["cache_event_lengths"]) for record in records
                )
                / len(records),
            }
            for task, records in grouped.items()
        }

    return {
        "schema_version": 1,
        "kind": "opsd_three_way_paired_control_aggregate",
        "source_revision": documents["off"]["source_revision"],
        "conditions": list(labels),
        "tasks": list(tasks),
        "limit_per_task": limit,
        "prompt_count": len(sample_manifest),
        "sample_manifest_sha256": hashlib.sha256(
            json.dumps(sample_manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_numpy_version": np.__version__,
        "bootstrap_quantile_method": "linear",
        "bootstrap_indices_sha256": draws_digest.hexdigest(),
        "macro_note": (
            "descriptive only: GSM-Hard is derived from GSM8K and tasks are not independent"
        ),
        "checkpoint_manifests": {
            label: documents[label]["checkpoint_manifest"] for label in labels
        },
        "accuracy": accuracy,
        "extraction_sensitivity_metrics": {
            label: _condition_metrics(documents[label]) for label in labels
        },
        "paired_contrasts": paired,
        "response_identity_counts": response_identity,
        "diagnostics": diagnostics,
    }


def audit_opsd_prompt_overlap(
    training_rows: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    *,
    expected_training_rows: int = 128,
    expected_evaluation_prompts: int = 192,
) -> dict[str, Any]:
    labels = ("off", "untrained", "trained")
    if set(documents) != set(labels):
        raise ValueError("OPSD overlap audit requires off, untrained, and trained documents")
    source_indices = [int(row["source_index"]) for row in training_rows]
    if len(training_rows) != expected_training_rows or len(set(source_indices)) != len(
        source_indices
    ):
        raise ValueError("OPSD overlap audit training manifest is incomplete or duplicated")
    hash_keys = ("prompt_text_sha256", "normalized_prompt_text_sha256")
    if any(any(not row.get(key) for key in hash_keys) for row in training_rows):
        raise ValueError("OPSD overlap audit training rows lack prompt hashes")

    reference_records = documents["off"]["generation_records"]
    if len(reference_records) != expected_evaluation_prompts:
        raise ValueError("OPSD overlap audit evaluation manifest has the wrong size")
    for label in labels[1:]:
        records = documents[label]["generation_records"]
        if any(
            [record.get(key) for record in records]
            != [record.get(key) for record in reference_records]
            for key in hash_keys
        ):
            raise ValueError("OPSD overlap audit evaluation manifests differ")
    if any(any(not record.get(key) for key in hash_keys) for record in reference_records):
        raise ValueError("OPSD overlap audit evaluation records lack prompt hashes")

    training_by_hash = {
        key: {
            str(row[key]): int(row["source_index"])
            for row in training_rows
        }
        for key in hash_keys
    }
    evaluation_by_hash = {
        key: {
            str(record[key]): int(record["request_index"])
            for record in reference_records
        }
        for key in hash_keys
    }
    overlaps = {}
    for key in hash_keys:
        shared = sorted(training_by_hash[key].keys() & evaluation_by_hash[key].keys())
        overlaps[key] = [
            {
                "sha256": value,
                "training_source_index": training_by_hash[key][value],
                "evaluation_request_index": evaluation_by_hash[key][value],
            }
            for value in shared
        ]
    training_manifest_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "source_index": int(row["source_index"]),
                    **{key: row[key] for key in hash_keys},
                }
                for row in training_rows
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    evaluation_manifest_sha256 = hashlib.sha256(
        json.dumps(
            [{key: record[key] for key in hash_keys} for record in reference_records],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "kind": "opsd_training_evaluation_prompt_overlap_audit",
        "training_rows": len(training_rows),
        "evaluation_prompts": len(reference_records),
        "training_manifest_sha256": training_manifest_sha256,
        "evaluation_manifest_sha256": evaluation_manifest_sha256,
        "exact_text_overlap_count": len(overlaps["prompt_text_sha256"]),
        "normalized_text_overlap_count": len(
            overlaps["normalized_prompt_text_sha256"]
        ),
        "held_out_prompt_text_verified": not any(overlaps.values()),
        "overlaps": overlaps,
    }


def _condition_metrics(document: dict[str, Any]) -> dict[str, Any]:
    task_results = document["lm_eval"]["results"]
    metrics: dict[str, Any] = {}
    for task in document["tasks"]:
        row = task_results[task]
        task_metrics = {
            key: float(value)
            for key, value in row.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and key.startswith(("exact_match,", "acc,"))
        }
        if not task_metrics:
            raise ValueError(f"task {task!r} has no supported finite accuracy metric")
        metrics[task] = task_metrics
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
