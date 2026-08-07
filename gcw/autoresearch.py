from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def validate_autoresearch_protocol(
    protocol: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> None:
    if (
        protocol.get("schema_version") != 1
        or protocol.get("kind") != "opsd_h200_autoresearch_campaign"
        or protocol.get("campaign_id") != "qwen3_06b_opsd_h200_autoresearch_v1"
    ):
        raise ValueError("autoresearch protocol identity mismatch")
    wall = protocol.get("maximum_wall_seconds")
    reserve = protocol.get("matrix_reserve_seconds")
    if (
        not isinstance(wall, int)
        or isinstance(wall, bool)
        or not 0 < wall <= 86400
        or not isinstance(reserve, int)
        or isinstance(reserve, bool)
        or not 0 < reserve < wall
    ):
        raise ValueError("autoresearch wall-time contract is invalid")
    hard = protocol.get("hard_invariants")
    if not isinstance(hard, Mapping):
        raise ValueError("autoresearch hard invariants are missing")
    expected_hard = {
        "freeze_backbone": True,
        "teacher_trajectory_mode": "native_full",
        "response_process_every": 64,
        "teacher_thinking": True,
        "student_thinking": False,
        "converted_backbone_label": (
            "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"
        ),
        "no_test_set_model_selection": True,
        "no_source_mutation_during_gpu_window": True,
        "stop_at_deadline": True,
    }
    if dict(hard) != expected_hard:
        raise ValueError("autoresearch hard invariant contract mismatch")
    if (
        protocol.get("teacher_precompute_initial_rows") != 128
        or protocol.get("integrated_probe_examples") != 16
        or not isinstance(protocol.get("minimum_teacher_realized_batch_size"), int)
        or isinstance(protocol.get("minimum_teacher_realized_batch_size"), bool)
        or int(protocol["minimum_teacher_realized_batch_size"]) <= 0
    ):
        raise ValueError("autoresearch teacher pilot contract is invalid")
    distill = base_config.get("distillation")
    metadata = base_config.get("metadata")
    if not isinstance(distill, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("autoresearch base config lacks distillation metadata")
    if (
        base_config.get("freeze_backbone") is not True
        or distill.get("teacher_trajectory_mode") != hard["teacher_trajectory_mode"]
        or distill.get("response_process_every") != hard["response_process_every"]
        or distill.get("teacher_thinking") is not hard["teacher_thinking"]
        or distill.get("student_thinking") is not hard["student_thinking"]
        or metadata.get("backbone_label") != hard["converted_backbone_label"]
    ):
        raise ValueError("autoresearch base config violates a hard invariant")
    evaluation = protocol.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("autoresearch evaluation contract is missing")
    worker_values = {
        "teacher_precompute_workers": protocol.get("teacher_precompute_workers"),
        "capacity_workers": protocol.get("capacity_workers"),
        "evaluation.max_workers": evaluation.get("max_workers"),
    }
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= 8
        for value in worker_values.values()
    ):
        raise ValueError(
            "autoresearch worker counts must be reviewed integers between 1 and 8"
        )
    tasks = evaluation.get("tasks")
    conditions = evaluation.get("conditions")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 5
        or len(set(tasks)) != 5
        or not isinstance(conditions, list)
        or [row.get("name") for row in conditions if isinstance(row, Mapping)]
        != [
            "cot-off__cp-off",
            "cot-off__cp-on",
            "cot-on__cp-off",
            "cot-on__cp-on",
        ]
    ):
        raise ValueError("autoresearch evaluation grid is not the reviewed 5×4 contract")


def select_disjoint_heldout_indices(
    valid_indices: Sequence[int],
    *,
    locked_examples: int,
    candidate_examples: int,
    minimum_examples: int,
    reserved_non_heldout_examples: int = 0,
) -> list[int]:
    indices = [int(index) for index in valid_indices]
    if (
        indices != sorted(set(indices))
        or locked_examples <= 0
        or candidate_examples <= 0
        or minimum_examples <= 0
        or reserved_non_heldout_examples < 0
    ):
        raise ValueError("held-out selection contract is invalid")
    eligible = indices[locked_examples:]
    reserved = max(locked_examples, reserved_non_heldout_examples)
    maximum_heldout = max(0, len(indices) - reserved)
    heldout_count = min(candidate_examples, len(eligible), maximum_heldout)
    heldout = eligible[-heldout_count:] if heldout_count else []
    if len(heldout) < minimum_examples:
        raise RuntimeError("teacher pilot lacks the minimum held-out pool after locking capacity examples")
    return heldout


def decision_dose_steps(
    measured_speedup: float,
    *,
    below_4x: int,
    middle: int,
    high: int,
) -> int:
    if not math.isfinite(measured_speedup) or measured_speedup <= 0:
        raise ValueError("measured_speedup must be positive and finite")
    if measured_speedup < 4.0:
        return below_4x
    if measured_speedup < 7.5:
        return middle
    return high


def project_end_to_end_training(
    *,
    baseline_effective_step_seconds: float,
    integrated_step_seconds: float,
    teacher_pilot_seconds: float,
    teacher_pilot_rows: int,
    valid_teacher_traces: int,
    optimizer_steps: int,
    effective_batch: int,
) -> dict[str, float | int]:
    positive = (
        baseline_effective_step_seconds,
        integrated_step_seconds,
        teacher_pilot_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        raise ValueError("timing inputs must be positive and finite")
    if (
        teacher_pilot_rows <= 0
        or valid_teacher_traces <= 0
        or valid_teacher_traces > teacher_pilot_rows
        or optimizer_steps <= 0
        or effective_batch <= 0
    ):
        raise ValueError("projection counts must be positive and valid")
    yield_fraction = valid_teacher_traces / teacher_pilot_rows
    required_valid = optimizer_steps * effective_batch
    projected_teacher_rows = math.ceil(required_valid / yield_fraction)
    teacher_seconds_per_row = teacher_pilot_seconds / teacher_pilot_rows
    projected_teacher_seconds = projected_teacher_rows * teacher_seconds_per_row
    projected_training_seconds = optimizer_steps * integrated_step_seconds
    projected_total_seconds = projected_teacher_seconds + projected_training_seconds
    baseline_total_seconds = optimizer_steps * baseline_effective_step_seconds
    return {
        "teacher_valid_yield_fraction": yield_fraction,
        "teacher_seconds_per_row": teacher_seconds_per_row,
        "projected_teacher_rows": projected_teacher_rows,
        "projected_teacher_seconds": projected_teacher_seconds,
        "projected_training_seconds": projected_training_seconds,
        "projected_total_seconds": projected_total_seconds,
        "end_to_end_speedup": baseline_total_seconds / projected_total_seconds,
    }


def feasible_dose_steps(
    requested_steps: int,
    *,
    valid_traces: int,
    effective_batch: int,
    candidates: Sequence[int] = (64, 32, 16),
) -> int:
    if requested_steps < 0 or valid_traces < 0 or effective_batch <= 0:
        raise ValueError("dose, trace count, and effective batch must be non-negative/positive")
    if any(step <= 0 for step in candidates):
        raise ValueError("dose candidates must be positive")
    capacity = valid_traces // effective_batch
    eligible = [step for step in candidates if step <= requested_steps and step <= capacity]
    return max(eligible, default=0)


def choose_capacity_cells(
    summaries: dict[str, dict[str, float]],
    *,
    minimum_reduction_fraction: float,
    maximum_cells: int,
) -> list[str]:
    if not 0.0 <= minimum_reduction_fraction <= 1.0:
        raise ValueError("minimum_reduction_fraction must lie in [0, 1]")
    if maximum_cells <= 0:
        raise ValueError("maximum_cells must be positive")
    scored: list[tuple[float, str]] = []
    for cell_id, summary in summaries.items():
        initial = float(summary["initial_loss"])
        final = float(summary["final_loss"])
        if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0:
            continue
        reduction = (initial - final) / initial
        if reduction >= minimum_reduction_fraction:
            scored.append((reduction, cell_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cell_id for _, cell_id in scored[:maximum_cells]]


def require_complete_capacity_promotion(
    expected_cell_ids: Sequence[str],
    summaries: Mapping[str, Mapping[str, Any]],
    promoted: Sequence[str],
) -> str:
    expected = tuple(expected_cell_ids)
    if len(expected) != len(set(expected)) or not expected:
        raise ValueError("capacity cell IDs must be unique and non-empty")
    missing = [cell_id for cell_id in expected if cell_id not in summaries]
    extra = sorted(set(summaries) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"capacity screen incomplete: missing={missing}, extra={extra}"
        )
    if not promoted:
        raise RuntimeError("capacity screen completed but no cell passed promotion")
    selected = promoted[0]
    if selected not in summaries:
        raise RuntimeError(f"promoted capacity cell lacks a summary: {selected}")
    return selected


def require_heldout_improvement(
    summary: Mapping[str, Any],
    *,
    minimum_reduction_fraction: float,
) -> Mapping[str, Any]:
    if not 0.0 <= minimum_reduction_fraction <= 1.0:
        raise ValueError("held-out minimum reduction must lie in [0, 1]")
    heldout = summary.get("heldout_validation")
    if not isinstance(heldout, Mapping):
        raise RuntimeError("candidate summary lacks held-out KL evidence")
    reduction = heldout.get("reduction_fraction")
    initial = heldout.get("initial_kl")
    final = heldout.get("final_kl")
    values = (reduction, initial, final)
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in values
    ):
        raise RuntimeError("held-out KL evidence contains non-finite metrics")
    assert isinstance(reduction, (int, float)) and not isinstance(reduction, bool)
    assert isinstance(initial, (int, float)) and not isinstance(initial, bool)
    assert isinstance(final, (int, float)) and not isinstance(final, bool)
    reduction_value = float(reduction)
    initial_value = float(initial)
    final_value = float(final)
    if (
        initial_value <= 0
        or final_value < 0
        or reduction_value < minimum_reduction_fraction
    ):
        raise RuntimeError(
            "held-out KL gate failed: "
            f"reduction={reduction_value:.6f}, required={minimum_reduction_fraction:.6f}"
        )
    return heldout


def require_realized_teacher_batch(
    *, selected_batch: int, realized_batch: int, minimum_batch: int
) -> None:
    if selected_batch <= 0 or realized_batch <= 0 or minimum_batch <= 0:
        raise ValueError("teacher batch sizes must be positive")
    if selected_batch < minimum_batch:
        raise RuntimeError(
            "synthetic benchmark selected a teacher batch below the reviewed real-pilot minimum"
        )
    if realized_batch > selected_batch:
        raise RuntimeError("teacher precompute exceeded the selected batch size")
    if realized_batch < minimum_batch:
        raise RuntimeError(
            "teacher trace pilot did not exercise the predeclared minimum real batch size"
        )


def materialize_train_config(
    base: dict[str, Any],
    *,
    experiment_id: str,
    output_dir: str,
    learning_rate: float,
    gate_init: float,
    effective_batch: int,
    execution_microbatch: int,
    optimizer_steps: int,
    take_rows: int,
    visible_trajectory_mode: str,
    displacement_audit_every: int,
) -> dict[str, Any]:
    if not experiment_id:
        raise ValueError("experiment_id must not be empty")
    if effective_batch <= 0 or execution_microbatch <= 0:
        raise ValueError("batch sizes must be positive")
    if effective_batch % execution_microbatch != 0:
        raise ValueError("execution microbatch must divide effective batch")
    if optimizer_steps <= 0 or take_rows <= 0:
        raise ValueError("optimizer_steps and take_rows must be positive")
    if visible_trajectory_mode not in {"student", "both"}:
        raise ValueError("visible_trajectory_mode must be student or both")
    result = copy.deepcopy(base)
    result["output_dir"] = output_dir
    result["learning_rate"] = learning_rate
    result["model"]["cache_processor"]["gate_init"] = gate_init
    result["data"]["sft_take_rows"] = take_rows
    distill = result["distillation"]
    distill["gradient_accumulation_steps"] = effective_batch
    distill["execution_microbatch_size"] = execution_microbatch
    distill["require_precomputed_teacher_traces"] = True
    distill["max_optimizer_steps"] = optimizer_steps
    distill["visible_trajectory_mode"] = visible_trajectory_mode
    distill["record_cache_displacements"] = True
    distill[
        "cache_displacement_audit_every_optimizer_steps"
    ] = displacement_audit_every
    wandb = result.setdefault("wandb", {})
    wandb["name"] = experiment_id
    wandb["group"] = "opsd-h200-autoresearch-v1"
    tags = list(wandb.get("tags", []))
    for tag in (
        "h200-autoresearch-v1",
        f"effective-batch-{effective_batch}",
        f"execution-microbatch-{execution_microbatch}",
        f"steps-{optimizer_steps}",
    ):
        if tag not in tags:
            tags.append(tag)
    wandb["tags"] = tags
    metadata = result.setdefault("metadata", {})
    metadata.update(
        {
            "campaign": "qwen3_06b_opsd_h200_autoresearch_v1",
            "experiment_id": experiment_id,
            "distillation_effective_batch_examples": effective_batch,
            "execution_microbatch_size": execution_microbatch,
            "requested_optimizer_steps": optimizer_steps,
        }
    )
    return result


def matrix_generation_count(
    tasks: Sequence[str],
    *,
    conditions: int,
    limit: int | None,
    full_task_counts: Mapping[str, int],
) -> int:
    if conditions <= 0:
        raise ValueError("conditions must be positive")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("tasks must be non-empty and unique")
    if limit is not None and limit <= 0:
        raise ValueError("evaluation limit must be positive when set")
    try:
        per_condition = sum(
            full_task_counts[task]
            if limit is None
            else min(limit, full_task_counts[task])
            for task in tasks
        )
    except KeyError as exc:
        raise ValueError(f"missing full sample count for task {exc.args[0]!r}") from exc
    if any(full_task_counts[task] <= 0 for task in tasks):
        raise ValueError("full task counts must be positive")
    return conditions * per_condition


def select_evaluation_limit(
    *,
    sentinel_elapsed_seconds: float,
    sentinel_generations: int,
    remaining_seconds: float,
    closure_reserve_seconds: float,
    tasks: Sequence[str],
    conditions: int,
    candidate_limits: Sequence[int],
    full_task_counts: Mapping[str, int],
    safety_factor: float,
) -> int | None:
    values = (
        sentinel_elapsed_seconds,
        remaining_seconds,
        closure_reserve_seconds,
        safety_factor,
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("evaluation timing values must be finite")
    if sentinel_elapsed_seconds <= 0 or sentinel_generations <= 0:
        raise ValueError("sentinel timing and generation count must be positive")
    if remaining_seconds <= 0 or closure_reserve_seconds < 0 or safety_factor < 1:
        raise ValueError("evaluation budget or safety factor is invalid")
    if any(limit <= 0 for limit in candidate_limits):
        raise ValueError("candidate evaluation limits must be positive")
    available = remaining_seconds - closure_reserve_seconds
    seconds_per_generation = sentinel_elapsed_seconds / sentinel_generations
    scopes: tuple[int | None, ...] = (None, *sorted(set(candidate_limits), reverse=True))
    for limit in scopes:
        generations = matrix_generation_count(
            tasks,
            conditions=conditions,
            limit=limit,
            full_task_counts=full_task_counts,
        )
        if seconds_per_generation * generations * safety_factor <= available:
            return limit
    return 0


def validate_opsd_task_shard(
    document: Mapping[str, Any],
    *,
    condition_name: str,
    task: str,
    limit: int | None,
    full_task_counts: Mapping[str, int],
) -> None:
    if document.get("kind") != "opsd_quick_screen_condition":
        raise ValueError("OPSD shard has the wrong document kind")
    condition = document.get("condition")
    if not isinstance(condition, Mapping) or condition.get("name") != condition_name:
        raise ValueError("OPSD shard condition mismatch")
    if document.get("tasks") != [task]:
        raise ValueError("OPSD shard must contain exactly its declared task")
    if document.get("limit_per_task") != limit:
        raise ValueError("OPSD shard limit mismatch")
    expected = matrix_generation_count(
        (task,),
        conditions=1,
        limit=limit,
        full_task_counts=full_task_counts,
    )
    manifest = document.get("sample_manifest")
    records = document.get("generation_records")
    if not isinstance(manifest, list) or not isinstance(records, list):
        raise ValueError("OPSD shard lacks sample or generation records")
    if len(manifest) != expected or len(records) != expected:
        raise ValueError(
            "OPSD shard record count mismatch: "
            f"samples={len(manifest)}, generations={len(records)}, expected={expected}"
        )
    if any(not isinstance(row, Mapping) or row.get("task") != task for row in manifest):
        raise ValueError("OPSD shard sample manifest contains another task")
    sample_identities = []
    for row in manifest:
        assert isinstance(row, Mapping)
        required = ("doc_id", "doc_hash", "prompt_hash", "target_hash")
        if any(key not in row or row[key] in (None, "") for key in required):
            raise ValueError("OPSD shard sample manifest lacks required identity fields")
        sample_identities.append((str(row["doc_id"]), str(row["doc_hash"])))
    if len(set(sample_identities)) != expected:
        raise ValueError("OPSD shard sample identities must be unique")
    if [row.get("request_index") for row in records if isinstance(row, Mapping)] != list(
        range(expected)
    ):
        raise ValueError("OPSD shard request indices are incomplete or unordered")
    for sample, record in zip(manifest, records, strict=True):
        if not isinstance(sample, Mapping) or not isinstance(record, Mapping):
            raise ValueError("OPSD shard contains a malformed sample or generation record")
        if record.get("prompt_sha256") != sample.get("prompt_hash"):
            raise ValueError("OPSD shard generation record does not match its sample manifest")
    for key in ("task_config_sha256", "dataset_files"):
        value = document.get(key)
        if not isinstance(value, Mapping) or set(value) != {task}:
            raise ValueError(f"OPSD shard {key} does not bind exactly one task")
    lm_eval = document.get("lm_eval")
    if not isinstance(lm_eval, Mapping):
        raise ValueError("OPSD shard lacks lm-eval evidence")
    results = lm_eval.get("results")
    if not isinstance(results, Mapping) or set(results) != {task}:
        raise ValueError("OPSD shard lm-eval results must bind exactly the declared task")
    samples = lm_eval.get("samples")
    if not isinstance(samples, Mapping) or set(samples) != {task}:
        raise ValueError("OPSD shard lm-eval samples must bind exactly the declared task")
    sample_rows = samples[task]
    if not isinstance(sample_rows, list):
        raise ValueError("OPSD shard lm-eval task samples must be a list")
    reconstructed_manifest: list[dict[str, Any]] = []
    seen_lm_eval_identities: set[tuple[Any, ...]] = set()
    required_identity = ("doc_id", "doc_hash", "prompt_hash", "target_hash")
    for row in sample_rows:
        if not isinstance(row, Mapping) or any(key not in row for key in required_identity):
            raise ValueError("OPSD shard lm-eval sample identity fields are incomplete")
        identity = (task, *(row[key] for key in required_identity))
        if identity in seen_lm_eval_identities:
            continue
        seen_lm_eval_identities.add(identity)
        reconstructed_manifest.append(
            {"task": task, **{key: row[key] for key in required_identity}}
        )
    if reconstructed_manifest != manifest:
        raise ValueError("OPSD shard sample manifest differs from ordered lm-eval evidence")
    for key in ("configs", "versions", "n-shot", "higher_is_better", "n-samples"):
        value = lm_eval.get(key)
        if value is not None and (not isinstance(value, Mapping) or set(value) != {task}):
            raise ValueError(f"OPSD shard lm-eval {key} must bind exactly the declared task")


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def merge_opsd_task_shards(
    documents: Sequence[Mapping[str, Any]],
    *,
    tasks: Sequence[str],
    full_task_counts: Mapping[str, int],
) -> dict[str, Any]:
    if not documents:
        raise ValueError("OPSD shard merge requires documents")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("OPSD shard merge tasks must be non-empty and unique")
    by_task: dict[str, Mapping[str, Any]] = {}
    for document in documents:
        shard_tasks = document.get("tasks")
        if not isinstance(shard_tasks, list) or len(shard_tasks) != 1:
            raise ValueError("OPSD shard merge received a non-single-task document")
        task = str(shard_tasks[0])
        if task in by_task:
            raise ValueError(f"duplicate OPSD shard for task {task}")
        by_task[task] = document
    if set(by_task) != set(tasks):
        raise ValueError("OPSD shard set does not match the requested task set")
    ordered = [by_task[task] for task in tasks]
    condition = ordered[0].get("condition")
    if not isinstance(condition, Mapping) or not isinstance(condition.get("name"), str):
        raise ValueError("OPSD shard condition is malformed")
    condition_name = str(condition["name"])
    limit = ordered[0].get("limit_per_task")
    if limit is not None and not isinstance(limit, int):
        raise ValueError("OPSD shard limit must be an integer or null")
    shared_keys = (
        "schema_version",
        "kind",
        "source_revision",
        "tokenizer_source",
        "tokenizer_identity_sha256",
        "datasets_cache",
        "condition",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_manifest",
        "model_config",
        "train_state",
        "limit_per_task",
        "execution_batch_size",
        "generation_policy",
    )
    for document in ordered:
        validate_opsd_task_shard(
            document,
            condition_name=condition_name,
            task=str(document["tasks"][0]),
            limit=limit,
            full_task_counts=full_task_counts,
        )
        for key in shared_keys:
            if document.get(key) != ordered[0].get(key):
                raise ValueError(f"OPSD task shards differ in {key}")

    merged = copy.deepcopy(dict(ordered[0]))
    merged["tasks"] = list(tasks)
    merged["task_config_sha256"] = {
        task: by_task[task]["task_config_sha256"][task] for task in tasks
    }
    merged["dataset_files"] = {
        task: by_task[task]["dataset_files"][task] for task in tasks
    }
    merged["sample_manifest"] = [
        copy.deepcopy(row)
        for task in tasks
        for row in by_task[task]["sample_manifest"]
    ]
    merged_records: list[dict[str, Any]] = []
    for task in tasks:
        for record in by_task[task]["generation_records"]:
            copied = copy.deepcopy(record)
            copied["shard_request_index"] = copied["request_index"]
            copied["request_index"] = len(merged_records)
            merged_records.append(copied)
    merged["generation_records"] = merged_records

    merged_lm_eval = copy.deepcopy(dict(ordered[0]["lm_eval"]))
    task_mapping_keys = (
        "results",
        "samples",
        "configs",
        "versions",
        "n-shot",
        "higher_is_better",
        "n-samples",
        "group_subtasks",
        "groups",
    )
    for key in task_mapping_keys:
        combined: dict[str, Any] = {}
        present = False
        for document in ordered:
            value = document["lm_eval"].get(key)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise ValueError(f"OPSD shard lm-eval {key} is not a mapping")
            present = True
            for item_key, item_value in value.items():
                if item_key in combined and combined[item_key] != item_value:
                    raise ValueError(f"OPSD shard lm-eval {key} conflicts at {item_key}")
                combined[str(item_key)] = copy.deepcopy(item_value)
        if present:
            merged_lm_eval[key] = combined
    merged["lm_eval"] = merged_lm_eval
    merged["shards"] = [
        {
            "task": task,
            "payload_sha256": _payload_sha256(by_task[task]),
            "sample_count": len(by_task[task]["sample_manifest"]),
        }
        for task in tasks
    ]
    merged["sharded_evaluation"] = True
    return merged
