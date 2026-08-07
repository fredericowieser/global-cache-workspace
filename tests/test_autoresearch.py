from __future__ import annotations

import copy
import json
import os
import signal
import subprocess
import sys
import time

import pytest
import yaml

from scripts.run_opsd_h200_autoresearch import (
    Campaign,
    CampaignCommandTimeout,
    ProcessOutcome,
    _benchmark_choices,
    _process_group_exists,
    _run_process,
    _runtime_matrix_reserve,
    _terminate_registered_process_groups,
    _validate_spend_approval,
)

from gcw.autoresearch import (
    choose_capacity_cells,
    decision_dose_steps,
    feasible_dose_steps,
    materialize_train_config,
    matrix_generation_count,
    merge_opsd_task_shards,
    project_end_to_end_training,
    require_complete_capacity_promotion,
    require_heldout_improvement,
    require_realized_teacher_batch,
    select_evaluation_limit,
    validate_opsd_task_shard,
    validate_autoresearch_protocol,
)


def test_decision_dose_steps_uses_predeclared_speed_gates():
    assert decision_dose_steps(3.99, below_4x=0, middle=32, high=64) == 0
    assert decision_dose_steps(4.0, below_4x=0, middle=32, high=64) == 32
    assert decision_dose_steps(7.49, below_4x=0, middle=32, high=64) == 32
    assert decision_dose_steps(7.5, below_4x=0, middle=32, high=64) == 64


def test_runtime_matrix_reserve_scales_to_authorised_window():
    assert _runtime_matrix_reserve(
        deadline_seconds=43_200,
        configured_reserve_seconds=43_200,
        closure_reserve_seconds=3_600,
    ) == 21_600
    assert _runtime_matrix_reserve(
        deadline_seconds=86_400,
        configured_reserve_seconds=43_200,
        closure_reserve_seconds=3_600,
    ) == 43_200
    with pytest.raises(ValueError, match="leave work time"):
        _runtime_matrix_reserve(
            deadline_seconds=7_200,
            configured_reserve_seconds=43_200,
            closure_reserve_seconds=3_600,
        )


def test_process_group_cleanup_kills_term_ignoring_descendants():
    child_code = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "time.sleep(60)"
    )
    leader_code = (
        "import signal,subprocess,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", leader_code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(0.1)
        _terminate_registered_process_groups((process,), grace_seconds=0.1)
        deadline = time.monotonic() + 2.0
        while _process_group_exists(process.pid) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert process.poll() is not None
        assert not _process_group_exists(process.pid)
    finally:
        if _process_group_exists(process.pid):
            os.killpg(process.pid, signal.SIGKILL)


def test_shutdown_requested_before_spawn_never_calls_popen(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.run_opsd_h200_autoresearch._SHUTDOWN_REQUESTED", True)
    monkeypatch.setattr(
        "scripts.run_opsd_h200_autoresearch.subprocess.Popen",
        lambda *args, **kwargs: pytest.fail("Popen must not run after shutdown"),
    )
    with (tmp_path / "log").open("w") as log:
        outcome = _run_process(["never"], cwd=tmp_path, log=log, timeout=1.0)
    assert outcome.returncode == 130
    assert outcome.error_kind == "interrupted"


def test_spawned_child_receives_sigterm_with_unblocked_mask(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_opsd_h200_autoresearch._SHUTDOWN_REQUESTED", False
    )
    code = (
        "import os,signal,sys,time; "
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); "
        "os.kill(os.getpid(), signal.SIGTERM); time.sleep(60)"
    )
    with (tmp_path / "log").open("w") as log:
        outcome = _run_process(
            [sys.executable, "-c", code], cwd=tmp_path, log=log, timeout=1.0
        )
    assert outcome.returncode == 0
    assert outcome.error_kind is None


def test_spend_approval_phrase_binds_exact_envelope_and_finite_money(tmp_path):
    source_revision = "a" * 40
    protocol_sha256 = "b" * 64
    approval = {
        "kind": "gcw_h200_spend_approval_v1",
        "source_revision": source_revision,
        "protocol_sha256": protocol_sha256,
        "provider": "nebius",
        "project_id": "project-e01d0ymrpr004k0s993tba",
        "region": "eu-west1",
        "gpu_model": "H200",
        "gpu_count": 1,
        "regular_capacity": True,
        "approved_hours": 12.0,
        "quoted_usd_per_gpu_hour": 4.5,
        "maximum_disk_spend_usd": 0.35,
        "maximum_spend_usd": 55.0,
        "approval_phrase": "APPROVE 12H $55",
    }
    path = tmp_path / "approval.json"
    path.write_text(json.dumps(approval))
    assert _validate_spend_approval(
        path,
        source_revision=source_revision,
        protocol_sha256=protocol_sha256,
        requested_hours=12.0,
    ) == approval

    approval["maximum_spend_usd"] = 110.0
    path.write_text(json.dumps(approval))
    with pytest.raises(RuntimeError, match="exact approved hours"):
        _validate_spend_approval(
            path,
            source_revision=source_revision,
            protocol_sha256=protocol_sha256,
            requested_hours=12.0,
        )

    approval["maximum_spend_usd"] = float("nan")
    path.write_text(json.dumps(approval))
    with pytest.raises(RuntimeError, match="must be finite"):
        _validate_spend_approval(
            path,
            source_revision=source_revision,
            protocol_sha256=protocol_sha256,
            requested_hours=12.0,
        )


def test_feasible_dose_steps_downgrades_to_authenticated_trace_capacity():
    assert feasible_dose_steps(64, valid_traces=1024, effective_batch=16) == 64
    assert feasible_dose_steps(64, valid_traces=800, effective_batch=16) == 32
    assert feasible_dose_steps(32, valid_traces=300, effective_batch=16) == 16
    assert feasible_dose_steps(32, valid_traces=200, effective_batch=16) == 0


def test_choose_capacity_cells_filters_gate_and_ranks_reduction():
    summaries = {
        "weak": {"initial_loss": 1.0, "final_loss": 0.3},
        "best": {"initial_loss": 1.0, "final_loss": 0.05},
        "second": {"initial_loss": 2.0, "final_loss": 0.3},
    }

    assert choose_capacity_cells(
        summaries,
        minimum_reduction_fraction=0.8,
        maximum_cells=2,
    ) == ["best", "second"]


def test_capacity_promotion_requires_complete_grid_and_a_passing_cell():
    expected = ["a", "b", "c", "d"]
    summaries = {cell: {"final_loss": 0.1} for cell in expected}
    assert require_complete_capacity_promotion(expected, summaries, ["c"]) == "c"
    with pytest.raises(RuntimeError, match="incomplete"):
        require_complete_capacity_promotion(expected, {"a": summaries["a"]}, ["a"])
    with pytest.raises(RuntimeError, match="no cell passed"):
        require_complete_capacity_promotion(expected, summaries, [])


def test_heldout_gate_requires_finite_configured_reduction():
    passing = {"heldout_validation": {"reduction_fraction": 0.11, "initial_kl": 1.0, "final_kl": 0.89}}
    assert require_heldout_improvement(passing, minimum_reduction_fraction=0.1)[
        "reduction_fraction"
    ] == 0.11
    with pytest.raises(RuntimeError, match="held-out KL gate failed"):
        require_heldout_improvement(
            {"heldout_validation": {"reduction_fraction": 0.09, "initial_kl": 1.0, "final_kl": 0.91}},
            minimum_reduction_fraction=0.1,
        )
    with pytest.raises(RuntimeError, match="lacks held-out"):
        require_heldout_improvement({}, minimum_reduction_fraction=0.1)


def test_end_to_end_projection_includes_teacher_generation_and_training():
    projection = project_end_to_end_training(
        baseline_effective_step_seconds=100.0,
        integrated_step_seconds=10.0,
        teacher_pilot_seconds=20.0,
        teacher_pilot_rows=10,
        valid_teacher_traces=5,
        optimizer_steps=100,
        effective_batch=4,
    )
    assert projection["projected_teacher_rows"] == 800
    assert projection["projected_teacher_seconds"] == 1600.0
    assert projection["projected_training_seconds"] == 1000.0
    assert projection["projected_total_seconds"] == 2600.0
    assert projection["end_to_end_speedup"] == pytest.approx(10000.0 / 2600.0)


def test_batch_benchmark_selects_teacher_and_student_independently():
    def row(batch, teacher_tps, student_tps):
        return {
            "status": "passed",
            "teacher_status": "passed",
            "student_status": "passed",
            "batch_size": batch,
            "teacher_tokens_per_second": teacher_tps,
            "student_tokens_per_second": student_tps,
            "teacher_batch_matches_serial": True,
            "student_batch_matches_serial": True,
            "student_event_schedules_match_serial": True,
            "teacher_peak_cuda_memory_bytes": 1_000,
            "student_peak_cuda_memory_bytes": 2_000,
        }

    teacher_batch, student_batch, teacher_speedup, student_speedup, peak = (
        _benchmark_choices(
            {"rows": [row(1, 10.0, 10.0), row(2, 30.0, 15.0), row(4, 20.0, 40.0)]}
        )
    )
    assert (teacher_batch, student_batch) == (2, 4)
    assert (teacher_speedup, student_speedup) == (3.0, 4.0)
    assert peak == 2_000


def test_batch_benchmark_keeps_valid_side_after_asymmetric_oom():
    rows = [
        {
            "batch_size": 1,
            "teacher_status": "passed",
            "student_status": "passed",
            "teacher_tokens_per_second": 10.0,
            "student_tokens_per_second": 10.0,
            "teacher_batch_matches_serial": True,
            "student_batch_matches_serial": True,
            "student_event_schedules_match_serial": True,
            "teacher_peak_cuda_memory_bytes": 1,
            "student_peak_cuda_memory_bytes": 1,
        },
        {
            "batch_size": 2,
            "teacher_status": "oom",
            "student_status": "passed",
            "student_tokens_per_second": 30.0,
            "student_batch_matches_serial": True,
            "student_event_schedules_match_serial": True,
            "student_peak_cuda_memory_bytes": 2,
        },
    ]
    teacher_batch, student_batch, *_ = _benchmark_choices({"rows": rows})
    assert (teacher_batch, student_batch) == (1, 2)


def test_real_teacher_batch_gate_rejects_selected_or_realized_singleton():
    with pytest.raises(RuntimeError, match="selected a teacher batch"):
        require_realized_teacher_batch(
            selected_batch=1, realized_batch=1, minimum_batch=2
        )
    with pytest.raises(RuntimeError, match="did not exercise"):
        require_realized_teacher_batch(
            selected_batch=4, realized_batch=1, minimum_batch=2
        )
    require_realized_teacher_batch(selected_batch=4, realized_batch=2, minimum_batch=2)


def test_campaign_protocol_enforces_schema_and_hard_invariants():
    protocol = json.loads(
        open(
            "configs/protocols/qwen3_06b_opsd_h200_autoresearch_v1.json",
            encoding="utf-8",
        ).read()
    )
    base = yaml.safe_load(
        open(
            "configs/train/opsd/qwen3_06b_opsd_native_base_pragmatic_steps1000_bs16_v1.yaml",
            encoding="utf-8",
        )
    )
    validate_autoresearch_protocol(protocol, base)
    tampered = copy.deepcopy(protocol)
    tampered["hard_invariants"]["response_process_every"] = 32
    with pytest.raises(ValueError, match="hard invariant"):
        validate_autoresearch_protocol(tampered, base)


def test_materialize_train_config_changes_only_declared_campaign_fields():
    base = {
        "model": {"cache_processor": {"gate_init": -5.0}, "hidden_size": 1024},
        "data": {"sft_take_rows": 128000},
        "output_dir": "/old",
        "learning_rate": 0.0001,
        "distillation": {
            "gradient_accumulation_steps": 16,
            "max_optimizer_steps": 1000,
            "visible_trajectory_mode": "both",
            "record_cache_displacements": True,
        },
        "wandb": {"name": "old", "group": "old", "tags": ["existing"]},
        "metadata": {"campaign": "old"},
    }
    before = copy.deepcopy(base)

    result = materialize_train_config(
        base,
        experiment_id="lr3e-4_gate-3_bs4_steps32",
        output_dir="/new/run",
        learning_rate=0.0003,
        gate_init=-3.0,
        effective_batch=4,
        execution_microbatch=4,
        optimizer_steps=32,
        take_rows=2048,
        visible_trajectory_mode="both",
        displacement_audit_every=16,
    )

    assert base == before
    assert result["model"]["hidden_size"] == 1024
    assert result["model"]["cache_processor"]["gate_init"] == -3.0
    assert result["learning_rate"] == 0.0003
    assert result["output_dir"] == "/new/run"
    assert result["data"]["sft_take_rows"] == 2048
    assert result["distillation"]["gradient_accumulation_steps"] == 4
    assert result["distillation"]["execution_microbatch_size"] == 4
    assert result["distillation"]["require_precomputed_teacher_traces"] is True
    assert result["distillation"]["max_optimizer_steps"] == 32
    assert result["distillation"]["cache_displacement_audit_every_optimizer_steps"] == 16
    assert result["wandb"]["name"] == "lr3e-4_gate-3_bs4_steps32"
    assert result["metadata"]["campaign"] == "qwen3_06b_opsd_h200_autoresearch_v1"


def _opsd_shard(task: str, *, condition: str = "cot-off__cp-off", count: int = 2):
    thinking = condition.startswith("cot-on")
    processor = condition.endswith("cp-on")
    records = [
        {
            "request_index": index,
            "prompt_sha256": f"prompt-{task}-{index}",
            "base_token_ids_sha256": f"base-{task}-{index}",
            "full_prompt_token_ids_sha256": f"full-{task}-{index}",
            "raw_response_tokens": 1,
            "response_tokens": 1,
            "max_response_tokens": 512,
            "base_tokens": 4,
            "trigger_tokens": 2,
            "cache_event_positions": [0] if processor else [],
            "cache_event_lengths": [4] if processor else [],
        }
        for index in range(count)
    ]
    samples = [
        {
            "task": task,
            "doc_id": index,
            "doc_hash": f"doc-{task}-{index}",
            "prompt_hash": f"prompt-{task}-{index}",
            "target_hash": f"target-{task}-{index}",
        }
        for index in range(count)
    ]
    return {
        "schema_version": 1,
        "kind": "opsd_quick_screen_condition",
        "source_revision": "a" * 40,
        "checkpoint": "/checkpoint",
        "checkpoint_sha256": "checkpoint-sha",
        "checkpoint_manifest": {"manifest_sha256": "manifest"},
        "tokenizer_source": "/tokenizer",
        "tokenizer_identity_sha256": "tokenizer-sha",
        "datasets_cache": "/datasets",
        "condition": {
            "name": condition,
            "enable_thinking": thinking,
            "processor_enabled": processor,
            "process_every": 64,
            "select_all": False,
            "reasoning_open_text": "<think>\n",
            "max_response_tokens": 512,
        },
        "model_config": {"variant": "bgt"},
        "train_state": {"optimizer_steps": 1},
        "tasks": [task],
        "task_config_sha256": {task: f"config-{task}"},
        "dataset_files": {task: {"sha256": f"data-{task}"}},
        "limit_per_task": count,
        "execution_batch_size": 4,
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "fewshot": 0,
            "apply_chat_template_in_lm_eval": False,
        },
        "sample_manifest": samples,
        "generation_records": records,
        "lm_eval": {
            "results": {task: {"exact_match,strict-match": 0.5}},
            "samples": {
                task: [
                    {key: value for key, value in sample.items() if key != "task"}
                    for sample in samples
                ]
            },
            "configs": {task: {"task": task}},
            "git_hash": "lm-eval-sha",
        },
    }


def test_opsd_task_shards_validate_and_merge_in_canonical_order():
    tasks = ("task-a", "task-b")
    shards = [_opsd_shard("task-b"), _opsd_shard("task-a")]

    for shard in shards:
        validate_opsd_task_shard(
            shard,
            condition_name="cot-off__cp-off",
            task=shard["tasks"][0],
            limit=2,
            full_task_counts={"task-a": 3, "task-b": 4},
        )
    merged = merge_opsd_task_shards(
        shards,
        tasks=tasks,
        full_task_counts={"task-a": 3, "task-b": 4},
    )

    assert merged["tasks"] == ["task-a", "task-b"]
    assert [row["task"] for row in merged["sample_manifest"]] == [
        "task-a",
        "task-a",
        "task-b",
        "task-b",
    ]
    assert [row["request_index"] for row in merged["generation_records"]] == list(range(4))
    assert set(merged["lm_eval"]["results"]) == set(tasks)
    assert len(merged["shards"]) == 2


def test_opsd_task_shard_validation_fails_closed_on_wrong_count():
    shard = _opsd_shard("task-a", count=2)
    shard["generation_records"].pop()

    with pytest.raises(ValueError, match="record count"):
        validate_opsd_task_shard(
            shard,
            condition_name="cot-off__cp-off",
            task="task-a",
            limit=2,
            full_task_counts={"task-a": 3},
        )


def test_opsd_task_shard_validation_binds_records_to_unique_samples():
    duplicate = _opsd_shard("task-a", count=2)
    duplicate["sample_manifest"][1] = copy.deepcopy(duplicate["sample_manifest"][0])
    with pytest.raises(ValueError, match="unique"):
        validate_opsd_task_shard(
            duplicate,
            condition_name="cot-off__cp-off",
            task="task-a",
            limit=2,
            full_task_counts={"task-a": 3},
        )

    mismatch = _opsd_shard("task-a", count=2)
    mismatch["generation_records"][1]["prompt_sha256"] = "wrong"
    with pytest.raises(ValueError, match="sample manifest"):
        validate_opsd_task_shard(
            mismatch,
            condition_name="cot-off__cp-off",
            task="task-a",
            limit=2,
            full_task_counts={"task-a": 3},
        )

    wrong_lm_eval = _opsd_shard("task-a", count=2)
    wrong_lm_eval["lm_eval"]["results"]["extra"] = {}
    with pytest.raises(ValueError, match="lm-eval results"):
        validate_opsd_task_shard(
            wrong_lm_eval,
            condition_name="cot-off__cp-off",
            task="task-a",
            limit=2,
            full_task_counts={"task-a": 3},
        )


def test_sentinel_selects_largest_scope_that_fits_conservative_budget():
    counts = {"a": 1319, "b": 300, "c": 1319, "d": 651, "e": 351}
    tasks = tuple(counts)
    assert matrix_generation_count(tasks, conditions=4, limit=None, full_task_counts=counts) == 15760
    assert matrix_generation_count(tasks, conditions=4, limit=128, full_task_counts=counts) == 2560

    assert select_evaluation_limit(
        sentinel_elapsed_seconds=640.0,
        sentinel_generations=640,
        remaining_seconds=4000.0,
        closure_reserve_seconds=0.0,
        tasks=tasks,
        conditions=4,
        candidate_limits=(1024, 512, 256, 128, 32),
        full_task_counts=counts,
        safety_factor=1.25,
    ) == 128
    assert select_evaluation_limit(
        sentinel_elapsed_seconds=64.0,
        sentinel_generations=640,
        remaining_seconds=4000.0,
        closure_reserve_seconds=0.0,
        tasks=tasks,
        conditions=4,
        candidate_limits=(1024, 512, 256, 128, 32),
        full_task_counts=counts,
        safety_factor=1.25,
    ) is None


def _campaign(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps({"output_root": str(tmp_path / "campaign")})
    )
    return Campaign(
        repo=repo,
        protocol_path=protocol_path,
        source_revision="a" * 40,
        deadline_seconds=3600,
        verify_source=False,
    )


def test_campaign_run_records_reserve_timeout(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    monkeypatch.setattr(
        "scripts.run_opsd_h200_autoresearch._run_process",
        lambda *args, **kwargs: ProcessOutcome(
            returncode=124,
            error_kind="timeout",
            error="reserved evaluation boundary reached",
        ),
    )

    with pytest.raises(CampaignCommandTimeout):
        campaign.run("training", ["train"], timeout=60)

    phase = campaign.state["phases"][-1]
    assert phase["name"] == "training"
    assert phase["status"] == "timed_out"
    assert phase["error_kind"] == "timeout"


def test_campaign_parallel_retry_localizes_launch_error(tmp_path, monkeypatch):
    campaign = _campaign(tmp_path)
    calls = {"a": 0, "b": 0}

    def fake_run(command, **kwargs):
        del kwargs
        name = command[0]
        calls[name] += 1
        if name == "a" and calls[name] == 1:
            return ProcessOutcome(127, "launch_error", "temporary exec failure")
        return ProcessOutcome(0)

    monkeypatch.setattr("scripts.run_opsd_h200_autoresearch._run_process", fake_run)

    elapsed = campaign.run_parallel(
        {"a": ["a"], "b": ["b"]},
        max_workers=1,
        max_attempts=3,
    )

    assert elapsed >= 0
    assert calls == {"a": 2, "b": 1}


def test_campaign_parallel_never_starts_next_batch_inside_closure_reserve(
    tmp_path, monkeypatch
):
    campaign = _campaign(tmp_path)
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command[0])
        campaign.started -= 3401
        return ProcessOutcome(0)

    monkeypatch.setattr("scripts.run_opsd_h200_autoresearch._run_process", fake_run)

    with pytest.raises(RuntimeError, match="deadline reserve"):
        campaign.run_parallel(
            {"a": ["a"], "b": ["b"]},
            max_workers=1,
            max_attempts=1,
        )

    assert calls == ["a"]
