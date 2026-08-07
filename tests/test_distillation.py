from __future__ import annotations

from dataclasses import replace
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import weakref

import pytest
import torch
import torch.nn.functional as F

from gcw.checkpoint import load_checkpoint
from gcw.config import CacheProcessorConfig, DataConfig, DistillationConfig, ModelConfig, TrainConfig, WandbConfig, load_train_config
from gcw.distill import pipeline as distill_pipeline
from gcw.distill import precompute as distill_precompute
from gcw.distill import trainer as distill_trainer
from gcw.distill.cache import truncate_last_cache_token
from gcw.distill.data import iter_openmath_prompts
from gcw.distill.objective import exact_forward_kl
from gcw.distill.pipeline import DistillationExample, distillation_example_loss
from gcw.distill.prompt import ChatPrompt, format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession, RolloutResult, sample_next_token
from gcw.distill.teacher import (
    parse_native_teacher_trajectory,
    sample_backbone_completion,
    sample_backbone_completions,
    split_teacher_reasoning,
    teacher_answer_logits,
)
from gcw.distill.traces import TeacherTraceStore
from safetensors.torch import save_file
from gcw.distill.schedule import processor_event_positions
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.models.cache_state import CacheState, StaticCacheState
from gcw.data.tokenizers import TransformersTokenizer


class FakeChatTokenizer:
    eot_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize
        assert messages == [{"role": "user", "content": "2+2?"}]
        base = [10, 11, 12]
        if not add_generation_prompt:
            return base
        return base + ([20, 21] if enable_thinking else [30, 31])

    def encode(self, text: str):
        assert text == "<think>\n"
        return [20, 21]


class ThinkingAgnosticTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


def _distillation_train_config(**overrides) -> TrainConfig:
    kwargs = {
        "model": ModelConfig(
            variant="bgt",
            family="qwen3",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(process_every=64, topk=4, hidden_size=16, intermediate_size=32, num_heads=2),
        ),
        "data": DataConfig(kind="openmath_prompts", dataset="prompts.jsonl", tokenizer="hf:Qwen/Qwen3-0.6B"),
        "mode": "cot_distillation",
        "freeze_backbone": True,
        "backbone_checkpoint": "checkpoint",
        "batch_size": 1,
        "max_steps": 1,
        "distillation": DistillationConfig(),
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_distillation_config_accepts_explicit_v0_contract():
    config = _distillation_train_config()

    assert config.distillation is not None
    assert config.distillation.response_process_every == 64
    assert config.distillation.teacher_thinking
    assert not config.distillation.student_thinking


def test_distillation_uses_exact_forward_kl_and_rejects_legacy_clipping():
    assert DistillationConfig().token_clip is None

    with pytest.raises(ValueError, match="signed per-vocabulary KL clipping"):
        DistillationConfig(token_clip=0.05)


def test_native_full_teacher_protocol_is_explicit_and_uses_both_visible_trajectories():
    config = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=2048,
        close_reasoning_at_eos=False,
    )

    assert config.teacher_trajectory_mode == "native_full"
    assert config.visible_trajectory_mode == "both"
    assert config.max_teacher_completion_tokens == 2048



@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"teacher_trajectory_mode": "native_full"}, "close_reasoning_at_eos"),
        ({"visible_trajectory_mode": "both"}, "native_full"),
        (
            {
                "teacher_trajectory_mode": "native_full",
                "visible_trajectory_mode": "teacher",
                "close_reasoning_at_eos": False,
            },
            "student or both",
        ),
        (
            {
                "teacher_trajectory_mode": "native_full",
                "visible_trajectory_mode": "both",
                "max_teacher_completion_tokens": 0,
                "close_reasoning_at_eos": False,
            },
            "completion token",
        ),
    ],
)
def test_native_full_teacher_protocol_rejects_ambiguous_configs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        DistillationConfig(**kwargs)



def test_distillation_optimizer_step_budget_must_be_positive():
    with pytest.raises(ValueError, match="max_optimizer_steps"):
        DistillationConfig(max_optimizer_steps=0)


def test_distillation_heldout_contract_is_unique_nonempty_and_excluded_from_training(
    monkeypatch,
):
    distill = DistillationConfig(
        heldout_source_indices=[1, 3],
        minimum_heldout_examples=1,
    )
    assert distill.heldout_source_indices == (1, 3)
    with pytest.raises(ValueError, match="unique"):
        DistillationConfig(
            heldout_source_indices=[1, 1],
            minimum_heldout_examples=1,
        )
    with pytest.raises(ValueError, match="minimum_heldout_examples"):
        DistillationConfig(heldout_source_indices=[1])

    config = _distillation_train_config(distillation=distill)
    records = tuple(
        distill_trainer.PromptRecord(source_index=index, prompt=f"p-{index}")
        for index in range(5)
    )
    monkeypatch.setattr(distill_trainer, "iter_openmath_prompts", lambda *a, **k: iter(records))
    training = list(distill_trainer._epoch_training_items(config, None))
    assert [record.source_index for record, _ in training] == [0, 2, 4]


def test_phase_timings_accumulate_repeated_intervals(monkeypatch):
    ticks = iter((1.0, 1.25, 2.0, 2.75))
    monkeypatch.setattr(distill_trainer.time, "monotonic", lambda: next(ticks))
    timings = distill_trainer._PhaseTimings()

    with timings.measure("teacher_and_student_generation"):
        pass
    with timings.measure("teacher_and_student_generation"):
        pass

    assert timings.values == {"teacher_and_student_generation": 1.0}


def test_training_items_prebuild_one_ordered_execution_microbatch(monkeypatch):
    records = (
        distill_trainer.PromptRecord(source_index=0, prompt="first"),
        distill_trainer.PromptRecord(source_index=1, prompt="second"),
    )
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(3,)),
        teacher_reasoning_ids=(4,),
        response_ids=(5,),
    )
    invalid = distill_trainer.InvalidDistillationExample("cached failure")
    calls = []
    monkeypatch.setattr(
        distill_trainer,
        "_epoch_training_items",
        lambda *args, **kwargs: ((record, None) for record in records),
    )

    def fake_build(*args, **kwargs):
        calls.append(args[2])
        return (example, invalid)

    monkeypatch.setattr(distill_trainer, "_build_cached_examples_batch", fake_build)
    monkeypatch.setattr(
        distill_trainer,
        "_autocast_context",
        lambda *args, **kwargs: __import__("contextlib").nullcontext(),
    )
    config = __import__("types").SimpleNamespace(
        distillation=DistillationConfig(
            gradient_accumulation_steps=2,
            execution_microbatch_size=2,
            require_precomputed_teacher_traces=True,
        ),
        seed=24,
    )

    items = list(
        distill_trainer._epoch_training_items_with_prebuild(
            object(),
            object(),
            config,
            None,
            object(),
            (),
            object(),
            torch.device("cpu"),
            remaining_before_update=lambda: 2,
        )
    )

    assert calls == [records]
    assert items == [
        (records[0], None, example),
        (records[1], None, invalid),
    ]


def test_training_items_never_prebuild_across_optimizer_boundary_after_invalid(
    monkeypatch,
):
    records = tuple(
        distill_trainer.PromptRecord(source_index=index, prompt=str(index))
        for index in range(4)
    )
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(3,)),
        teacher_reasoning_ids=(4,),
        response_ids=(5,),
    )
    invalid = distill_trainer.InvalidDistillationExample("cached failure")
    calls = []
    outcomes = iter(((invalid, example), (example,), (example,)))
    monkeypatch.setattr(
        distill_trainer,
        "_epoch_training_items",
        lambda *args, **kwargs: ((record, None) for record in records),
    )

    def fake_build(*args, **kwargs):
        calls.append(args[2])
        return next(outcomes)

    monkeypatch.setattr(distill_trainer, "_build_cached_examples_batch", fake_build)
    monkeypatch.setattr(
        distill_trainer,
        "_autocast_context",
        lambda *args, **kwargs: __import__("contextlib").nullcontext(),
    )
    config = __import__("types").SimpleNamespace(
        distillation=DistillationConfig(
            gradient_accumulation_steps=2,
            execution_microbatch_size=2,
            require_precomputed_teacher_traces=True,
        ),
        seed=24,
    )
    accumulated = 0
    items = distill_trainer._epoch_training_items_with_prebuild(
        object(),
        object(),
        config,
        None,
        object(),
        (),
        object(),
        torch.device("cpu"),
        remaining_before_update=lambda: 2 - accumulated,
    )

    first = next(items)
    assert isinstance(first[2], distill_trainer.InvalidDistillationExample)
    second = next(items)
    accumulated += 1
    assert second[2] == example
    third = next(items)
    accumulated = 0  # this row triggers the optimizer update before another next()
    assert third[2] == example
    fourth = next(items)
    assert fourth[2] == example
    with pytest.raises(StopIteration):
        next(items)

    assert calls == [(records[0], records[1]), (records[2],), (records[3],)]


def test_distillation_execution_microbatch_must_divide_effective_batch():
    with pytest.raises(ValueError, match="execution_microbatch_size"):
        DistillationConfig(
            gradient_accumulation_steps=16,
            execution_microbatch_size=3,
        )


def test_distillation_execution_microbatch_requires_precomputed_teacher_traces():
    with pytest.raises(ValueError, match="precomputed teacher traces"):
        DistillationConfig(
            gradient_accumulation_steps=2,
            execution_microbatch_size=2,
        )


def test_distillation_cache_displacement_audit_cadence_must_be_positive():
    with pytest.raises(ValueError, match="cache displacement audit cadence"):
        DistillationConfig(cache_displacement_audit_every_optimizer_steps=0)


def test_cache_displacement_audit_schedule_covers_first_periodic_and_final_examples():
    distill = DistillationConfig(
        record_cache_displacements=True,
        cache_displacement_audit_every_optimizer_steps=16,
        gradient_accumulation_steps=4,
        max_optimizer_steps=20,
    )

    audited = {
        (step, accumulated)
        for step in range(20)
        for accumulated in range(4)
        if distill_trainer._should_record_cache_displacements(
            distill,
            valid_examples=step * 4 + accumulated,
            accumulated_examples=accumulated,
            optimizer_steps=step,
        )
    }

    assert audited == {(0, 0), (15, 0), (19, 3)}


def test_cache_displacement_audit_schedule_preserves_legacy_every_example_default():
    distill = DistillationConfig(record_cache_displacements=True)

    assert distill_trainer._should_record_cache_displacements(
        distill,
        valid_examples=7,
        accumulated_examples=0,
        optimizer_steps=7,
    )


def test_distillation_repeated_epochs_require_explicit_overfit_diagnostic():
    with pytest.raises(ValueError, match="overfit diagnostic"):
        DistillationConfig(epochs=2)

    with pytest.raises(ValueError, match="expected_locked_examples"):
        DistillationConfig(epochs=2, overfit_diagnostic=True)

    diagnostic = DistillationConfig(
        epochs=2,
        overfit_diagnostic=True,
        expected_locked_examples=1,
    )

    assert diagnostic.epochs == 2
    assert diagnostic.overfit_diagnostic
    assert diagnostic.expected_locked_examples == 1


@pytest.mark.parametrize(
    ("path", "take_rows", "epochs", "expected_locked"),
    [
        ("configs/train/opsd/qwen3_06b_opsd_overfit16_smoke.yaml", 1, 2, 1),
        ("configs/train/opsd/qwen3_06b_opsd_overfit16x20.yaml", 21, 20, 16),
        ("configs/train/opsd/qwen3_06b_opsd_overfit16x40.yaml", 21, 40, 16),
    ],
)
def test_opsd_overfit_configs_are_explicit_diagnostics(path, take_rows, epochs, expected_locked):
    config = load_train_config(path)

    assert config.data.sft_take_rows == take_rows
    assert config.distillation is not None
    assert config.distillation.epochs == epochs
    assert config.distillation.overfit_diagnostic
    assert config.distillation.expected_locked_examples == expected_locked


@pytest.mark.parametrize(
    ("path", "take_rows", "accumulation", "optimizer_steps"),
    [
        (
            "configs/train/opsd/qwen3_06b_opsd_native_full_smoke_v2.yaml",
            64,
            1,
            1,
        ),
        (
            "configs/train/opsd/qwen3_06b_opsd_native_full_steps1000_bs16_v2.yaml",
            128_000,
            16,
            1_000,
        ),
        (
            "configs/train/opsd/qwen3_06b_opsd_native_full_steps10000_bs16_v2.yaml",
            1_280_000,
            16,
            10_000,
        ),
    ],
)
def test_native_full_campaign_configs_lock_budget_and_online_tracking(
    path, take_rows, accumulation, optimizer_steps
):
    config = load_train_config(path)

    assert config.data.sft_take_rows == take_rows
    assert config.learning_rate == 1.0e-4
    assert config.distillation is not None
    assert config.distillation.teacher_trajectory_mode == "native_full"
    assert config.distillation.visible_trajectory_mode == "both"
    assert config.distillation.gradient_accumulation_steps == accumulation
    assert config.distillation.max_optimizer_steps == optimizer_steps
    assert config.wandb.enabled
    assert config.wandb.mode == "online"
    assert f"steps{optimizer_steps}" in config.wandb.name
    assert not __import__("pathlib").Path(config.output_dir).exists()



def test_conversion_diagnostic_wrapper_is_fail_closed():
    wrapper = (
        Path(__file__).parents[1] / "scripts/slurm/diagnose_qwen3_base_conversion.sbatch"
    ).read_text()

    assert "#SBATCH --gres=gpu:1" in wrapper
    assert "#SBATCH --partition=agent-dev-15,agent-dev" in wrapper
    assert "#SBATCH --nodes=1" in wrapper
    assert "#SBATCH --ntasks=1" in wrapper
    assert "#SBATCH --cpus-per-task=4" in wrapper
    assert "#SBATCH --mem=32G" in wrapper
    assert "git status --porcelain" in wrapper
    assert "git ls-files --error-unmatch" in wrapper
    assert "PARENT_PARITY_SHA256" in wrapper
    assert "1aca29e6db3099d652d8635d57f8f79a9acccfbec915a378c0477e7104b86216" in wrapper
    assert "[[ ! -e \"$OUT\" ]]" in wrapper
    assert "scripts/diagnose_hf_conversion.py" in wrapper
    assert "--validate-only" in wrapper
    assert "torch.cuda.device_count()" in wrapper
    assert "nvidia-smi --query-gpu" not in wrapper
    assert "sbatch --wrap" not in wrapper


@pytest.mark.parametrize(
    ("filename", "steps", "effective_batch", "output_name"),
    (
        ("qwen3_06b_opsd_native_base_pragmatic_canary_v1.yaml", 1, 1, "canary-steps1-bs1"),
        ("qwen3_06b_opsd_native_base_pragmatic_steps1000_bs16_v1.yaml", 1000, 16, "steps1000-bs16"),
    ),
)
def test_pragmatic_base_configs_are_exact_waived_protocol(
    filename, steps, effective_batch, output_name
):
    root = Path(__file__).parents[1]
    validator = importlib.import_module("scripts.validate_opsd_pragmatic_waiver")
    assert validator.TOKENIZER == Path(
        "/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1"
    )
    config = load_train_config(root / "configs/train/opsd" / filename)

    assert config.freeze_backbone is True
    assert config.distillation is not None
    assert config.distillation.teacher_trajectory_mode == "native_full"
    assert config.distillation.visible_trajectory_mode == "both"
    assert config.distillation.max_optimizer_steps == steps
    assert config.distillation.gradient_accumulation_steps == effective_batch
    assert config.metadata == {
        "campaign": "opsd-native-base-pragmatic-v1",
        "scientific_scope": "pragmatic_same_gcw_backbone_sensitivity_after_failed_hf_parity",
        "required_conversion_parity": "/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1/parity.json",
        "required_conversion_diagnostic": "/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1/diagnostics/layerwise-distribution-v1-fp64.json",
        "conversion_parity_waiver": "configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v2.json",
        "comparison_reference": "configs/train/bottleneck_llm/qwen3_06b_bgt_from_sft_frozen.yaml",
        "backbone_label": "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF",
        "distillation_effective_batch_examples": effective_batch,
        "requested_optimizer_steps": steps,
        **(
            {
                "promotion_evidence": (
                    "/mnt/jfs-01/shared/gcw/experiments/"
                    "qwen3_06b_opsd_native_base_pragmatic_v1/"
                    "canary-steps1-bs1/SUCCESS.json"
                )
            }
            if steps == 1000
            else {}
        ),
    }
    assert config.output_dir.endswith(f"/qwen3_06b_opsd_native_base_pragmatic_v1/{output_name}")


def test_pragmatic_parity_waiver_is_closed_and_tamper_rejected():
    root = Path(__file__).parents[1]
    path = root / "configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v2.json"
    document = json.loads(path.read_text())
    validator = importlib.import_module("scripts.validate_opsd_pragmatic_waiver")

    validator.validate_waiver_document(document)
    assert document["constraints"]["training_gpu_families"] == ["H100", "H200"]
    assert validator.validate_training_gpu(document, "NVIDIA H100 80GB HBM3") == "H100"
    assert validator.validate_training_gpu(document, "NVIDIA H200") == "H200"
    with pytest.raises(ValueError, match="GPU"):
        validator.validate_training_gpu(document, "NVIDIA A100-SXM4-80GB")
    with pytest.raises(ValueError, match="GPU"):
        validator.validate_training_gpu(document, "counterfeit-H200-device")
    with pytest.raises(ValueError, match="GPU"):
        validator.validate_training_gpu(document, "NVIDIA GH200 480GB")
    assert document["constraints"]["training_gpu_model_topology_must_match_canary"] is True
    tampered = copy.deepcopy(document)
    tampered["constraints"]["max_optimizer_steps"] = 10_000
    with pytest.raises(ValueError, match="waiver document"):
        validator.validate_waiver_document(tampered)


def test_native_train_wrapper_preserves_parity_gate_and_has_strict_pragmatic_branch():
    wrapper = (
        Path(__file__).parents[1] / "scripts/slurm/opsd_native_train.sbatch"
    ).read_text()

    assert "REQUIRES_CONVERSION_PARITY" in wrapper
    assert "validate_parity_evidence" in wrapper
    assert "REQUIRES_PARITY_WAIVER" in wrapper
    assert "validate_opsd_pragmatic_waiver.py" in wrapper
    assert "PRAGMATIC_WAIVER_EVIDENCE" in wrapper
    assert "pragmatic_same_gcw_backbone_sensitivity_after_failed_hf_parity" in wrapper


def test_pragmatic_full_matrix_preserves_waiver_label_and_exact_factorial():
    root = Path(__file__).parents[1]
    wrapper = (root / "scripts/slurm/opsd_full_matrix.sbatch").read_text()
    publisher = (root / "scripts/log_opsd_matrix_wandb.py").read_text()

    assert "EXPECTED_PRAGMATIC_WAIVER" in wrapper
    assert "PRAGMATIC_WAIVER.json" in wrapper
    assert "pragmatic_parity_waiver" in wrapper
    assert "parity_gate_remains_failed" in wrapper
    for condition in (
        "cot-off__cp-off",
        "cot-off__cp-on",
        "cot-on__cp-off",
        "cot-on__cp-on",
    ):
        assert condition in wrapper
    assert "WANDB_PRAGMATIC_WAIVER" in publisher
    assert "pragmatic-parity-waiver" in publisher
    assert "parity_gate_remains_failed" in publisher


def test_pragmatic_parity_validator_accepts_only_schema3_numeric_failure():
    validator = importlib.import_module("scripts.validate_opsd_pragmatic_waiver")
    document = {
        "passed": False,
        "policy_id": "qwen3-06b-bf16-parameters-fp32-comparison-v2",
        "mean_abs_error": 0.03596403822302818,
        "mean_abs_threshold": 0.01,
    }

    validator.validate_parity_failure_document(document)
    for mutation in (
        {**document, "passed": "false"},
        {**document, "mean_abs_error": "0.03596403822302818"},
        {**document, "mean_abs_error": 0.01},
        {
            "passed": False,
            "policy_id": document["policy_id"],
            "mean_abs_error": document["mean_abs_error"],
            "thresholds": {"mean_abs_error": 0.01},
        },
    ):
        with pytest.raises(ValueError, match="approved numerical failure"):
            validator.validate_parity_failure_document(mutation)


def test_pragmatic_runtime_gpu_evidence_is_detected_from_cuda(monkeypatch):
    import sys
    from types import SimpleNamespace

    root = Path(__file__).parents[1]
    document = json.loads(
        (
            root
            / "configs/protocols/qwen3_06b_opsd_pragmatic_parity_waiver_v2.json"
        ).read_text()
    )
    fake_cuda = SimpleNamespace(
        device_count=lambda: 1,
        get_device_properties=lambda _: SimpleNamespace(
            name="NVIDIA H200", total_memory=143_771 * 1024 * 1024
        ),
        get_device_capability=lambda _: (9, 0),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    validator = importlib.import_module("scripts.validate_opsd_pragmatic_waiver")

    assert validator.detect_runtime_gpu(document) == {
        "gpu_name": "NVIDIA H200",
        "gpu_family": "H200",
        "gpu_total_memory_bytes": 143_771 * 1024 * 1024,
        "gpu_compute_capability": "9.0",
    }


def test_pragmatic_protocol_binds_exact_configs_and_promotion_evidence():
    root = Path(__file__).parents[1]
    validator = importlib.import_module("scripts.validate_opsd_pragmatic_waiver")
    wrapper = (root / "scripts/slurm/opsd_native_train.sbatch").read_text()

    for filename, expected in validator.CONFIGS.items():
        path = root / "configs/train/opsd" / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"]
    assert "qwen3_06b_opsd_pragmatic_parity_waiver_v2.json" in wrapper
    validator_source = (
        root / "scripts/validate_opsd_pragmatic_waiver.py"
    ).read_text()
    assert "--gpu-name" not in wrapper
    assert "detect_runtime_gpu(waiver)" in validator_source
    assert 'success.get("gpu_name") != os.environ["GPU_NAME"]' in wrapper
    assert 'waiver.get("gpu_name") != os.environ["GPU_NAME"]' in wrapper
    assert '"gpu_family": os.environ["GPU_FAMILY"]' in wrapper
    assert '"gpu_total_memory_bytes": int(os.environ["GPU_TOTAL_MEMORY_BYTES"])' in wrapper
    assert '"gpu_compute_capability": os.environ["GPU_COMPUTE_CAPABILITY"]' in wrapper
    assert 'success_path != Path(os.environ["CONFIG_PROMOTION_EVIDENCE"])' in wrapper
    assert "CANARY_CONFIG_SHA256" in wrapper
    assert 'success.get("pragmatic_parity_waiver") is not True' in wrapper
    assert 'success.get("parity_gate_remains_failed") is not True' in wrapper
    assert 'waiver_entry = files.get("PRAGMATIC_WAIVER.json")' in wrapper
    assert "canary config digest mismatch" in wrapper


def test_pragmatic_labels_tokenizer_prepublication_and_safe_claim_order():
    root = Path(__file__).parents[1]
    train = (root / "scripts/slurm/opsd_native_train.sbatch").read_text()
    matrix = (root / "scripts/slurm/opsd_full_matrix.sbatch").read_text()
    publisher = (root / "scripts/log_opsd_matrix_wandb.py").read_text()
    label = "converted_GCW_Qwen3_0.6B_base_not_numerically_identical_to_HF"

    assert label in train
    assert label in matrix
    assert label in publisher
    assert "/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1" in matrix
    assert 'state.get("tokenizer_source") != tokenizer' in matrix
    assert 'success.get("pragmatic_parity_waiver") is not expected_waiver' in matrix
    assert 'success.get("parity_gate_remains_failed") is not expected_waiver' in matrix
    assert "canonical_matrix_validated_before_publication=true" in matrix
    assert matrix.index("canonical_matrix_validated_before_publication=true") < matrix.index(
        "scripts/log_opsd_matrix_wandb.py"
    )
    assert train.index('[[ ! -e "$OUTPUT_DIR"') > train.index("wandb_authenticated=true")
    assert "#SBATCH --partition=agent-xlong-15" in train
    assert "#SBATCH --partition=agent-xlong-15" in matrix


def test_pragmatic_submitter_hard_wires_afterok_chain_and_identities():
    submitter = (
        Path(__file__).parents[1] / "scripts/submit_opsd_pragmatic_chain.py"
    ).read_text()

    assert "afterok:{canary_job_id}" in submitter
    assert "afterok:{train_job_id}" in submitter
    assert "validate_static(" in submitter
    assert "validate_all(" not in submitter
    assert "PROMOTION_EVIDENCE" in submitter
    assert '"EXPECTED_PRAGMATIC_WAIVER": "true"' in submitter
    assert "agent-dev-15" in submitter
    assert "agent-xlong-15" in submitter
    assert "scancel" in submitter
    assert "submission-receipt" in submitter
    assert "os.chmod(path, 0o444)" in submitter


def test_native_train_wrapper_rejects_external_config(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=repo, check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    external_config = tmp_path / "external.yaml"
    external_config.write_text("untrusted: true\n")
    wrapper = (
        Path(__file__).resolve().parents[1]
        / "scripts/slurm/opsd_native_train.sbatch"
    )
    result = subprocess.run(
        ["bash", str(wrapper)],
        cwd=repo,
        env={
            **os.environ,
            "REPO": str(repo),
            "CONFIG": str(external_config),
            "SOURCE_REVISION": revision,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "config must resolve inside pinned source checkout" in result.stderr


def test_native_base_smoke_is_distinct_and_parity_gated():
    config = load_train_config(
        "configs/train/opsd/qwen3_06b_opsd_native_base_smoke_v1.yaml"
    )

    assert config.backbone_checkpoint == (
        "/mnt/jfs-01/shared/gcw/checkpoints/base_backbones/qwen3_06b/tf-v1"
    )
    assert config.data.tokenizer == "hf:/mnt/jfs-01/shared/gcw/hf/Qwen3-0.6B-base-v1"
    assert config.output_dir.endswith("qwen3_06b_opsd_native_base_v1/smoke-steps1-bs1")
    assert config.distillation is not None
    assert config.distillation.max_optimizer_steps == 1
    assert config.distillation.teacher_trajectory_mode == "native_full"
    assert config.distillation.visible_trajectory_mode == "both"
    assert config.metadata["scientific_scope"] == (
        "distinct_base_backbone_fallback_after_sft_zero_yield"
    )
    assert config.metadata["required_conversion_parity"].endswith("/tf-v1/parity.json")

    one_k = load_train_config(
        "configs/train/opsd/qwen3_06b_opsd_native_base_steps1000_bs16_v1.yaml"
    )
    assert one_k.backbone_checkpoint == config.backbone_checkpoint
    assert one_k.data.tokenizer == config.data.tokenizer
    assert one_k.distillation is not None
    assert one_k.distillation.max_optimizer_steps == 1000
    assert one_k.distillation.gradient_accumulation_steps == 16
    assert one_k.output_dir.endswith("qwen3_06b_opsd_native_base_v1/steps1000-bs16")
    assert one_k.metadata["required_conversion_parity"] == (
        config.metadata["required_conversion_parity"]
    )



def test_opsd_exact_kl_pilot_config_is_seed_locked_and_fresh():
    config = load_train_config(
        "configs/train/opsd/qwen3_06b_opsd_pilot128_exact_kl_v2.yaml"
    )

    assert config.data.sft_skip_rows == 0
    assert config.data.sft_take_rows == 128
    assert config.seed == 24
    assert config.learning_rate == 1.0e-4
    assert config.distillation is not None
    assert config.distillation.epochs == 1
    assert not config.distillation.overfit_diagnostic
    assert config.distillation.record_cache_displacements
    assert config.distillation.token_clip is None
    assert config.output_dir.endswith("/pilot-128-exact-kl-v2")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"freeze_backbone": False}, "frozen backbone"),
        ({"model": ModelConfig()}, "BGT"),
        ({"batch_size": 2}, "batch_size=1"),
        ({"max_steps": 2}, "max_steps=1"),
    ],
)
def test_distillation_config_rejects_unsupported_v0(overrides, message):
    with pytest.raises(ValueError, match=message):
        _distillation_train_config(**overrides)


def test_processor_event_positions_are_response_relative():
    assert processor_event_positions(max_response_tokens=0, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=63, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=64, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=65, process_every=64) == (0, 64)
    assert processor_event_positions(max_response_tokens=130, process_every=64) == (0, 64, 128)


def test_format_chat_prompt_splits_assistant_trigger_without_retokenising():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=False)

    assert prompt.base_ids == (10, 11, 12)
    assert prompt.trigger_ids == (30, 31)
    assert prompt.full_ids == (10, 11, 12, 30, 31)


def test_format_chat_prompt_keeps_thinking_mode_in_trigger():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=True)

    assert prompt.trigger_ids == (20, 21)


def test_native_teacher_trajectory_compacts_only_visible_tokens():
    trajectory = parse_native_teacher_trajectory(
        generated_ids=(40, 41, 22, 23, 50, 51, 0),
        teacher_trigger_ids=(20, 21),
        open_ids=(20, 21),
        close_ids=(22, 23),
        eos_token_id=0,
    )

    assert trajectory.reasoning_ids == (40, 41, 22, 23)
    assert trajectory.visible_response_ids == (50, 51, 0)
    assert trajectory.full_completion_ids == (40, 41, 22, 23, 50, 51, 0)



def test_native_teacher_trajectory_rejects_budget_exhaustion_without_eot():
    with pytest.raises(ValueError, match="end with EOT"):
        parse_native_teacher_trajectory(
            generated_ids=(40, 41, 22, 23, 50, 51),
            teacher_trigger_ids=(20, 21),
            open_ids=(20, 21),
            close_ids=(22, 23),
            eos_token_id=0,
        )



@pytest.mark.parametrize(
    ("generated_ids", "teacher_trigger_ids", "message"),
    [
        ((40, 22, 23, 50, 0), (30, 31), "trigger"),
        ((40, 22, 23, 50, 0), (20, 21, 20, 21), "exactly one thinking-open"),
        ((20, 21, 40, 22, 23, 50, 0), (20, 21), "repeats"),
        ((40, 22, 23, 50, 22, 23, 0), (20, 21), "exactly one"),
        ((22, 23, 50, 0), (20, 21), "non-empty reasoning"),
        ((40, 22, 23, 0), (20, 21), "visible answer"),
        ((40, 0, 22, 23, 50, 0), (20, 21), "embedded EOT"),
    ],
)
def test_native_teacher_trajectory_rejects_malformed_thinking_regions(
    generated_ids, teacher_trigger_ids, message
):
    with pytest.raises(ValueError, match=message):
        parse_native_teacher_trajectory(
            generated_ids=generated_ids,
            teacher_trigger_ids=teacher_trigger_ids,
            open_ids=(20, 21),
            close_ids=(22, 23),
            eos_token_id=0,
        )



def test_teacher_prompt_does_not_duplicate_template_thinking_open():
    prompt = format_teacher_prompt(FakeChatTokenizer(), "2+2?", reasoning_open_text="<think>\n")

    assert prompt.trigger_ids == (20, 21)


def test_teacher_prompt_appends_thinking_open_when_template_ignores_flag():
    prompt = format_teacher_prompt(ThinkingAgnosticTokenizer(), "2+2?", reasoning_open_text="<think>\n")

    assert prompt.trigger_ids == (30, 31, 20, 21)


def test_forward_kl_matches_exact_pytorch_definition_and_is_non_negative():
    student = torch.tensor(
        [[[12.19104, 1.0137914, 12.252707, 10.12801, 8.895775, -4.5896535, -2.2890944, -3.6223664]]],
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[[6.3993907, -4.970331, 9.074863, -3.0142415, 8.074074, 9.650991, -2.1125073, -0.4159319]]],
        requires_grad=True,
    )
    mask = torch.tensor([[True]])
    temperature = 1.1

    loss = exact_forward_kl(
        student,
        teacher,
        mask,
        temperature=temperature,
        token_clip=None,
    )

    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher.detach() / temperature, dim=-1)
    per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    expected = per_vocab.sum(dim=-1)[mask].mean()
    legacy_surrogate = per_vocab.clamp(max=0.05).sum(dim=-1)[mask].mean()
    torch.testing.assert_close(loss, expected)
    assert float(loss.detach()) >= 0.0
    assert float(legacy_surrogate.detach()) < 0.0

    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_forward_kl_rejects_signed_contribution_clipping():
    logits = torch.zeros(1, 1, 3)
    mask = torch.ones(1, 1, dtype=torch.bool)

    with pytest.raises(ValueError, match="signed per-vocabulary KL clipping"):
        exact_forward_kl(logits, logits, mask, token_clip=0.05)


def test_exact_forward_kl_rejects_empty_answer_mask():
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="at least one answer token"):
        exact_forward_kl(logits, logits, torch.zeros(1, 2, dtype=torch.bool))


def test_truncate_last_cache_token_preserves_earlier_values_and_gradients():
    keys = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2).requires_grad_()
    values = (keys.detach() + 100).requires_grad_()
    scores = torch.arange(4, dtype=torch.float32).reshape(1, 4).requires_grad_()
    cache = CacheState([keys], [values], scores)

    truncated = truncate_last_cache_token(cache)

    assert truncated.seq_len == 3
    torch.testing.assert_close(truncated.keys[0], keys[:, :, :-1])
    torch.testing.assert_close(truncated.values[0], values[:, :, :-1])
    torch.testing.assert_close(truncated.scores, scores[:, :-1])
    truncated.keys[0].sum().backward()
    assert keys.grad is not None
    assert torch.count_nonzero(keys.grad[:, :, -1]) == 0


def test_truncate_last_cache_token_rejects_empty_cache():
    cache = CacheState([], [], torch.empty(1, 0))
    with pytest.raises(ValueError, match="empty cache"):
        truncate_last_cache_token(cache)



def _tiny_bgt():
    model = build_model(
        ModelConfig(
            variant="bgt",
            family="qwen3",
            sequence_len=256,
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(
                process_every=64,
                topk=4,
                hidden_size=16,
                intermediate_size=32,
                num_heads=1,
                num_kv_heads=1,
            ),
        )
    )
    model.init_weights()
    return model


class RecordingZeroProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, bool]] = []

    def forward(
        self,
        cache: CacheState | StaticCacheState,
        *,
        select_all: bool = False,
    ) -> CacheState | StaticCacheState:
        self.calls.append((cache.seq_len, select_all))
        if isinstance(cache, StaticCacheState):
            for key, value in zip(cache.keys, cache.values, strict=True):
                key.zero_()
                value.zero_()
            return cache
        return CacheState(
            [torch.zeros_like(key) for key in cache.keys],
            [torch.zeros_like(value) for value in cache.values],
            cache.scores,
        )


def test_replay_processes_boundary_before_token_one_and_every_64_response_tokens():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor
    prompt = ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4, 5))
    response_ids = tuple([6] * 65)

    with torch.no_grad():
        _, unprocessed_cache = model.backbone.forward_chunk(torch.tensor([prompt.base_ids]), 0, logits=False)
        unprocessed_logits, _ = model.backbone.forward_chunk(
            torch.tensor([prompt.trigger_ids]),
            len(prompt.base_ids),
            unprocessed_cache,
        )
        replay = BGTDecodeSession(model, process_every=64).replay(prompt, response_ids)

    assert processor.calls == [(3, False), (69, False)]
    assert replay.event_cache_lengths == (3, 69)
    assert replay.logits.shape == (1, 65, 64)
    assert not torch.allclose(replay.logits[:, 0], unprocessed_logits[:, -1])


def test_replay_select_all_is_explicit_and_applies_to_every_event():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor

    BGTDecodeSession(model, process_every=2, select_all=True).replay(
        ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)),
        (4, 5, 6),
    )

    assert processor.calls == [(2, True), (5, True)]


def test_replay_backpropagates_only_into_bgt_when_backbone_is_frozen():
    model = _tiny_bgt()
    model.backbone.requires_grad_(False)
    session = BGTDecodeSession(model, process_every=2)

    replay = session.replay(ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,)), (5, 6, 7))
    replay.logits.square().mean().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    processor_grads = [parameter.grad for parameter in model.cache_processor.parameters()]
    assert processor_grads
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in processor_grads)



def test_sample_next_token_top_k_one_is_argmax():
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    token = sample_next_token(logits, temperature=1.1, top_p=1.0, top_k=1)
    assert token.item() == 1


def test_rollout_and_replay_share_response_relative_event_schedule():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor
    session = BGTDecodeSession(model, process_every=2)
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    generator = torch.Generator().manual_seed(7)

    rollout = session.rollout(
        prompt,
        max_response_tokens=5,
        eos_token_id=None,
        temperature=1.1,
        top_p=1.0,
        top_k=0,
        generator=generator,
    )
    rollout_calls = tuple(processor.calls)
    processor.calls.clear()
    replay = session.replay(prompt, rollout.response_ids)

    assert len(rollout.response_ids) == 5
    assert rollout.event_cache_lengths == (2, 5, 7)
    assert replay.event_cache_lengths == rollout.event_cache_lengths
    assert tuple(processor.calls) == rollout_calls



class PositionLogitBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs: list[torch.Tensor] = []

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.inputs.append(input_ids.detach().clone())
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        return positions[None, :, None].expand(input_ids.shape[0], -1, self.vocab_size)


def test_teacher_reasoning_is_context_but_not_a_kl_position():
    backbone = PositionLogitBackbone()
    teacher_context = (10, 11, 12, 90, 91, 92)
    response = (20, 21)

    logits = teacher_answer_logits(backbone, teacher_context, response)

    assert backbone.inputs[0].tolist() == [list(teacher_context + response)]
    assert logits.shape == (1, 2, backbone.vocab_size)
    torch.testing.assert_close(logits[:, 0], torch.full((1, backbone.vocab_size), 5.0))
    torch.testing.assert_close(logits[:, 1], torch.full((1, backbone.vocab_size), 6.0))
    assert not logits.requires_grad


def test_split_teacher_reasoning_stops_after_close_separator():
    generated = (40, 41, 50, 51, 52, 60, 61)
    assert split_teacher_reasoning(generated, close_ids=(50, 51, 52)) == (40, 41, 50, 51, 52)


def test_split_teacher_reasoning_rejects_missing_close_separator():
    with pytest.raises(ValueError, match="close separator"):
        split_teacher_reasoning((1, 2, 3), close_ids=(8, 9))


def test_teacher_reasoning_closes_terminal_eot_for_direct_answer_checkpoint():
    reasoning = split_teacher_reasoning(
        (1, 2, 3, 0),
        close_ids=(8, 9),
        eos_token_id=0,
        close_at_eos=True,
    )

    assert reasoning == (1, 2, 3, 8, 9)


def test_openmath_prompt_reader_ignores_answer_content(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"prompt": "p0", "answer": "secret-a"}\n'
        '{"problem": "p1", "solution": "secret-b"}\n'
        '{"question": "p2", "output": "secret-c"}\n',
        encoding="utf-8",
    )

    records = list(iter_openmath_prompts(path))

    assert [record.prompt for record in records] == ["p0", "p1", "p2"]
    assert [record.source_index for record in records] == [0, 1, 2]
    assert all("secret" not in record.prompt for record in records)



class FakeHFTokenizerBackend:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


def test_transformers_tokenizer_exposes_thinking_chat_template():
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = FakeHFTokenizerBackend()

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert ids == [1, 2, 3]
    assert tokenizer.enc.calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }



def test_distillation_both_visible_trajectories_average_shared_prefix_losses():
    model = _tiny_bgt()
    model.backbone.requires_grad_(False)
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,)),
        teacher_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(10, 11, 12),
        response_ids=(5, 6, 7),
        teacher_response_ids=(13, 14),
    )
    session = BGTDecodeSession(model, process_every=2)

    student = distillation_example_loss(
        model,
        example,
        session,
        temperature=1.1,
        token_clip=None,
        visible_trajectory_mode="student",
    )
    teacher = distillation_example_loss(
        model,
        example,
        session,
        temperature=1.1,
        token_clip=None,
        visible_trajectory_mode="teacher",
    )
    with pytest.raises(RuntimeError, match="serial"):
        distillation_example_loss(
            model,
            example,
            session,
            temperature=1.1,
            token_clip=None,
            visible_trajectory_mode="both",
        )
    with torch.no_grad():
        both = distillation_example_loss(
            model,
            example,
            session,
            temperature=1.1,
            token_clip=None,
            visible_trajectory_mode="both",
        )

    torch.testing.assert_close(both.loss, (student.loss + teacher.loss) / 2)
    assert both.student_trajectory_tokens == 3
    assert both.teacher_trajectory_tokens == 2
    assert both.answer_tokens == 5



def test_distillation_both_no_grad_releases_first_trajectory_logits(monkeypatch):
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(40, 10, 11),
        response_ids=(52, 53, 62),
        teacher_response_ids=(50, 62),
    )
    state = {
        "student_calls": 0,
        "teacher_calls": 0,
        "first_student": None,
        "first_teacher": None,
    }

    class Replay:
        def __init__(self, logits):
            self.logits = logits
            self.event_cache_lengths = ()
            self.event_relative_l2_displacements = ()

    class Session:
        def replay(self, prompt, response_ids):
            del prompt
            if state["student_calls"] == 1:
                assert state["first_student"]() is None
            logits = torch.randn(1, len(response_ids), 8)
            if state["student_calls"] == 0:
                state["first_student"] = weakref.ref(logits)
            state["student_calls"] += 1
            return Replay(logits)

    def fake_teacher(backbone, context_ids, response_ids):
        del backbone, context_ids
        if state["teacher_calls"] == 1:
            assert state["first_teacher"]() is None
        logits = torch.randn(1, len(response_ids), 8)
        if state["teacher_calls"] == 0:
            state["first_teacher"] = weakref.ref(logits)
        state["teacher_calls"] += 1
        return logits

    monkeypatch.setattr(distill_pipeline, "teacher_answer_logits", fake_teacher)
    with torch.no_grad():
        result = distillation_example_loss(
            _tiny_bgt(),
            example,
            Session(),
            temperature=1.1,
            token_clip=None,
            visible_trajectory_mode="both",
        )

    assert result.answer_tokens == 5
    assert state["student_calls"] == 2
    assert state["teacher_calls"] == 2



def test_distillation_example_loss_uses_teacher_cot_and_trains_only_bgt():
    model = _tiny_bgt()
    model.backbone.requires_grad_(False)
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,)),
        teacher_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(10, 11, 12),
        response_ids=(5, 6, 7),
    )

    result = distillation_example_loss(
        model,
        example,
        BGTDecodeSession(model, process_every=2),
        temperature=1.1,
        token_clip=None,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.answer_tokens == 3
    assert result.event_cache_lengths == (3, 6)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.cache_processor.parameters())



def test_batched_teacher_sampling_matches_independent_serial_generation():
    backbone = _tiny_bgt().backbone
    prompts = ((1, 2, 3), (4, 5, 6))
    serial = tuple(
        sample_backbone_completion(
            backbone,
            prompt,
            max_new_tokens=4,
            eos_token_id=None,
            stop_ids=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generator=torch.Generator().manual_seed(10 + index),
        )
        for index, prompt in enumerate(prompts)
    )

    batched = sample_backbone_completions(
        backbone,
        prompts,
        max_new_tokens=4,
        eos_token_id=None,
        stop_ids=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        generators=tuple(
            torch.Generator().manual_seed(10 + index) for index in range(len(prompts))
        ),
    )

    assert batched == serial


def test_batched_teacher_sampling_rejects_mixed_prompt_lengths():
    backbone = _tiny_bgt().backbone

    with pytest.raises(ValueError, match="equal token lengths"):
        sample_backbone_completions(
            backbone,
            ((1, 2), (3, 4, 5)),
            max_new_tokens=2,
            eos_token_id=None,
            stop_ids=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generators=(torch.Generator(), torch.Generator()),
        )


def test_teacher_sampling_honours_reasoning_stop_sequence():
    backbone = _tiny_bgt().backbone
    prompt_ids = (1, 2, 3)
    full = sample_backbone_completion(
        backbone,
        prompt_ids,
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=None,
        temperature=1.1,
        top_p=1.0,
        top_k=1,
    )
    stopped = sample_backbone_completion(
        backbone,
        prompt_ids,
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=(full[0],),
        temperature=1.1,
        top_p=1.0,
        top_k=1,
    )

    assert len(full) == 3
    assert stopped == (full[0],)



def test_teacher_trace_store_round_trips_prompt_bound_failure(tmp_path):
    path = tmp_path / "teacher-trajectories.jsonl"
    store = TeacherTraceStore(path, fingerprint="teacher-native-v1")
    store.put_failure(
        source_index=7,
        prompt="original",
        reason="native teacher completion must end with EOT",
    )

    reopened = TeacherTraceStore(path, fingerprint="teacher-native-v1")
    assert reopened.get_failure(source_index=7, prompt="original") == (
        "native teacher completion must end with EOT"
    )
    with pytest.raises(ValueError, match="prompt hash"):
        reopened.get_failure(source_index=7, prompt="changed")


def test_build_example_reuses_cached_teacher_failure(tmp_path, monkeypatch):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    calls = []

    def fail_teacher(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("native teacher completion must end with EOT")

    monkeypatch.setattr(distill_trainer, "sample_backbone_completion", fail_teacher)
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=5,
        max_response_tokens=3,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "record": distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
        "distill": distill,
        "trace_store": store,
        "close_ids": (10, 11),
        "decode_session": FixedRolloutSession((52, 62)),
        "seed": 24,
        "device": torch.device("cpu"),
    }

    for _ in range(2):
        with pytest.raises(
            distill_trainer.InvalidDistillationExample,
            match="native teacher completion must end with EOT",
        ):
            distill_trainer._build_example(**kwargs)

    assert len(calls) == 1
    assert store.get_failure(source_index=0, prompt="solve this") == (
        "native teacher completion must end with EOT"
    )


def test_teacher_trace_store_round_trips_native_full_trajectory(tmp_path):
    path = tmp_path / "teacher-trajectories.jsonl"
    store = TeacherTraceStore(path, fingerprint="teacher-native-v1")
    store.put_native_trajectory(
        source_index=7,
        prompt="original",
        reasoning_ids=(40, 22, 23),
        visible_response_ids=(50, 51, 0),
        full_completion_ids=(40, 22, 23, 50, 51, 0),
    )

    trajectory = TeacherTraceStore(path, fingerprint="teacher-native-v1").get_native_trajectory(
        source_index=7, prompt="original"
    )
    assert trajectory is not None
    assert trajectory.reasoning_ids == (40, 22, 23)
    assert trajectory.visible_response_ids == (50, 51, 0)
    assert trajectory.full_completion_ids == (40, 22, 23, 50, 51, 0)



def test_teacher_trace_store_round_trips_and_rejects_prompt_drift(tmp_path):
    path = tmp_path / "teacher-traces.jsonl"
    store = TeacherTraceStore(path, fingerprint="teacher-v1")
    store.put(source_index=7, prompt="original", reasoning_ids=(10, 11, 12))

    reopened = TeacherTraceStore(path, fingerprint="teacher-v1")
    assert reopened.get(source_index=7, prompt="original") == (10, 11, 12)
    with pytest.raises(ValueError, match="prompt hash"):
        reopened.get(source_index=7, prompt="changed")


def test_teacher_trace_store_rejects_teacher_fingerprint_drift(tmp_path):
    path = tmp_path / "teacher-traces.jsonl"
    TeacherTraceStore(path, fingerprint="teacher-v1").put(
        source_index=0,
        prompt="p",
        reasoning_ids=(1, 2),
    )

    with pytest.raises(ValueError, match="fingerprint"):
        TeacherTraceStore(path, fingerprint="teacher-v2")



class TrainingHFBackend:
    def apply_chat_template(self, messages, **kwargs):
        base = [1, 2]
        if not kwargs["add_generation_prompt"]:
            return base
        return base + ([8] if kwargs["enable_thinking"] else [3])

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [8, 9]


class FixedRolloutSession:
    def __init__(self, response_ids):
        self.response_ids = response_ids

    def rollout(self, *args, **kwargs):
        del args, kwargs
        return type("Rollout", (), {"response_ids": self.response_ids})()


class NativeTrainingHFBackend(TrainingHFBackend):
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<think>\n":
            return [8, 9]
        if text == "</think>\n\n":
            return [10, 11]
        raise AssertionError(text)


def test_precompute_native_teacher_records_batches_equal_prompt_lengths(
    tmp_path, monkeypatch
):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    records = (
        distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
        distill_trainer.PromptRecord(source_index=1, prompt="solve this"),
    )
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    calls = []

    def fake_batch(*args, **kwargs):
        calls.append((args, kwargs))
        return (
            (40, 10, 11, 50, 62),
            (40, 10, 11, 51, 62),
        )

    monkeypatch.setattr(distill_precompute, "sample_backbone_completions", fake_batch)
    report = distill_precompute.precompute_native_teacher_records(
        model,
        tokenizer,
        records,
        DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            max_teacher_completion_tokens=5,
            max_teacher_reasoning_tokens=3,
            max_response_tokens=3,
            reasoning_close_text="</think>\n\n",
            close_reasoning_at_eos=False,
        ),
        store,
        batch_size=2,
        seed=24,
        device=torch.device("cpu"),
    )

    assert len(calls) == 1
    assert report == {
        "records": 2,
        "cached": 0,
        "generated": 2,
        "valid": 2,
        "invalid": 0,
        "completion_tokens": 10,
        "generation_batches": 1,
        "singleton_batches": 0,
        "max_observed_batch_size": 2,
    }
    assert store.get_native_trajectory(source_index=0, prompt="solve this") is not None
    assert store.get_native_trajectory(source_index=1, prompt="solve this") is not None


def test_build_cached_examples_batch_matches_serial_student_rollout(tmp_path):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    distill = DistillationConfig(
        teacher_trajectory_mode="reasoning_only",
        visible_trajectory_mode="student",
        max_teacher_completion_tokens=5,
        max_teacher_reasoning_tokens=3,
        max_response_tokens=3,
        top_k=1,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
        gradient_accumulation_steps=2,
        execution_microbatch_size=2,
        require_precomputed_teacher_traces=True,
    )
    records = (
        distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
        distill_trainer.PromptRecord(source_index=1, prompt="solve this"),
    )
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    for record in records:
        store.put(
            source_index=record.source_index,
            prompt=record.prompt,
            reasoning_ids=(40, 10, 11),
        )
    serial_session = BGTDecodeSession(
        model,
        process_every=distill.response_process_every,
        valid_vocab_size=tokenizer.get_vocab_size(),
    )
    serial = tuple(
        distill_trainer._build_example(
            model,
            tokenizer,
            record,
            distill,
            store,
            close_ids=(10, 11),
            decode_session=serial_session,
            seed=24,
            device=torch.device("cpu"),
        )[0]
        for record in records
    )

    batched = distill_trainer._build_cached_examples_batch(
        model,
        tokenizer,
        records,
        distill,
        store,
        close_ids=(10, 11),
        decode_session=BGTDecodeSession(
            model,
            process_every=distill.response_process_every,
            valid_vocab_size=tokenizer.get_vocab_size(),
        ),
        seed=24,
        device=torch.device("cpu"),
    )

    assert tuple(result for result in batched) == serial


def test_build_cached_examples_batch_localizes_student_rollout_value_errors(tmp_path):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    distill = DistillationConfig(
        teacher_trajectory_mode="reasoning_only",
        visible_trajectory_mode="student",
        max_teacher_reasoning_tokens=3,
        max_response_tokens=3,
        top_k=1,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
        gradient_accumulation_steps=2,
        execution_microbatch_size=2,
        require_precomputed_teacher_traces=True,
    )
    records = (
        distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
        distill_trainer.PromptRecord(source_index=1, prompt="solve this"),
    )
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    for record in records:
        store.put(
            source_index=record.source_index,
            prompt=record.prompt,
            reasoning_ids=(40, 10, 11),
        )

    class FailingBatchSession:
        def __init__(self):
            self.serial_calls = 0

        def rollout_many(self, *args, **kwargs):
            raise ValueError("one row failed")

        def rollout(self, *args, **kwargs):
            self.serial_calls += 1
            if self.serial_calls == 2:
                raise ValueError("second row failed")
            return RolloutResult(response_ids=(52, 62), event_cache_lengths=())

    session = FailingBatchSession()
    built = distill_trainer._build_cached_examples_batch(
        model,
        tokenizer,
        records,
        distill,
        store,
        close_ids=(10, 11),
        decode_session=session,
        seed=24,
        device=torch.device("cpu"),
    )

    assert isinstance(built[0], DistillationExample)
    assert isinstance(built[1], distill_trainer.InvalidDistillationExample)
    assert str(built[1]) == "second row failed"
    assert session.serial_calls == 2


def test_build_cached_examples_batch_converts_prompt_errors_per_row(
    tmp_path, monkeypatch
):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    distill = DistillationConfig(
        max_teacher_reasoning_tokens=3,
        max_response_tokens=3,
        top_k=1,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
        gradient_accumulation_steps=2,
        execution_microbatch_size=2,
        require_precomputed_teacher_traces=True,
    )
    records = (
        distill_trainer.PromptRecord(source_index=0, prompt="bad"),
        distill_trainer.PromptRecord(source_index=1, prompt="good"),
    )
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    store.put(source_index=1, prompt="good", reasoning_ids=(40, 10, 11))
    real_format = distill_trainer.format_chat_prompt

    def sometimes_bad(tokenizer, prompt, *, enable_thinking):
        if prompt == "bad":
            raise ValueError("bad prompt")
        return real_format(tokenizer, prompt, enable_thinking=enable_thinking)

    monkeypatch.setattr(distill_trainer, "format_chat_prompt", sometimes_bad)

    class Session:
        def rollout_many(self, prompts, **kwargs):
            assert len(prompts) == 1
            return (RolloutResult(response_ids=(52, 62), event_cache_lengths=()),)

    built = distill_trainer._build_cached_examples_batch(
        model,
        tokenizer,
        records,
        distill,
        store,
        close_ids=(10, 11),
        decode_session=Session(),
        seed=24,
        device=torch.device("cpu"),
    )

    assert isinstance(built[0], distill_trainer.InvalidDistillationExample)
    assert str(built[0]) == "bad prompt"
    assert isinstance(built[1], DistillationExample)


def test_build_example_native_full_generates_through_eot_and_caches_visible_answer(
    tmp_path, monkeypatch
):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    calls = []

    def fake_teacher(*args, **kwargs):
        calls.append(kwargs)
        return (40, 41, 10, 11, 50, 51, 62)

    monkeypatch.setattr(distill_trainer, "sample_backbone_completion", fake_teacher)
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=7,
        max_response_tokens=3,
        top_k=1,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )
    store = TeacherTraceStore(tmp_path / "native-trajectories.jsonl", fingerprint="native-v1")
    record = distill_trainer.PromptRecord(source_index=0, prompt="solve this")
    session = FixedRolloutSession((52, 53, 62))

    example, cached = distill_trainer._build_example(
        model,
        tokenizer,
        record,
        distill,
        store,
        close_ids=(10, 11),
        decode_session=session,
        seed=24,
        device=torch.device("cpu"),
    )

    assert cached is False
    assert calls[0]["stop_ids"] is None
    assert calls[0]["max_new_tokens"] == 7
    assert example.teacher_reasoning_ids == (40, 41, 10, 11)
    assert example.teacher_response_ids == (50, 51, 62)
    persisted = store.get_native_trajectory(source_index=0, prompt="solve this")
    assert persisted is not None
    assert persisted.full_completion_ids == (40, 41, 10, 11, 50, 51, 62)



def test_build_example_native_full_rejects_student_budget_exhaustion(tmp_path, monkeypatch):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    monkeypatch.setattr(
        distill_trainer,
        "sample_backbone_completion",
        lambda *args, **kwargs: (40, 10, 11, 50, 62),
    )
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=5,
        max_response_tokens=3,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )

    with pytest.raises(distill_trainer.InvalidDistillationExample, match="student.*EOT"):
        distill_trainer._build_example(
            model,
            tokenizer,
            distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
            distill,
            TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1"),
            close_ids=(10, 11),
            decode_session=FixedRolloutSession((52, 53, 54)),
            seed=24,
            device=torch.device("cpu"),
        )



@pytest.mark.parametrize("response_ids", [(52, 8, 9, 53, 62), (52, 10, 11, 62)])
def test_build_example_native_full_rejects_student_thinking_tokens(
    tmp_path, monkeypatch, response_ids
):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    monkeypatch.setattr(
        distill_trainer,
        "sample_backbone_completion",
        lambda *args, **kwargs: (40, 10, 11, 50, 62),
    )
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=5,
        max_response_tokens=5,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )

    with pytest.raises(distill_trainer.InvalidDistillationExample, match="student.*thinking"):
        distill_trainer._build_example(
            model,
            tokenizer,
            distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
            distill,
            TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1"),
            close_ids=(10, 11),
            decode_session=FixedRolloutSession(response_ids),
            seed=24,
            device=torch.device("cpu"),
        )



def test_build_example_native_full_rejects_malformed_cached_trajectory(tmp_path):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    store.put_native_trajectory(
        source_index=0,
        prompt="solve this",
        reasoning_ids=(40,),
        visible_response_ids=(50, 62),
        full_completion_ids=(40, 50, 62),
    )
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_completion_tokens=5,
        max_response_tokens=2,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )

    with pytest.raises(ValueError, match="exactly one thinking-close"):
        distill_trainer._build_example(
            model,
            tokenizer,
            distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
            distill,
            store,
            close_ids=(10, 11),
            decode_session=FixedRolloutSession((52, 62)),
            seed=24,
            device=torch.device("cpu"),
        )



@pytest.mark.parametrize(
    ("max_reasoning", "max_completion", "message"),
    [(4, 10, "reasoning"), (10, 6, "completion")],
)
def test_build_example_native_full_rechecks_cached_token_caps(
    tmp_path, max_reasoning, max_completion, message
):
    model = _tiny_bgt()
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    store = TeacherTraceStore(tmp_path / "native.jsonl", fingerprint="native-v1")
    store.put_native_trajectory(
        source_index=0,
        prompt="solve this",
        reasoning_ids=(40, 41, 42, 10, 11),
        visible_response_ids=(50, 62),
        full_completion_ids=(40, 41, 42, 10, 11, 50, 62),
    )
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_reasoning_tokens=max_reasoning,
        max_teacher_completion_tokens=max_completion,
        max_response_tokens=2,
        reasoning_close_text="</think>\n\n",
        close_reasoning_at_eos=False,
    )

    with pytest.raises(distill_trainer.InvalidDistillationExample, match=message):
        distill_trainer._build_example(
            model,
            tokenizer,
            distill_trainer.PromptRecord(source_index=0, prompt="solve this"),
            distill,
            store,
            close_ids=(10, 11),
            decode_session=FixedRolloutSession((52, 62)),
            seed=24,
            device=torch.device("cpu"),
        )



def test_locked_example_payload_retains_native_teacher_visible_response():
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(40, 10, 11),
        response_ids=(52, 62),
        teacher_response_ids=(50, 62),
    )

    payload = distill_trainer._locked_examples_payload(
        [(distill_trainer.PromptRecord(source_index=7, prompt="p"), example)]
    )

    assert payload["schema"] == "gcw.opsd-overfit-locked-examples.v2"
    assert payload["examples"][0]["teacher_response_ids"] == [50, 62]



def test_trainer_passes_native_both_trajectory_mode_to_loss(tmp_path, monkeypatch):
    source_model = _tiny_bgt()
    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    save_file(
        {
            name: tensor.detach().clone()
            for name, tensor in source_model.backbone.state_dict().items()
        },
        checkpoint / "model.safetensors",
    )
    data_path = tmp_path / "prompts.jsonl"
    data_path.write_text(
        '{"prompt":"solve this"}\n{"prompt":"solve this too"}\n', encoding="utf-8"
    )
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = NativeTrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    monkeypatch.setattr(distill_trainer, "get_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        distill_trainer,
        "sample_backbone_completion",
        lambda *args, **kwargs: (40, 10, 11, 50, 62),
    )
    monkeypatch.setattr(
        BGTDecodeSession,
        "rollout",
        lambda *args, **kwargs: type("Rollout", (), {"response_ids": (52, 62)})(),
    )
    modes = []
    wandb_inits = []
    wandb_logs = []
    wandb_finished = []

    class FakeWandbRun:
        def __init__(self):
            self.summary = {}
            self.id = "fake-run-id"
            self.url = "https://wandb.example/fake-run-id"

        def define_metric(self, *args, **kwargs):
            return None

        def log(self, metrics, step):
            wandb_logs.append((metrics, step))

        def finish(self):
            wandb_finished.append(True)

    fake_wandb = type(
        "FakeWandb",
        (),
        {
            "init": staticmethod(
                lambda **kwargs: (wandb_inits.append(kwargs) or FakeWandbRun())
            )
        },
    )()
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    def fake_loss(model, example, session, **kwargs):
        del example, session
        mode = kwargs["visible_trajectory_mode"]
        modes.append(mode)
        loss = torch.stack(
            [parameter.float().sum() for parameter in model.cache_processor.parameters()]
        ).sum() * 1e-6
        zero = torch.zeros((), dtype=torch.float32)
        return type(
            "Result",
            (),
            {
                "loss": loss,
                "unclipped_kl": loss.detach(),
                "clipping_fraction": zero,
                "student_entropy": zero,
                "teacher_entropy": zero,
                "answer_tokens": 2,
                "student_trajectory_tokens": 2 if mode == "student" else 0,
                "teacher_trajectory_tokens": 2 if mode == "teacher" else 0,
                "event_cache_lengths": (2,),
                "event_relative_l2_displacements": (
                    (0.1,) if mode == "student" else (0.2,)
                ),
                "teacher_trajectory_event_cache_lengths": (
                    (2,) if mode == "teacher" else ()
                ),
                "teacher_trajectory_event_relative_l2_displacements": (),
            },
        )()

    monkeypatch.setattr(distill_trainer, "distillation_example_loss", fake_loss)
    config = TrainConfig(
        model=source_model.config,
        data=DataConfig(
            kind="openmath_prompts",
            data_dir=str(data_path),
            tokenizer="hf:fake",
            sft_take_rows=2,
        ),
        mode="cot_distillation",
        freeze_backbone=True,
        backbone_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "run"),
        batch_size=1,
        max_steps=1,
        wandb=WandbConfig(enabled=True, project="test-project", name="native-test"),
        distillation=DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            max_teacher_completion_tokens=5,
            max_teacher_reasoning_tokens=3,
            max_response_tokens=2,
            reasoning_close_text="</think>\n\n",
            close_reasoning_at_eos=False,
            max_optimizer_steps=1,
        ),
    )

    distill_trainer.train_distillation(config)

    assert modes == ["student", "teacher"]
    metric = __import__("json").loads(
        (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()[0]
    )
    assert metric["visible_trajectory_mode"] == "both"
    assert metric["student_trajectory_tokens"] == 2
    assert metric["teacher_trajectory_tokens"] == 2
    assert metric["teacher_visible_response_tokens"] == 2
    assert metric["teacher_trajectory_cache_events"] == [2]
    assert metric["cache_displacement_audited"] is False
    assert set(metric["phase_seconds"]) == {
        "example_construction",
        "backward",
        "optimizer",
    }
    assert all(value >= 0.0 for value in metric["phase_seconds"].values())
    assert metric["cache_relative_l2_displacements"] == [0.1]
    assert metric["teacher_trajectory_cache_relative_l2_displacements"] == [0.2]
    assert len(wandb_inits) == 1
    assert wandb_inits[0]["project"] == "test-project"
    assert wandb_inits[0]["name"] == "native-test"
    assert len(wandb_logs) == 1
    logged, step = wandb_logs[0]
    assert step == 1
    assert logged["train/step"] == 1
    assert logged["train/loss"] == pytest.approx(metric["loss"])
    assert logged["train/effective_batch_examples"] == 1
    assert wandb_finished == [True]
    summary = __import__("json").loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["wandb_run_id"] == "fake-run-id"
    assert summary["wandb_run_url"] == "https://wandb.example/fake-run-id"

    assert config.distillation is not None
    partial_config = replace(
        config,
        output_dir=str(tmp_path / "partial-run"),
        wandb=WandbConfig(enabled=False),
        distillation=replace(
            config.distillation,
            gradient_accumulation_steps=3,
            max_optimizer_steps=1,
        ),
    )
    with pytest.raises(
        RuntimeError,
        match=r"partial_examples=2, required_batch_examples=3",
    ):
        distill_trainer.train_distillation(partial_config)
    assert not (tmp_path / "partial-run" / "final").exists()



def test_native_both_locked_diagnostics_use_equal_example_objective_and_both_displacements(
    monkeypatch,
):
    model = _tiny_bgt()
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(40, 10, 11),
        response_ids=(52, 62),
        teacher_response_ids=(50, 62),
    )
    values = iter(((1.0, 2, 0.1, 0.2), (3.0, 10, 0.3, 0.4)))

    def fake_loss(*args, **kwargs):
        del args
        assert kwargs["visible_trajectory_mode"] == "both"
        loss_value, tokens, student_displacement, teacher_displacement = next(values)
        scalar = torch.tensor(loss_value)
        return type(
            "Result",
            (),
            {
                "loss": scalar,
                "unclipped_kl": scalar,
                "answer_tokens": tokens,
                "event_relative_l2_displacements": (student_displacement,),
                "teacher_trajectory_event_relative_l2_displacements": (
                    teacher_displacement,
                ),
            },
        )()

    monkeypatch.setattr(distill_trainer, "distillation_example_loss", fake_loss)
    config = _distillation_train_config(
        distillation=DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            close_reasoning_at_eos=False,
        )
    )

    metrics = distill_trainer._evaluate_locked_examples(
        model,
        [example, example],
        BGTDecodeSession(model, process_every=2),
        config,
        torch.device("cpu"),
    )

    assert metrics["objective_aggregation"] == "equal_example_mean_of_equal_trajectory_means"
    assert metrics["objective_mean_loss"] == pytest.approx(2.0)
    assert metrics["objective_mean_unclipped_kl"] == pytest.approx(2.0)
    assert metrics["student_cache_events"] == 2
    assert metrics["teacher_cache_events"] == 2
    assert metrics["mean_student_cache_relative_l2_displacement"] == pytest.approx(0.2)
    assert metrics["mean_teacher_cache_relative_l2_displacement"] == pytest.approx(0.3)
    assert "token_weighted_loss" not in metrics


def test_student_locked_diagnostics_use_equal_example_mean_with_unequal_lengths(
    monkeypatch,
):
    model = _tiny_bgt()
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
        teacher_prompt=ChatPrompt(base_ids=(1,), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(40, 10, 11),
        response_ids=(52, 62),
    )
    values = iter(((1.0, 2), (3.0, 10)))

    def fake_loss(*args, **kwargs):
        del args
        assert kwargs["visible_trajectory_mode"] == "student"
        value, tokens = next(values)
        scalar = torch.tensor(value)
        return type(
            "Result",
            (),
            {
                "loss": scalar,
                "unclipped_kl": scalar,
                "answer_tokens": tokens,
                "event_relative_l2_displacements": (),
                "teacher_trajectory_event_relative_l2_displacements": (),
            },
        )()

    monkeypatch.setattr(distill_trainer, "distillation_example_loss", fake_loss)
    config = _distillation_train_config(
        distillation=DistillationConfig(
            teacher_trajectory_mode="reasoning_only",
            visible_trajectory_mode="student",
            close_reasoning_at_eos=False,
        )
    )
    metrics = distill_trainer._evaluate_locked_examples(
        model,
        [example, example],
        BGTDecodeSession(model, process_every=2),
        config,
        torch.device("cpu"),
    )

    assert metrics["objective_aggregation"] == "equal_example_mean"
    assert metrics["objective_mean_loss"] == pytest.approx(2.0)
    assert metrics["objective_mean_unclipped_kl"] == pytest.approx(2.0)
    assert metrics["answer_tokens"] == 12
    assert "token_weighted_unclipped_kl" not in metrics


def test_distillation_trainer_reuses_fixed_trace_across_overfit_epochs(tmp_path, monkeypatch):
    source_model = _tiny_bgt()
    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    save_file(
        {name: tensor.detach().clone() for name, tensor in source_model.backbone.state_dict().items()},
        checkpoint / "model.safetensors",
    )
    data_path = tmp_path / "prompts.jsonl"
    data_path.write_text('{"prompt":"solve this"}\n', encoding="utf-8")
    output_dir = tmp_path / "run"

    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = TrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    monkeypatch.setattr(distill_trainer, "get_tokenizer", lambda _: tokenizer)
    teacher_calls = []

    def fake_teacher(*args, **kwargs):
        teacher_calls.append((args, kwargs))
        return (8, 9)

    monkeypatch.setattr(distill_trainer, "sample_backbone_completion", fake_teacher)
    verification_saw_staging = []
    real_verify = distill_trainer._verify_overfit_checkpoint

    def verify_staging(checkpoint_path, *args, **kwargs):
        verification_saw_staging.append(checkpoint_path.name.startswith(".final-overfit-staging"))
        assert not (output_dir / "final").exists()
        return real_verify(checkpoint_path, *args, **kwargs)

    monkeypatch.setattr(distill_trainer, "_verify_overfit_checkpoint", verify_staging)
    config = TrainConfig(
        model=source_model.config,
        data=DataConfig(
            kind="openmath_prompts",
            data_dir=str(data_path),
            tokenizer="hf:fake",
            sft_take_rows=1,
        ),
        mode="cot_distillation",
        freeze_backbone=True,
        backbone_checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        distillation=DistillationConfig(
            epochs=2,
            overfit_diagnostic=True,
            expected_locked_examples=1,
            max_teacher_reasoning_tokens=2,
            max_response_tokens=3,
            top_k=1,
            response_process_every=2,
            reasoning_close_text="close",
        ),
    )

    distill_trainer.train_distillation(config)

    summary = __import__("json").loads((output_dir / "summary.json").read_text())
    assert summary["epoch"] == 2
    assert summary["seen_prompts"] == 2
    assert summary["valid_prompts"] == 2
    assert summary["skipped_prompts"] == 0
    assert summary["optimizer_steps"] == 2
    assert (output_dir / "final" / "model.safetensors").is_file()
    assert verification_saw_staging == [True]
    metric_rows = [
        __import__("json").loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in metric_rows] == [1, 2]
    assert [row["teacher_trace_cached"] for row in metric_rows] == [True, True]
    assert all(isinstance(row["processor_gate_before_example"], float) for row in metric_rows)
    assert all(isinstance(row["processor_gate_after_example"], float) for row in metric_rows)
    assert all(
        len(row["cache_relative_l2_displacements"]) == len(row["cache_events"])
        for row in metric_rows
    )
    assert len(teacher_calls) == 1
    diagnostic = summary["overfit_diagnostic"]
    assert diagnostic["locked_examples"] == 1
    assert diagnostic["lock_attempts"] == 1
    assert diagnostic["lock_skips"] == []
    assert diagnostic["locked_source_indices"] == [0]
    assert diagnostic["initial"]["examples"] == 1
    assert diagnostic["pre_save"]["examples"] == 1
    assert diagnostic["post_load"]["examples"] == 1
    assert diagnostic["pre_save"] == diagnostic["post_load"]
    assert diagnostic["processor_changed_from_initialization"] is True
    assert diagnostic["initial_processor_sha256"] != diagnostic["pre_save_processor_sha256"]
    assert diagnostic["logit_probe_response_tokens"] == 3
    assert len(diagnostic["logit_probe_event_cache_lengths"]) == 2
    assert diagnostic["pre_save_logit_probe_sha256"] == diagnostic["post_load_logit_probe_sha256"]
    assert diagnostic["post_load_logits_verified"] is True
    loaded, loaded_config, loaded_state = load_checkpoint(output_dir / "final")
    assert isinstance(loaded, BGT)
    assert loaded_config == source_model.config
    assert loaded_state["valid_prompts"] == 2



def test_exact_forward_kl_computes_bfloat16_logits_in_float32():
    student = torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.bfloat16)
    loss = exact_forward_kl(student, teacher, torch.ones((1, 1), dtype=torch.bool))

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)



def test_blockwise_replay_matches_tokenwise_reference_between_events():
    model = _tiny_bgt()
    session = BGTDecodeSession(model, process_every=64)
    prompt = ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,))
    response = (5, 6, 7, 8, 9)

    next_logits, cache, events, displacements = session._start(prompt)
    assert displacements == []
    tokenwise = []
    for response_index, token_id in enumerate(response):
        tokenwise.append(next_logits)
        consumed = response_index + 1
        if consumed < len(response):
            next_logits, cache = session._consume(
                prompt,
                token_id,
                response_index,
                consumed,
                cache,
                events,
                will_continue=True,
            )

    blockwise = session.replay(prompt, response)
    torch.testing.assert_close(blockwise.logits, torch.stack(tokenwise, dim=1), atol=1e-6, rtol=1e-5)



def test_train_dispatches_distillation_before_distributed_initialisation(monkeypatch):
    from gcw.train import train

    calls = []
    monkeypatch.setattr(distill_trainer, "train_distillation", lambda config: calls.append(config))
    config = _distillation_train_config()

    train(config)

    assert calls == [config]



def test_sample_next_token_masks_padded_model_vocabulary():
    logits = torch.tensor([[0.0, 1.0, 100.0]])
    token = sample_next_token(
        logits,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        valid_vocab_size=2,
    )
    assert token.item() == 1



def test_native_teacher_incremental_answer_logits_match_full_forward(monkeypatch):
    backbone = _tiny_bgt().backbone
    context = (1, 2, 3, 4)
    response = (5, 6, 7)
    expected_full = backbone.forward_full(torch.tensor([context + response]))
    expected = expected_full[:, len(context) - 1 : len(context) - 1 + len(response)]
    monkeypatch.setattr(
        backbone,
        "forward_full",
        lambda _: (_ for _ in ()).throw(AssertionError("native teacher scoring must not project context logits")),
    )

    actual = teacher_answer_logits(backbone, context, response)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)



def test_chunked_kl_matches_unchunked_value_and_student_gradient():
    torch.manual_seed(3)
    base_student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])
    student_full = base_student.clone().requires_grad_(True)
    student_chunked = base_student.clone().requires_grad_(True)

    full = exact_forward_kl(
        student_full,
        teacher,
        mask,
        temperature=1.1,
        token_clip=None,
        sequence_chunk_size=5,
    )
    chunked = exact_forward_kl(
        student_chunked,
        teacher,
        mask,
        temperature=1.1,
        token_clip=None,
        sequence_chunk_size=2,
    )
    full.backward()
    chunked.backward()

    torch.testing.assert_close(chunked, full, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(student_chunked.grad, student_full.grad, atol=1e-7, rtol=1e-6)



@pytest.mark.parametrize(
    ("processor_overrides", "message"),
    [
        ({"attention_dropout": 0.1}, "zero processor attention dropout"),
        ({"processor_mode": "workspace_broadcast"}, "standard cache processor mode"),
    ],
)
def test_distillation_config_rejects_nondeterministic_or_nonstandard_processor(processor_overrides, message):
    processor = CacheProcessorConfig(
        process_every=64,
        topk=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        **processor_overrides,
    )
    model = ModelConfig(
        variant="bgt",
        family="qwen3",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=processor,
    )
    with pytest.raises(ValueError, match=message):
        _distillation_train_config(model=model)


def test_openmath_take_zero_yields_no_records(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"problem": "unused"}\n')
    assert list(iter_openmath_prompts(source, take_rows=0)) == []



def test_teacher_fingerprint_invalidates_seed_limit_and_chat_template(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"frozen-weights")
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = FakeHFTokenizerBackend()
    tokenizer.enc.chat_template = "template-v1"
    tokenizer.enc.name_or_path = "fake"
    tokenizer.enc.eos_token_id = 62
    tokenizer.enc.pad_token_id = None
    tokenizer.enc.bos_token_id = 1
    tokenizer.enc.special_tokens_map = {"eos_token": "<eos>"}
    tokenizer.enc.get_vocab = lambda: {"a": 0, "<eos>": 62}
    tokenizer.source = "fake"
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    base = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=8),
    )
    baseline = distill_trainer._teacher_fingerprint(base, (6, 7), (8, 9), tokenizer)
    replace = __import__("dataclasses").replace
    assert base.model.cache_processor is not None
    changed_gate = replace(
        base,
        model=replace(
            base.model,
            cache_processor=replace(base.model.cache_processor, gate_init=-3.0),
        ),
    )
    assert (
        distill_trainer._teacher_fingerprint(
            changed_gate, (6, 7), (8, 9), tokenizer
        )
        == baseline
    )
    changed_seed = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=8,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=8),
    )
    changed_limit = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=9),
    )
    assert distill_trainer._teacher_fingerprint(changed_seed, (6, 7), (8, 9), tokenizer) != baseline
    assert distill_trainer._teacher_fingerprint(changed_limit, (6, 7), (8, 9), tokenizer) != baseline
    changed_native_mode = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            max_teacher_completion_tokens=64,
            close_reasoning_at_eos=False,
        ),
    )
    changed_native_budget = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            max_teacher_completion_tokens=65,
            close_reasoning_at_eos=False,
        ),
    )
    native = distill_trainer._teacher_fingerprint(
        changed_native_mode, (6, 7), (8, 9), tokenizer
    )
    assert (
        distill_trainer._teacher_fingerprint(
            changed_native_budget, (6, 7), (8, 9), tokenizer
        )
        != native
    )
    tokenizer.enc.chat_template = "template-v2"
    assert distill_trainer._teacher_fingerprint(base, (6, 7), (8, 9), tokenizer) != baseline



def test_bf16_backbone_fp32_processor_rollout_and_replay_under_autocast():
    model = _tiny_bgt()
    model.backbone.to(dtype=torch.bfloat16)
    assert next(model.cache_processor.parameters()).dtype == torch.float32
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(model, process_every=2, valid_vocab_size=model.config.vocab_size)
    generator = torch.Generator().manual_seed(4)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        rollout = session.rollout(
            prompt,
            max_response_tokens=3,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generator=generator,
        )
        replay = session.replay(prompt, rollout.response_ids)

    assert replay.logits.shape[:2] == (1, 3)
    replay.logits.float().sum().backward()
    assert any(parameter.grad is not None for parameter in model.cache_processor.parameters())



def test_mixed_final_checkpoint_verifies_exact_processor_round_trip(tmp_path):
    model = _tiny_bgt()
    model.backbone.to(dtype=torch.bfloat16)
    model.cache_processor.to(dtype=torch.float32)
    model.cache_processor.block.gate.data.fill_(-4.9896)
    destination = tmp_path / "final"
    expected_processor = {
        name: tensor.detach().clone()
        for name, tensor in model.cache_processor.state_dict().items()
    }

    published_state = distill_trainer._publish_checkpoint(
        destination,
        model,
        model.config,
        {
            "dtype": "mixed_bfloat16_backbone_float32_processor",
            "valid_prompts": 1,
        },
    )
    loaded, _, state = load_checkpoint(destination)

    assert published_state["checkpoint_fidelity_verified"] is True
    assert state == published_state
    assert state["pre_save_processor_sha256"] == state["post_load_processor_sha256"]
    assert state["pre_save_processor_gate"] == pytest.approx(-4.9896)
    assert state["post_load_processor_gate"] == pytest.approx(-4.9896)
    assert all(parameter.dtype == torch.bfloat16 for parameter in loaded.backbone.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in loaded.cache_processor.parameters())
    for name, expected in expected_processor.items():
        torch.testing.assert_close(loaded.cache_processor.state_dict()[name], expected, rtol=0, atol=0)


def test_replay_records_relative_cache_displacement_when_enabled():
    model = _tiny_bgt()
    session = BGTDecodeSession(
        model,
        process_every=2,
        valid_vocab_size=model.config.vocab_size,
        record_displacements=True,
    )

    replay = session.replay(ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)), (4, 5, 6))

    assert len(replay.event_relative_l2_displacements) == len(replay.event_cache_lengths)
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in replay.event_relative_l2_displacements)


def test_student_rollout_uses_static_cache(monkeypatch):
    model = _tiny_bgt()
    session = BGTDecodeSession(model, process_every=2, valid_vocab_size=model.config.vocab_size)
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    monkeypatch.setattr(
        model.backbone,
        "forward_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rollout must use static cache")),
    )

    rollout = session.rollout(
        prompt,
        max_response_tokens=3,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(5),
    )

    assert len(rollout.response_ids) == 3
    assert len(rollout.event_cache_lengths) == 2



def test_native_teacher_sampling_uses_static_cache(monkeypatch):
    backbone = _tiny_bgt().backbone
    monkeypatch.setattr(
        backbone,
        "forward_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("teacher sampling must use static cache")),
    )
    generated = sample_backbone_completion(
        backbone,
        (1, 2, 3),
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=(),
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        valid_vocab_size=backbone.config.vocab_size,
        generator=torch.Generator().manual_seed(6),
    )
    assert len(generated) == 3



def test_batched_student_rollout_matches_independent_serial_generation():
    model = _tiny_bgt()
    prompts = (
        ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)),
        ChatPrompt(base_ids=(4, 5), trigger_ids=(6,)),
    )
    session = BGTDecodeSession(
        model,
        process_every=2,
        valid_vocab_size=model.config.vocab_size,
    )
    serial = tuple(
        session.rollout(
            prompt,
            max_response_tokens=4,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generator=torch.Generator().manual_seed(20 + index),
        )
        for index, prompt in enumerate(prompts)
    )

    batched = session.rollout_many(
        prompts,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        generators=tuple(
            torch.Generator().manual_seed(20 + index) for index in range(len(prompts))
        ),
    )

    assert batched == serial


def test_batched_student_rollout_rejects_mixed_prompt_shapes():
    model = _tiny_bgt()
    session = BGTDecodeSession(model, process_every=2)

    with pytest.raises(ValueError, match="equal base and trigger token lengths"):
        session.rollout_many(
            (
                ChatPrompt(base_ids=(1,), trigger_ids=(2,)),
                ChatPrompt(base_ids=(3, 4), trigger_ids=(5,)),
            ),
            max_response_tokens=2,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generators=(torch.Generator(), torch.Generator()),
        )


def test_decode_session_can_disable_processor_for_matched_eval_control():
    model = _tiny_bgt()
    recorder = RecordingZeroProcessor()
    model.cache_processor = recorder
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )

    rollout = session.rollout(
        ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)),
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )

    assert len(rollout.response_ids) == 4
    assert rollout.event_cache_lengths == ()
    assert recorder.calls == []



def test_processor_disabled_session_matches_raw_backbone_greedy_generation():
    model = _tiny_bgt()
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )

    rollout = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        greedy=True,
    )
    raw = model.backbone.generate_fast(
        torch.tensor([prompt.full_ids], dtype=torch.long),
        max_new_tokens=4,
        eos_token_id=None,
    )

    assert rollout.response_ids == tuple(raw[0, len(prompt.full_ids) :].tolist())



def test_greedy_decode_session_stops_online_on_token_sequence():
    model = _tiny_bgt()
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )
    baseline = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        greedy=True,
    )
    stopped = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        stop_sequences=[[baseline.response_ids[0]]],
        greedy=True,
    )

    assert stopped.response_ids == baseline.response_ids[:1]
    assert stopped.event_cache_lengths == ()
