import pytest

from gcw.config import (
    DataConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    WandbConfig,
    load_train_config,
    process_branch_training_contract,
    to_dict,
)


def test_train_config_builds_explicit_resume_and_wandb_state():
    config = TrainConfig(
        resume={"checkpoint": "runs/example/checkpoint", "topology": "elastic_ddp"},
        wandb={"enabled": True, "id": "run-123", "resume": "must"},
    )

    assert config.resume == ResumeConfig(checkpoint="runs/example/checkpoint", topology="elastic_ddp")
    assert config.wandb == WandbConfig(enabled=True, id="run-123", resume="must")


def test_train_config_rejects_conflicting_checkpoint_sources():
    with pytest.raises(ValueError, match="resume and backbone_checkpoint"):
        TrainConfig(
            backbone_checkpoint="runs/backbone/checkpoint",
            resume={"checkpoint": "runs/training/checkpoint"},
        )


@pytest.mark.parametrize("grad_clip", [0.0, -1.0, float("inf"), float("nan")])
def test_train_config_requires_finite_positive_grad_clip(grad_clip):
    with pytest.raises(ValueError, match="grad_clip must be finite and positive"):
        TrainConfig(grad_clip=grad_clip)


def test_wandb_must_resume_requires_run_id():
    with pytest.raises(ValueError, match="wandb.id"):
        WandbConfig(enabled=True, resume="must")


@pytest.mark.parametrize(
    ("path", "topology"),
    [
        ("configs/train/resume/smollm360m_matched_bgt_20b_pe512_resume.yaml", "same"),
        ("configs/train/resume/smollm360m_matched_bt_20b_pe512_resume.yaml", "same"),
        ("configs/train/resume/smollm360m_matched_bt_20b_pe512_resume_2gpu.yaml", "elastic_ddp"),
    ],
)
def test_resume_configs_use_versioned_resume_interface(path, topology):
    config = load_train_config(path)

    assert config.resume is not None
    assert config.resume.topology == topology
    assert config.wandb.id
    assert config.wandb.resume == "must"


@pytest.mark.parametrize("indices", ([1, 0], [0, 0]))
def test_processor_layer_indices_must_be_sorted_and_unique(indices):
    with pytest.raises(ValueError, match="sorted and unique"):
        ModelConfig(num_layers=2, processor_layer_indices=indices)


@pytest.mark.parametrize(
    ("artifact_hash", "source_ids_hash"),
    [
        (None, "1" * 64),
        ("1" * 64, None),
        ("0" * 64, "1" * 64),
        ("1" * 64, "placeholder"),
    ],
)
def test_canonical_data_config_requires_non_placeholder_artifact_identity_hashes(
    artifact_hash,
    source_ids_hash,
):
    with pytest.raises(ValueError, match="canonical data requires authenticated"):
        DataConfig(
            canonical_execution=True,
            expected_artifact_sha256=artifact_hash,
            expected_source_ids_sha256=source_ids_hash,
        )


def test_qwen3_answer_only_precursor_configs_are_parity_bound():
    paths = [
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml",
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_random_depth_6000_seed24.yaml",
    ]
    fixed, random_depth = [load_train_config(path) for path in paths]

    assert fixed.process_branch is not None
    assert random_depth.process_branch is not None
    assert fixed.process_branch.objective == random_depth.process_branch.objective == "answer_only"
    assert fixed.process_branch.cycle_choices == (1,)
    assert random_depth.process_branch.cycle_choices == (1, 2, 4)
    for config in (fixed, random_depth):
        assert config.data.kind == "answer_only_jsonl"
        assert config.data.data_dir == (
            "data/process_branch/gsm8k-answer-only-v1.jsonl"
        )
        assert config.data.canonical_execution is True
        assert config.data.sft_skip_rows == 0
        assert config.data.sft_take_rows == 6000
        assert config.process_branch.heldout_skip_rows == 6000
        assert config.process_branch.heldout_take_rows == 256
        assert config.process_branch.heldout_skip_rows + config.process_branch.heldout_take_rows == 6256
        assert config.metadata["untouched_row_interval"] == [6256, 6512]
        assert config.data.expected_artifact_sha256 == (
            "feaff6bb9bee24a772c4c9d30a5a9c60d1e61bafa5d392f5a719f1e3cedf03c6"
        )
        assert config.data.expected_source_ids_sha256 == (
            "c0330c4dc7c13d36c87e75c03b34d2c616a0ee945d0c8c7e616353dc6de60e3f"
        )

    normalized = []
    for config in (fixed, random_depth):
        payload = to_dict(config)
        payload["output_dir"] = "<ARM_OUTPUT>"
        payload["process_branch"]["cycle_choices"] = ["<DEPTH_CHOICES>"]
        payload["metadata"]["arm"] = "<ARM_NAME>"
        normalized.append(payload)
    assert normalized[0] == normalized[1]


def test_qwen3_answer_only_canary_is_one_step_random_depth_with_frozen_p1_identity():
    canary = load_train_config(
        "configs/train/process_branch_qwen3_06b/"
        "qwen3_06b_answer_only_random_depth_canary_seed24.yaml"
    )
    p1 = load_train_config(
        "configs/train/process_branch_qwen3_06b/"
        "qwen3_06b_answer_only_random_depth_6000_seed24.yaml"
    )

    assert canary.process_branch is not None
    assert p1.process_branch is not None
    assert canary.process_branch.max_optimizer_steps == 1
    assert canary.process_branch.cycle_choices == (1, 2, 4)
    assert "canary-random-depth" in canary.output_dir
    frozen_fields = (
        "expected_train_eligible_records",
        "expected_train_eligible_source_ids_sha256",
        "expected_heldout_eligible_records",
        "expected_heldout_eligible_source_ids_sha256",
        "expected_initial_backbone_sha256",
        "expected_tokenizer_sha256",
    )
    for field in frozen_fields:
        assert getattr(canary.process_branch, field) == getattr(p1.process_branch, field)


def test_process_branch_training_contract_ignores_resume_operational_paths() -> None:
    from gcw.config import ResumeConfig, process_branch_training_contract

    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/"
        "qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    resumed = load_train_config(
        "configs/train/process_branch_qwen3_06b/"
        "qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    resumed.output_dir = "artifacts/resumed"
    resumed.resume = ResumeConfig(checkpoint="artifacts/rolling/step-100")
    resumed.backbone_checkpoint = None
    resumed.data.data_dir = "elsewhere/answer-only.jsonl"
    resumed.data.tokenizer = "hf:elsewhere/tokenizer"
    resumed.data.tokenizer_source = "elsewhere/tokenizer"

    assert process_branch_training_contract(resumed) == process_branch_training_contract(config)


def test_process_branch_training_contract_is_json_normalized() -> None:
    import json

    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    contract = process_branch_training_contract(config)
    assert json.loads(json.dumps(contract)) == contract


def test_load_train_config_bytes_uses_captured_snapshot(tmp_path) -> None:
    from gcw.config import load_train_config_bytes

    path = tmp_path / "train.yaml"
    path.write_text("mode: pretrain\nseed: 17\n")
    snapshot = path.read_bytes()
    path.write_text("mode: process_branch\nseed: 999\n")

    loaded = load_train_config_bytes(snapshot)
    assert loaded.mode == "pretrain"
    assert loaded.seed == 17
