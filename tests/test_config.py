import pytest

from gcw.config import DataConfig, ModelConfig, ResumeConfig, TrainConfig, WandbConfig, load_train_config


def test_train_config_builds_explicit_resume_and_wandb_state():
    config = TrainConfig(
        resume={"checkpoint": "runs/example/checkpoint", "topology": "elastic_ddp"},
        wandb={"enabled": True, "id": "run-123", "resume": "must"},
    )

    assert config.resume == ResumeConfig(checkpoint="runs/example/checkpoint", topology="elastic_ddp")
    assert config.wandb == WandbConfig(enabled=True, id="run-123", resume="must")


def test_train_config_rejects_conflicting_checkpoint_sources():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainConfig(
            backbone_checkpoint="runs/backbone/checkpoint",
            resume={"checkpoint": "runs/training/checkpoint"},
        )


@pytest.mark.parametrize("grad_clip", [0.0, -1.0, float("inf"), float("nan")])
def test_train_config_requires_finite_positive_grad_clip(grad_clip):
    with pytest.raises(ValueError, match="grad_clip must be finite and positive"):
        TrainConfig(grad_clip=grad_clip)


@pytest.mark.parametrize("trainable", ["all", "backbone", "cache_processor"])
def test_train_config_accepts_explicit_trainable_policy(trainable):
    config = TrainConfig(
        model=ModelConfig(variant="bgt"),
        data=DataConfig(kind="random"),
        trainable=trainable,
    )
    assert config.trainable == trainable


def test_processor_only_policy_requires_a_processor():
    with pytest.raises(ValueError, match="requires a cache processor"):
        TrainConfig(trainable="cache_processor")


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



def test_load_train_config_bytes_uses_captured_snapshot(tmp_path) -> None:
    from gcw.config import load_train_config_bytes

    path = tmp_path / "train.yaml"
    path.write_text("mode: pretrain\nseed: 17\n")
    snapshot = path.read_bytes()
    path.write_text("mode: sft\nseed: 999\n")

    loaded = load_train_config_bytes(snapshot)
    assert loaded.mode == "pretrain"
    assert loaded.seed == 17
