import pytest

from gcw.config import ModelConfig, ResumeConfig, TrainConfig, WandbConfig, load_train_config


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
