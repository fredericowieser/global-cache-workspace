import pytest

from gcw.config import ResumeConfig, TrainConfig, WandbConfig


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
