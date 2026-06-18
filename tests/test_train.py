import sys

from gcw.config import DataConfig, ModelConfig, TrainConfig, WandbConfig
from gcw.train import train


class FakeRun:
    def __init__(self):
        self.logs = []
        self.metrics = []
        self.summary = {}
        self.finished = False

    def define_metric(self, *args, **kwargs):
        self.metrics.append((args, kwargs))

    def log(self, metrics, step=None):
        self.logs.append((step, dict(metrics)))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.run = FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def tiny_train_config(tmp_path):
    return TrainConfig(
        model=ModelConfig(
            variant="tf",
            family="smollm",
            sequence_len=16,
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            tie_embeddings=True,
        ),
        data=DataConfig(kind="random"),
        distributed="none",
        output_dir=str(tmp_path / "run"),
        batch_size=2,
        sequence_len=8,
        max_steps=2,
        learning_rate=1e-3,
        warmup_steps=1,
        eval_every=1,
        eval_batches=1,
        eval_on_start=True,
        wandb=WandbConfig(enabled=True, project="test-project", name="tiny", mode="offline", log_every=1),
    )


def test_train_logs_wandb_train_and_validation_metrics(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)

    train(tiny_train_config(tmp_path))

    assert fake.init_kwargs["project"] == "test-project"
    assert fake.init_kwargs["name"] == "tiny"
    assert fake.run.finished
    logs = fake.run.logs
    assert [step for step, _ in logs] == [0, 1, 2]
    assert "val/loss" in logs[0][1]
    assert all("train/loss" in metrics for _, metrics in logs[1:])
    assert all("val/loss" in metrics for _, metrics in logs[1:])
    assert all("train/tokens" in metrics for _, metrics in logs[1:])
