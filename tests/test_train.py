import sys

import torch

from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, TrainConfig, WandbConfig
from gcw.checkpoint import load_checkpoint, load_trainer_state, save_checkpoint
from gcw.data.shards import ShardWriter, data_loader
from gcw.data.sft import sft_batch
from gcw.models import build_model
from gcw.train import _checkpoint_metadata, train


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


class TinyTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        table = {"A": 1, "B": 2, "C": 3, "D": 4, "\n": 5}
        return [table[ch] for ch in text]


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


def test_train_passes_wandb_resume_id(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = tiny_train_config(tmp_path)
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1
    cfg.wandb.id = "existing-run"
    cfg.wandb.resume = "must"

    train(cfg)

    assert fake.init_kwargs["id"] == "existing-run"
    assert fake.init_kwargs["resume"] == "must"


def test_resumed_train_logs_wandb_without_explicit_internal_step(monkeypatch, tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1
    train(cfg)

    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    resume = tiny_train_config(tmp_path)
    resume.eval_every = 0
    resume.eval_on_start = False
    resume.max_steps = 2
    resume.resume_from_checkpoint = str(tmp_path / "run" / "checkpoint")
    resume.wandb.id = "existing-run"
    resume.wandb.resume = "must"

    train(resume)

    assert fake.run.logs[-1][0] is None
    assert fake.run.logs[-1][1]["train/step"] == 2


def test_train_reads_pretrain_shards(tmp_path):
    data_dir = tmp_path / "data"
    writer = ShardWriter(data_dir, "train", shard_tokens=1000)
    writer.add_tokens([i % 128 for i in range(200)])
    writer.close()
    cfg = tiny_train_config(tmp_path)
    cfg.data = DataConfig(kind="fineweb", data_dir=str(data_dir))
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1

    train(cfg)

    assert (tmp_path / "run" / "checkpoint" / "train_state.json").exists()


def test_fineweb_loader_resumes_after_saved_step(tmp_path):
    data_dir = tmp_path / "data"
    writer = ShardWriter(data_dir, "train", shard_tokens=1000)
    writer.add_tokens(list(range(40)))
    writer.close()
    shard = data_dir / "train_000000.bin"

    loader = data_loader(
        batch_size=1,
        sequence_len=4,
        data_dir=data_dir,
        start_state={"epoch": 0, "shard": str(shard), "step_in_shard": 2},
    )
    x, y, state = next(loader)

    assert x.tolist() == [[8, 9, 10, 11]]
    assert y.tolist() == [[9, 10, 11, 12]]
    assert state["step_in_shard"] == 3


def test_train_resumes_legacy_checkpoint_without_trainer_state(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1
    train(cfg)
    checkpoint = tmp_path / "run" / "checkpoint"
    (checkpoint / "trainer_state.pt").unlink()

    resume = tiny_train_config(tmp_path)
    resume.wandb.enabled = False
    resume.eval_every = 0
    resume.eval_on_start = False
    resume.max_steps = 2
    resume.resume_from_checkpoint = str(checkpoint)

    train(resume)

    _, _, state = load_checkpoint(checkpoint)
    assert state["step"] == 2
    assert (checkpoint / "trainer_state.pt").exists()


def test_train_writes_reloadable_trainer_state(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1

    train(cfg)

    trainer_state = load_trainer_state(tmp_path / "run" / "checkpoint")
    assert trainer_state is not None
    assert "optimizer" in trainer_state
    assert "torch_rng_state" in trainer_state


def test_train_loads_native_backbone_checkpoint(tmp_path):
    backbone_cfg = ModelConfig(
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
    )
    backbone = build_model(backbone_cfg)
    backbone.init_weights()
    with torch.no_grad():
        backbone.embed_tokens.weight.fill_(0.123)
    checkpoint = tmp_path / "backbone"
    save_checkpoint(checkpoint, backbone, backbone_cfg)

    cfg = tiny_train_config(tmp_path)
    cfg.model = ModelConfig(
        variant="bt",
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
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
        ),
    )
    cfg.freeze_backbone = True
    cfg.backbone_checkpoint = str(checkpoint)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1

    train(cfg)

    trained, _, _ = load_checkpoint(tmp_path / "run" / "checkpoint")
    assert torch.allclose(trained.backbone.embed_tokens.weight, torch.full_like(trained.backbone.embed_tokens.weight, 0.123))


def test_checkpoint_metadata_records_hf_tokenizer_source(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.mode = "sft"
    cfg.data = DataConfig(
        kind="sft",
        dataset="/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl",
        tokenizer="hf",
        tokenizer_source="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model",
    )
    cfg.backbone_checkpoint = "/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"

    state = _checkpoint_metadata(cfg, {"step": 0, "world_size": 1})

    assert state["source"] == cfg.data.tokenizer_source
    assert state["tokenizer_source"] == cfg.data.tokenizer_source
    assert state["tokenizer"] == "hf"
    assert state["backbone_checkpoint"] == cfg.backbone_checkpoint


def test_sft_batch_uses_next_token_targets():
    x, y = sft_batch(
        [{"prompt": "AB", "answer": "CD"}],
        TinyTokenizer(),
        max_length=8,
        response_prefix="\n",
    )

    assert x.tolist() == [[1, 2, 5, 3, 4, 0]]
    assert y.tolist() == [[-100, 5, 3, 4, 0, -100]]
