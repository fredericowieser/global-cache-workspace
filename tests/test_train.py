import json
import sys

import pytest
import torch

from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
    WandbConfig,
)
from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.data.shards import ShardWriter
from gcw.data.sft import sft_batch
from gcw.models import build_model
from gcw.train import (
    _build_optimizer,
    _build_training_model,
    _checkpoint_metadata,
    _checkpoint_state,
    _loader,
    _model_loss,
    _transformer_layer_classes,
    _uses_bf16_autocast,
    _validate_train_batch,
    train,
)


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
        wandb=WandbConfig(
            enabled=True,
            project="test-project",
            name="tiny",
            mode="offline",
            log_every=1,
        ),
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


def test_train_logs_longshot_iteration_overlap_metrics(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = tiny_train_config(tmp_path)
    cfg.model = ModelConfig(
        variant="longshot",
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
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
            encoder_seq_len=4,
            iterations=2,
        ),
    )
    cfg.max_steps = 1
    cfg.eval_every = 0
    cfg.eval_on_start = False

    train(cfg)

    _, metrics = fake.run.logs[0]
    assert "cache_processor/selected_head_overlap_after_iter_1" in metrics
    assert (
        0.0
        <= metrics["cache_processor/selected_head_overlap_after_iter_1"]
        <= 1.0
    )


def test_train_logs_longshot_external_iteration_overlap_metrics(
    monkeypatch, tmp_path
):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    cfg = tiny_train_config(tmp_path)
    cfg.model = ModelConfig(
        variant="longshot",
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
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
            encoder_seq_len=4,
            iterations=2,
            iteration_mode="external",
        ),
    )
    cfg.max_steps = 1
    cfg.eval_every = 0
    cfg.eval_on_start = False

    train(cfg)

    _, metrics = fake.run.logs[0]
    assert "cache_processor/selected_head_overlap_after_iter_1" in metrics
    assert (
        0.0
        <= metrics["cache_processor/selected_head_overlap_after_iter_1"]
        <= 1.0
    )


def test_train_precision_defaults_and_legacy_bf16_alias():
    assert TrainConfig().precision == "fp32"
    assert TrainConfig(bf16=True).precision == "bf16_mixed"
    assert TrainConfig(precision="bf16").bf16 is True
    cuda = torch.device("cuda")
    assert _uses_bf16_autocast(TrainConfig(bf16=True), cuda) is True
    assert _uses_bf16_autocast(TrainConfig(precision="bf16"), cuda) is False


def test_train_precision_rejects_invalid_or_conflicting_values():
    with pytest.raises(ValueError, match="unsupported training precision"):
        TrainConfig(precision="float16")
    with pytest.raises(ValueError, match="conflicts"):
        TrainConfig(precision="fp32", bf16=True)


def test_full_bf16_uses_bf16_model_gradients_and_optimizer_moments(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.precision = "bf16"
    model = _build_training_model(
        cfg, torch.device("cpu"), rank=0, use_fsdp=False
    )
    floating_tensors = [
        *model.parameters(),
        *(buffer for buffer in model.buffers() if buffer.is_floating_point()),
    ]
    assert floating_tensors
    assert {tensor.dtype for tensor in floating_tensors} == {torch.bfloat16}

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = _build_optimizer(params, cfg)
    x = torch.randint(0, cfg.model.vocab_size, (1, 4))
    y = torch.roll(x, shifts=-1, dims=1)
    loss = model(x, y)
    assert loss.dtype == torch.float32
    loss.backward()
    assert {param.grad.dtype for param in params if param.grad is not None} == {
        torch.bfloat16
    }
    optimizer.step()
    moment_dtypes = {
        value.dtype
        for state in optimizer.state.values()
        for key, value in state.items()
        if key in {"exp_avg", "exp_avg_sq"}
    }
    assert moment_dtypes == {torch.bfloat16}


def test_llama_fsdp_layer_lookup_does_not_import_qwen35(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcw.models.qwen3", None)

    classes = _transformer_layer_classes("llama")

    assert classes


def test_fsdp_nonzero_rank_builds_on_meta_device(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.precision = "bf16"

    model = _build_training_model(
        cfg, torch.device("cpu"), rank=1, use_fsdp=True
    )

    assert all(param.is_meta for param in model.parameters())
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}


def test_loader_uses_configured_batch_size_for_random_data(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.batch_size = 5
    cfg.sequence_len = 7

    x, y, state = next(_loader(cfg, torch.device("cpu")))

    assert state["kind"] == "random"
    assert x.shape == (5, 7)
    assert y.shape == (5, 7)


def test_validate_train_batch_rejects_batch_size_mismatch(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.batch_size = 4

    x = torch.zeros(3, cfg.sequence_len, dtype=torch.long)
    y = torch.zeros_like(x)

    try:
        _validate_train_batch(x, y, cfg)
    except ValueError as exc:
        assert "config.batch_size 4" in str(exc)
    else:
        raise AssertionError("expected batch size mismatch to fail")


def test_train_does_not_duplicate_final_checkpoint_when_save_every_hits_max_step(
    monkeypatch, tmp_path
):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.save_every = 1
    calls = []

    def fake_save(
        path, model, model_config, checkpoint_state, *, use_fsdp=False
    ):
        calls.append((path, checkpoint_state["step"], use_fsdp))

    monkeypatch.setattr("gcw.train._save_training_checkpoint", fake_save)

    train(cfg)

    assert [(path.name, step, use_fsdp) for path, step, use_fsdp in calls] == [
        ("checkpoint", 1, False),
        ("checkpoint", 2, False),
    ]


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


def test_train_can_retain_numbered_step_checkpoints(monkeypatch, tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.save_every = 1
    monkeypatch.setenv("GCW_KEEP_STEP_CHECKPOINTS", "1")
    monkeypatch.setenv("GCW_STEP_CHECKPOINT_EVERY", "2")

    train(cfg)

    assert (tmp_path / "run" / "checkpoint" / "train_state.json").exists()
    assert not (tmp_path / "run" / "checkpoint-1").exists()
    retained = tmp_path / "run" / "checkpoint-2" / "train_state.json"
    assert retained.exists()
    assert json.loads(retained.read_text(encoding="utf-8"))["step"] == 2


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
    assert torch.allclose(
        trained.backbone.embed_tokens.weight,
        torch.full_like(trained.backbone.embed_tokens.weight, 0.123),
    )


def test_checkpoint_metadata_records_hf_tokenizer_source(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.mode = "sft"
    cfg.data = DataConfig(
        kind="sft",
        dataset="/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl",
        tokenizer="hf",
        tokenizer_source="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model",
    )
    cfg.backbone_checkpoint = (
        "/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"
    )

    state = _checkpoint_metadata(cfg, {"step": 0, "world_size": 1})

    assert state["source"] == cfg.data.tokenizer_source
    assert state["tokenizer_source"] == cfg.data.tokenizer_source
    assert state["tokenizer"] == "hf"
    assert state["backbone_checkpoint"] == cfg.backbone_checkpoint


def test_checkpoint_state_drops_runtime_loader_tensors():
    state = _checkpoint_state(
        {"world_size": 1},
        5,
        {
            "kind": "sft",
            "row": 7,
            "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        },
    )

    assert state == {
        "world_size": 1,
        "step": 5,
        "loader": {"kind": "sft", "row": 7},
    }


def test_sft_batch_uses_next_token_targets():
    x, y = sft_batch(
        [{"prompt": "AB", "answer": "CD"}],
        TinyTokenizer(),
        max_length=8,
        response_prefix="\n",
    )

    assert x.tolist() == [[1, 2, 5, 3, 4, 0]]
    assert y.tolist() == [[-100, 5, 3, 4, 0, -100]]


def test_sft_batch_can_return_attention_mask():
    x, y, attention_mask = sft_batch(
        [{"prompt": "A", "answer": "B"}, {"prompt": "AB", "answer": "C"}],
        TinyTokenizer(),
        max_length=8,
        response_prefix="",
        return_attention_mask=True,
    )

    assert x.tolist() == [[1, 2, 0, 0], [1, 2, 3, 0]]
    assert y.tolist() == [[2, 0, -100, -100], [-100, 3, 0, -100]]
    assert attention_mask.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_model_loss_forwards_attention_mask():
    class MaskedLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_mask = None

        def forward(
            self, input_ids, targets, loss_reduction="mean", attention_mask=None
        ):
            self.seen_mask = attention_mask
            return input_ids.float().sum() + targets.float().sum()

    model = MaskedLossModel()
    x = torch.ones(1, 2, dtype=torch.long)
    y = torch.ones(1, 2, dtype=torch.long)
    attention_mask = torch.tensor([[True, False]])

    _model_loss(model, x, y, attention_mask=attention_mask)

    assert model.seen_mask is attention_mask
