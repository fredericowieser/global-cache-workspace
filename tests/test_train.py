import json
import sys

import pytest
import torch

import gcw.training.core as train_module
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    FlowProcessorConfig,
    LoRAConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    WandbConfig,
)
from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.data.shards import ShardWriter
from gcw.data.training import build_training_stream
from gcw.data.sft import sft_batch
from gcw.models import build_model
from gcw.models.lora import LoRALinear
from gcw.training.core import (
    _build_training_model,
    _checkpoint_metadata,
    _checkpoint_state,
    _apply_trainable_policy,
    _load_initial_model,
    _model_loss,
    _transformer_layer_classes,
    _validate_train_batch,
    _wsd_lr,
    train,
)
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.checkpoint import save_training_checkpoint as real_save_training_checkpoint


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


def test_lora_trainable_policy_freezes_base_weights():
    config = ModelConfig(
        variant="tf",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        lora=LoRAConfig(rank=4),
    )
    model = build_model(config)

    _apply_trainable_policy(model, "lora")

    adapters = [module for module in model.modules() if isinstance(module, LoRALinear)]
    adapter_ids = {id(parameter) for module in adapters for parameter in (module.lora_a, module.lora_b)}
    assert adapters
    assert all(parameter.requires_grad == (id(parameter) in adapter_ids) for parameter in model.parameters())


def test_wsd_reaches_peak_then_exact_final_rate():
    scales = [_wsd_lr(step, 100, 3, 0.10, 0.0) for step in range(100)]

    assert scales[:3] == pytest.approx([1 / 3, 2 / 3, 1.0])
    assert scales[89] == 1.0
    assert scales[90] == 1.0
    assert scales[-1] == 0.0
    assert scales[90:] == sorted(scales[90:], reverse=True)


def test_trainable_policy_selects_backbone_processor_or_both():
    model = build_model(
        ModelConfig(
            variant="bgt",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(hidden_size=8, intermediate_size=16, num_heads=1),
        )
    )

    _apply_trainable_policy(model, "cache_processor")
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.cache_processor.parameters())

    _apply_trainable_policy(model, "backbone")
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())
    assert not any(parameter.requires_grad for parameter in model.cache_processor.parameters())

    _apply_trainable_policy(model, "all")
    assert all(parameter.requires_grad for parameter in model.parameters())


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


def test_train_writes_flow_pilot_gradient_and_runtime_audit(monkeypatch, tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.model = ModelConfig(
        variant="bgt",
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
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            encoder_seq_len=4,
        ),
        flow_processor=FlowProcessorConfig(num_steps=2),
    )
    cfg.trainable = "cache_processor"
    cfg.max_steps = 1
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.wandb.enabled = False
    audit_path = tmp_path / "pilot-audit.json"
    monkeypatch.setenv("GCW_TRAIN_AUDIT_OUT", str(audit_path))
    monkeypatch.setenv("GCW_AUDIT_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("GCW_AUDIT_MANIFEST_SHA256", "b" * 64)

    train(cfg)

    audit = json.loads(audit_path.read_text())
    assert audit["processor_grad_nonzero"] is True
    assert audit["refinement_gate_grad_nonzero"] is True
    assert audit["backbone_grad_tensors"] == 0
    assert audit["nonfinite_grad_tensors"] == 0
    assert audit["tokens_per_second"] > 0.0
    assert audit["peak_memory_mib"] >= 0.0
    assert audit["source_commit"] == "a" * 40
    assert audit["experiment_manifest_sha256"] == "b" * 64


def test_train_passes_explicit_wandb_resume_identity(monkeypatch, tmp_path):
    fake = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    config = tiny_train_config(tmp_path)
    config.wandb.id = "run-123"
    config.wandb.resume = "must"

    train(config)

    assert fake.init_kwargs["id"] == "run-123"
    assert fake.init_kwargs["resume"] == "must"

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
    assert 0.0 <= metrics["cache_processor/selected_head_overlap_after_iter_1"] <= 1.0


def test_train_logs_longshot_external_iteration_overlap_metrics(monkeypatch, tmp_path):
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
    assert 0.0 <= metrics["cache_processor/selected_head_overlap_after_iter_1"] <= 1.0


def test_llama_fsdp_layer_lookup_does_not_import_qwen35(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcw.models.backbones.qwen3", None)

    classes = _transformer_layer_classes("llama")

    assert classes


def test_fsdp_nonzero_rank_builds_on_meta_device(tmp_path):
    cfg = tiny_train_config(tmp_path)

    model = _build_training_model(cfg, torch.device("cpu"), rank=1, use_fsdp=True)

    assert all(param.is_meta for param in model.parameters())


def test_loader_uses_configured_batch_size_for_random_data(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.batch_size = 5
    cfg.sequence_len = 7

    batch = next(build_training_stream(cfg, torch.device("cpu"), rank=0, world_size=1))

    assert batch.x.shape == (5, 7)
    assert batch.y.shape == (5, 7)


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


def test_train_does_not_duplicate_final_checkpoint_when_save_every_hits_max_step(monkeypatch, tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.save_every = 1
    calls = []

    def fake_save(path, model, optimizer, config, resume_state, fingerprint, rank):
        calls.append((path, resume_state.step, rank))

    monkeypatch.setattr("gcw.training.core._save_training_state", fake_save)

    train(cfg)

    assert [(path.name, step, rank) for path, step, rank in calls] == [
        ("checkpoint", 1, 0),
        ("checkpoint", 2, 0),
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
    cfg.trainable = "cache_processor"
    cfg.backbone_checkpoint = str(checkpoint)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1

    train(cfg)

    trained, _, _ = load_checkpoint(tmp_path / "run" / "checkpoint")
    assert torch.allclose(trained.backbone.embed_tokens.weight, torch.full_like(trained.backbone.embed_tokens.weight, 0.123))


def test_complete_model_initialization_loads_backbone_and_processor(tmp_path):
    config = ModelConfig(
        variant="bgt",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(hidden_size=8, intermediate_size=16, num_heads=1),
    )
    source = build_model(config)
    source.init_weights()
    with torch.no_grad():
        source.cache_processor.block.gate.fill_(0.75)
    checkpoint = tmp_path / "complete-model"
    save_checkpoint(checkpoint, source, config)

    target = build_model(config)
    target.init_weights()
    _load_initial_model(target, checkpoint)

    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_checkpoint_metadata_records_hf_tokenizer_source(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.mode = "sft"
    cfg.data = DataConfig(
        kind="sft",
        dataset="/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl",
        tokenizer="hf",
        tokenizer_source="/models/llama-3.2-1b-tokenizer",
    )
    cfg.backbone_checkpoint = "/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"

    state = _checkpoint_metadata(cfg, {"step": 0, "world_size": 1})

    assert state["source"] == cfg.data.tokenizer_source
    assert state["tokenizer_source"] == cfg.data.tokenizer_source
    assert state["tokenizer"] == "hf"
    assert state["backbone_checkpoint"] == cfg.backbone_checkpoint
    assert state["admission"]["schema"] == "gcw.admission.v1"
    assert state["admission"]["model_assembly_receipt"]["receipt_sha256"]
    assert state["admission"]["composition_receipt"]["receipt_sha256"]


def test_checkpoint_state_drops_runtime_loader_tensors():
    state = _checkpoint_state(
        {"world_size": 1},
        5,
        {"kind": "sft", "row": 7, "attention_mask": torch.ones(1, 2, dtype=torch.bool)},
    )

    assert state == {"world_size": 1, "step": 5, "loader": {"kind": "sft", "row": 7}}


def test_sft_batch_uses_next_token_targets():
    x, y = sft_batch(
        [{"prompt": "AB", "answer": "CD"}],
        TinyTokenizer(),
        max_length=8,
        response_prefix="\n",
    )

    assert x.tolist() == [[1, 2, 5, 3, 4, 0]]
    assert y.tolist() == [[-100, 5, 3, 4, 0, -100]]


def test_interrupted_training_resumes_to_identical_parameters(monkeypatch, tmp_path):
    baseline = tiny_train_config(tmp_path)
    baseline.output_dir = str(tmp_path / "baseline")
    baseline.max_steps = 4
    baseline.save_every = 0
    baseline.eval_every = 0
    baseline.eval_on_start = False
    baseline.wandb.enabled = False
    train(baseline)

    interrupted = tiny_train_config(tmp_path)
    interrupted.output_dir = str(tmp_path / "resumed")
    interrupted.max_steps = 4
    interrupted.save_every = 2
    interrupted.eval_every = 0
    interrupted.eval_on_start = False
    interrupted.wandb.enabled = False

    def save_then_interrupt(
        path,
        model,
        model_config,
        optimizer,
        resume_state,
        training_fingerprint,
    ):
        real_save_training_checkpoint(
            path,
            model,
            model_config,
            optimizer,
            resume_state,
            training_fingerprint,
        )
        if resume_state.step == 2 and str(path).endswith("checkpoint"):
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(train_module, "save_training_checkpoint", save_then_interrupt, raising=False)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(interrupted)

    monkeypatch.setattr(train_module, "save_training_checkpoint", real_save_training_checkpoint)
    resumed = tiny_train_config(tmp_path)
    resumed.output_dir = str(tmp_path / "resumed")
    resumed.max_steps = 4
    resumed.save_every = 2
    resumed.eval_every = 0
    resumed.eval_on_start = False
    resumed.wandb.enabled = False
    resumed.resume = ResumeConfig(checkpoint=str(tmp_path / "resumed" / "checkpoint"))
    train(resumed)

    baseline_model, _, baseline_state = load_checkpoint(tmp_path / "baseline" / "checkpoint")
    resumed_model, _, resumed_state = load_checkpoint(tmp_path / "resumed" / "checkpoint")
    assert baseline_state["step"] == resumed_state["step"] == 4
    for expected, actual in zip(baseline_model.parameters(), resumed_model.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_resume_at_max_steps_exits_without_retraining(capsys, tmp_path):
    config = tiny_train_config(tmp_path)
    config.wandb.enabled = False
    config.eval_every = 0
    config.eval_on_start = False
    train(config)

    resumed = tiny_train_config(tmp_path)
    resumed.wandb.enabled = False
    resumed.eval_every = 0
    resumed.eval_on_start = False
    resumed.resume = ResumeConfig(checkpoint=str(tmp_path / "run" / "checkpoint"))
    train(resumed)

    assert "checkpoint already completed max_steps=2" in capsys.readouterr().out


def test_resume_rejects_changed_training_dynamics(tmp_path):
    config = tiny_train_config(tmp_path)
    config.wandb.enabled = False
    config.eval_every = 0
    config.eval_on_start = False
    train(config)

    resumed = tiny_train_config(tmp_path)
    resumed.wandb.enabled = False
    resumed.eval_every = 0
    resumed.eval_on_start = False
    resumed.learning_rate *= 2
    resumed.resume = ResumeConfig(checkpoint=str(tmp_path / "run" / "checkpoint"))

    with pytest.raises(CheckpointCompatibilityError, match="optimizer"):
        train(resumed)

def test_sft_chat_template_applies_suffix_to_message_rows():
    class ChatTokenizer:
        eot_id = 9

        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, **_kwargs):
            self.messages = messages
            return [1]

        @staticmethod
        def encode(_text):
            return [2]

    tokenizer = ChatTokenizer()
    sft_batch(
        [{"messages": [{"role": "user", "content": "problem"}, {"role": "assistant", "content": "answer"}]}],
        tokenizer,
        8,
        prompt_suffix="return boxed",
        chat_template=True,
    )
    assert tokenizer.messages == [{"role": "user", "content": "problem\n\nreturn boxed"}]


def test_sft_full_sequence_loss_includes_prompt_tokens():
    _, answer_only = sft_batch(
        [{"prompt": "AB", "answer": "CD"}],
        TinyTokenizer(),
        16,
        loss_scope="answer_only",
    )
    _, full_sequence = sft_batch(
        [{"prompt": "AB", "answer": "CD"}],
        TinyTokenizer(),
        16,
        loss_scope="full_sequence",
    )

    assert answer_only[0, 0].item() == -100
    assert full_sequence[0, 0].item() >= 0
    assert torch.equal(answer_only[0, -3:], full_sequence[0, -3:])


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
    assert attention_mask.tolist() == [[True, True, True, False], [True, True, True, True]]


def test_model_loss_forwards_attention_mask():
    class MaskedLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_mask = None

        def forward(self, input_ids, targets, loss_reduction="mean", attention_mask=None):
            self.seen_mask = attention_mask
            return input_ids.float().sum() + targets.float().sum()

    model = MaskedLossModel()
    x = torch.ones(1, 2, dtype=torch.long)
    y = torch.ones(1, 2, dtype=torch.long)
    attention_mask = torch.tensor([[True, False]])

    _model_loss(model, x, y, attention_mask=attention_mask)

    assert model.seen_mask is attention_mask
