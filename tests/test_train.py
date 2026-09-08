from dataclasses import replace
import json
import sys
from typing import get_args

import pytest
import torch

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
from gcw.performance.checkpointing import (
    transformer_layer_classes as _transformer_layer_classes,
)
from gcw.training.objectives import model_loss as _model_loss
from gcw.training.audit import (
    _checkpoint_metadata,
    _checkpoint_state,
    _validate_train_batch,
)
from gcw.training.distributed import _fsdp_wrap_classes
from gcw.training.model_setup import (
    _apply_trainable_policy,
    _build_training_model,
    _load_initial_model,
)
from gcw.training.optimizer import _build_optimizer, _wsd_lr
from gcw.training.precision import _uses_bf16_autocast
from gcw.training.core import _eval_loss, train
from gcw.training.checkpoint import CheckpointCompatibilityError
from gcw.training.checkpoint import (
    save_training_checkpoint as real_save_training_checkpoint,
)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("was_training", [False, True])
def test_validation_does_not_forward_training_truncation(
    tmp_path, grouped: bool, was_training: bool
) -> None:
    base = tiny_train_config(tmp_path)
    config = replace(
        base,
        model=replace(
            base.model,
            variant="bgt",
            cache_processor=CacheProcessorConfig(
                process_every=4,
                topk=2,
                topk_heads=2,
                hidden_size=16,
                intermediate_size=32,
                num_heads=2,
                num_kv_heads=1,
            ),
        ),
        mode="sft",
        data=DataConfig(kind="sft", dataset="unused.jsonl"),
        sequence_len=12,
        batch_size=2,
        eval_batches=1,
        logical_pair_grouping=grouped,
        global_batch_rows=2 if grouped else 0,
        truncated_bptt_windows=2,
    )
    model = build_model(config.model)
    model.init_weights()
    model.train(was_training)
    calls = []

    def check_forward(module, args, kwargs):
        assert not torch.is_grad_enabled() and not module.training
        assert kwargs.get("truncated_bptt_windows", 0) == 0
        calls.append(kwargs)

    with model.register_forward_pre_hook(check_forward, with_kwargs=True):
        torch.manual_seed(41)
        actual = _eval_loss(model, config, torch.device("cpu"))
        assert model.training == was_training
        torch.manual_seed(41)
        expected = _eval_loss(
            model,
            replace(config, truncated_bptt_windows=0),
            torch.device("cpu"),
        )
    assert len(calls) == 2
    assert actual == expected and model.training == was_training


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

    adapters = [
        module for module in model.modules() if isinstance(module, LoRALinear)
    ]
    adapter_ids = {
        id(parameter)
        for module in adapters
        for parameter in (module.lora_a, module.lora_b)
    }
    assert adapters
    assert all(
        parameter.requires_grad == (id(parameter) in adapter_ids)
        for parameter in model.parameters()
    )


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
            cache_processor=CacheProcessorConfig(
                hidden_size=8, intermediate_size=16, num_heads=1
            ),
        )
    )

    _apply_trainable_policy(model, "cache_processor")
    assert not any(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.cache_processor.parameters()
    )

    _apply_trainable_policy(model, "backbone")
    assert all(
        parameter.requires_grad for parameter in model.backbone.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.cache_processor.parameters()
    )

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


def test_train_writes_flow_pilot_gradient_and_runtime_audit(
    monkeypatch, tmp_path
):
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


def test_full_bf16_training_run_keeps_bf16_parameters(tmp_path):
    from safetensors.torch import load_file

    cfg = tiny_train_config(tmp_path)
    cfg.precision = "bf16"
    cfg.wandb = WandbConfig(enabled=False)
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.save_final = True

    train(cfg)

    weights = load_file(tmp_path / "run" / "checkpoint" / "model.safetensors")
    assert weights
    assert {tensor.dtype for tensor in weights.values()} == {torch.bfloat16}


@pytest.mark.parametrize("precision", ["fp32", "bf16_mixed"])
def test_mixed_and_fp32_precisions_keep_float32_parameters(tmp_path, precision):
    cfg = tiny_train_config(tmp_path)
    cfg.precision = precision
    model = _build_training_model(
        cfg, torch.device("cpu"), rank=0, use_fsdp=False
    )
    assert {param.dtype for param in model.parameters()} == {torch.float32}


@pytest.mark.parametrize("use_fsdp", [False, True])
def test_full_bf16_draws_initial_weights_in_float32_before_casting(
    tmp_path, use_fsdp
):
    cfg = tiny_train_config(tmp_path)
    cfg.precision = "bf16"

    torch.manual_seed(cfg.seed)
    built = _build_training_model(
        cfg, torch.device("cpu"), rank=0, use_fsdp=use_fsdp
    )

    torch.manual_seed(cfg.seed)
    reference = build_model(cfg.model)
    reference.init_weights()
    reference = reference.to(dtype=torch.bfloat16)

    built_state = built.state_dict()
    reference_state = reference.state_dict()
    assert built_state.keys() == reference_state.keys()
    assert all(
        torch.equal(built_state[key], tensor)
        for key, tensor in reference_state.items()
    )


def test_supported_precisions_match_the_declared_type():
    from gcw.config.types import TrainingPrecision

    assert set(get_args(TrainingPrecision)) == {"fp32", "bf16_mixed", "bf16"}
    for precision in get_args(TrainingPrecision):
        assert TrainConfig(precision=precision).precision == precision
    for removed in ("auto", "fp16", "fp16_mixed"):
        with pytest.raises(ValueError, match="unsupported training precision"):
            TrainConfig(precision=removed)


@pytest.mark.parametrize(
    ("precision", "expected"),
    [("fp32", "float32"), ("bf16_mixed", "float32"), ("bf16", "bfloat16")],
)
def test_checkpoint_metadata_records_precision_and_the_real_parameter_dtype(
    tmp_path, precision, expected
):
    from gcw.training.audit import _checkpoint_metadata

    cfg = tiny_train_config(tmp_path)
    cfg.precision = precision
    metadata = _checkpoint_metadata(cfg, {}, admission={})

    assert metadata["precision"] == precision
    assert metadata["dtype"] == expected


def test_training_fingerprint_records_precision_and_separates_the_bf16_modes(
    tmp_path,
):
    from gcw.training.audit import _training_fingerprint

    fingerprints = {}
    for precision in ("fp32", "bf16_mixed", "bf16"):
        cfg = tiny_train_config(tmp_path)
        cfg.precision = precision
        fingerprints[precision] = _training_fingerprint(cfg, 1, 8, 8, 1)
        assert fingerprints[precision]["precision"] == precision
        assert "bf16" not in fingerprints[precision]

    assert fingerprints["bf16"] != fingerprints["bf16_mixed"]
    assert fingerprints["fp32"] != fingerprints["bf16_mixed"]


def test_precision_summary_reports_autocast_a_device_cannot_run():
    from gcw.training.precision import _precision_summary

    cfg = TrainConfig(precision="bf16_mixed")
    summary = _precision_summary(cfg, torch.device("cpu"))

    assert "parameters float32" in summary
    assert "autocast disabled on cpu" in summary


def test_llama_fsdp_layer_lookup_does_not_import_qwen35(monkeypatch):
    monkeypatch.setitem(sys.modules, "gcw.models.backbones.qwen3", None)

    classes = _transformer_layer_classes("llama")

    assert classes


def test_qwen35_fsdp_wraps_gated_delta_dynamics_separately():
    from gcw.models.backbones.qwen3 import Qwen35GatedDeltaDynamics

    default_classes = _transformer_layer_classes("qwen3_5")
    fsdp_classes = _fsdp_wrap_classes("qwen3_5")

    assert Qwen35GatedDeltaDynamics not in default_classes
    assert Qwen35GatedDeltaDynamics in fsdp_classes


def test_fsdp_nonzero_rank_builds_on_meta_device(tmp_path):
    cfg = tiny_train_config(tmp_path)

    model = _build_training_model(
        cfg, torch.device("cpu"), rank=1, use_fsdp=True
    )

    assert all(param.is_meta for param in model.parameters())


def test_loader_uses_configured_batch_size_for_random_data(tmp_path):
    cfg = tiny_train_config(tmp_path)
    cfg.batch_size = 5
    cfg.sequence_len = 7

    batch = next(
        build_training_stream(cfg, torch.device("cpu"), rank=0, world_size=1)
    )

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
        path, model, optimizer, config, resume_state, fingerprint, rank
    ):
        calls.append((path, resume_state.step, rank))

    monkeypatch.setattr("gcw.training.state._save_training_state", fake_save)

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
    cfg.freeze_backbone = True
    cfg.backbone_checkpoint = str(checkpoint)
    cfg.wandb.enabled = False
    cfg.eval_every = 0
    cfg.eval_on_start = False
    cfg.max_steps = 1

    train(cfg)

    from safetensors.torch import load_file

    trained_checkpoint = tmp_path / "run" / "checkpoint"
    weights = load_file(trained_checkpoint / "model.safetensors")
    assert weights
    assert all(key.startswith("cache_processor.") for key in weights)
    trained, _, _ = load_checkpoint(trained_checkpoint)
    trained_state = trained.state_dict()
    assert all(
        torch.equal(trained_state[key], tensor)
        for key, tensor in weights.items()
    )
    assert torch.allclose(
        trained.backbone.embed_tokens.weight,
        torch.full_like(trained.backbone.embed_tokens.weight, 0.123),
    )


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
        cache_processor=CacheProcessorConfig(
            hidden_size=8, intermediate_size=16, num_heads=1
        ),
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

    for expected, actual in zip(
        source.parameters(), target.parameters(), strict=True
    ):
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
    cfg.backbone_checkpoint = (
        "/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"
    )

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


def test_interrupted_training_resumes_to_identical_parameters(
    monkeypatch, tmp_path
):
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

    monkeypatch.setattr(
        "gcw.training.state.save_training_checkpoint",
        save_then_interrupt,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(interrupted)

    monkeypatch.setattr(
        "gcw.training.state.save_training_checkpoint",
        real_save_training_checkpoint,
    )
    resumed = tiny_train_config(tmp_path)
    resumed.output_dir = str(tmp_path / "resumed")
    resumed.max_steps = 4
    resumed.save_every = 2
    resumed.eval_every = 0
    resumed.eval_on_start = False
    resumed.wandb.enabled = False
    resumed.resume = ResumeConfig(
        checkpoint=str(tmp_path / "resumed" / "checkpoint")
    )
    train(resumed)

    baseline_model, _, baseline_state = load_checkpoint(
        tmp_path / "baseline" / "checkpoint"
    )
    resumed_model, _, resumed_state = load_checkpoint(
        tmp_path / "resumed" / "checkpoint"
    )
    assert baseline_state["step"] == resumed_state["step"] == 4
    for expected, actual in zip(
        baseline_model.parameters(), resumed_model.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)


def test_resume_with_backbone_checkpoint_preserves_fingerprint_and_continues(
    monkeypatch, tmp_path
):
    # A resume config must retain the original backbone_checkpoint so the
    # training fingerprint matches, while the architecture validation allows
    # backbone_checkpoint + resume together (resume wins model restoration).
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
    backbone_checkpoint = tmp_path / "backbone"
    save_checkpoint(backbone_checkpoint, backbone, backbone_cfg)

    def bt_config(output_dir: str) -> TrainConfig:
        config = tiny_train_config(tmp_path)
        config.model = ModelConfig(
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
        config.trainable = "cache_processor"
        config.backbone_checkpoint = str(backbone_checkpoint)
        config.output_dir = output_dir
        config.max_steps = 4
        config.save_every = 2
        config.eval_every = 0
        config.eval_on_start = False
        config.wandb.enabled = False
        return config

    first = bt_config(str(tmp_path / "run"))
    monkeypatch.setenv("GCW_STOP_AFTER_CHECKPOINT_STEP", "2")
    train(first)
    monkeypatch.delenv("GCW_STOP_AFTER_CHECKPOINT_STEP")
    assert (
        json.loads(
            (tmp_path / "run" / "checkpoint" / "train_state.json").read_text()
        )["step"]
        == 2
    )

    resumed = bt_config(str(tmp_path / "run"))
    resumed.resume = ResumeConfig(
        checkpoint=str(tmp_path / "run" / "checkpoint")
    )
    train(resumed)
    assert (
        json.loads(
            (tmp_path / "run" / "checkpoint" / "train_state.json").read_text()
        )["step"]
        == 4
    )


def test_training_can_stop_cleanly_at_an_authenticated_checkpoint(
    monkeypatch, tmp_path
):
    config = tiny_train_config(tmp_path)
    config.max_steps = 4
    config.save_every = 2
    config.eval_every = 0
    config.eval_on_start = False
    config.wandb.enabled = False
    monkeypatch.setenv("GCW_STOP_AFTER_CHECKPOINT_STEP", "3")

    train(config)

    checkpoint = tmp_path / "run" / "checkpoint"
    assert (
        json.loads((checkpoint / "manifest.json").read_text())["complete"]
        is True
    )
    assert (
        json.loads((checkpoint / "train_state.json").read_text())["step"] == 3
    )

    resumed = tiny_train_config(tmp_path)
    resumed.max_steps = 4
    resumed.save_every = 2
    resumed.eval_every = 0
    resumed.eval_on_start = False
    resumed.wandb.enabled = False
    resumed.resume = ResumeConfig(checkpoint=str(checkpoint))
    train(resumed)
    assert (
        json.loads((checkpoint / "train_state.json").read_text())["step"] == 3
    )


def test_checkpoint_stop_policy_rejects_malformed_step_before_training(
    monkeypatch, tmp_path
):
    config = tiny_train_config(tmp_path)
    config.wandb.enabled = False
    config.save_every = 1
    monkeypatch.setenv("GCW_STOP_AFTER_CHECKPOINT_STEP", "not-an-integer")

    with pytest.raises(ValueError, match="must be an integer"):
        train(config)


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
    resumed.resume = ResumeConfig(
        checkpoint=str(tmp_path / "run" / "checkpoint")
    )
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
    resumed.resume = ResumeConfig(
        checkpoint=str(tmp_path / "run" / "checkpoint")
    )

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
        [
            {
                "messages": [
                    {"role": "user", "content": "problem"},
                    {"role": "assistant", "content": "answer"},
                ]
            }
        ],
        tokenizer,
        8,
        prompt_suffix="return boxed",
        chat_template=True,
    )
    assert tokenizer.messages == [
        {"role": "user", "content": "problem\n\nreturn boxed"}
    ]


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
    assert attention_mask.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_model_loss_forwards_attention_mask():
    class MaskedLossModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen_mask = None
            self.seen_loss_backend = None

        def forward(
            self,
            input_ids,
            targets,
            loss_reduction="mean",
            attention_mask=None,
            loss_backend="reference",
        ):
            self.seen_mask = attention_mask
            self.seen_loss_backend = loss_backend
            return input_ids.float().sum() + targets.float().sum()

    model = MaskedLossModel()
    x = torch.ones(1, 2, dtype=torch.long)
    y = torch.ones(1, 2, dtype=torch.long)
    attention_mask = torch.tensor([[True, False]])

    _model_loss(model, x, y, attention_mask=attention_mask)

    assert model.seen_mask is attention_mask
    assert model.seen_loss_backend == "reference"
