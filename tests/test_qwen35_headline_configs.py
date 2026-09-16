"""Offline contracts for draft headline configs, not launch qualification."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml

from gcw.architecture import resolve_experiment, resolve_model
from gcw.architecture.execution import resolve_execution_plan
from gcw.config import load_train_config
from gcw.models import build_model
from gcw.models.backbones.qwen3.models import Qwen35TF
from gcw.models.backbones.transformer import RotaryCache
from gcw.models.lora import LoRALinear
from gcw.training.model_setup import _apply_trainable_policy, _model_dtype
from gcw.training.state import _save_training_state

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs/train/qwen35_headline"
ARMS = (
    "sft",
    "lora",
    "diffusion_h50_k4_vector_gate",
    "sft_stage1",
    "sft_then_sft",
    "sft_then_lora",
    "sft_then_diffusion_h50_k4_vector_gate",
)
LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
}
TARGET_FRACTION = 22_427_520 / 1_881_825_088
DATASET = (
    "/mnt/jfs-01/shared/gcw/data/" "OpenMathInstruct-2-headline-v1/train.jsonl"
)


@pytest.fixture
def meta_rotary(monkeypatch):
    # RotaryCache explicitly requests CPU even inside torch.device("meta").
    build = RotaryCache._build
    monkeypatch.setattr(
        RotaryCache,
        "_build",
        staticmethod(
            lambda seq_len, head_dim, theta, device: build(
                seq_len, head_dim, theta, torch.device("meta")
            )
        ),
    )


def test_headline_inventory_and_unique_outputs() -> None:
    paths = sorted(CONFIGS.glob("*/*.yaml"))
    assert {path.relative_to(CONFIGS).as_posix() for path in paths} == {
        f"{size}/{arm}.yaml" for size in ("2b", "4b", "9b") for arm in ARMS
    }
    configs = [load_train_config(path) for path in paths]
    assert len({config.output_dir for config in configs}) == 21
    assert {config.data.dataset for config in configs} == {DATASET}


@pytest.mark.parametrize("size", ["2b", "4b", "9b"])
@pytest.mark.parametrize("arm", ARMS)
def test_draft_protocol_and_stage_lineage(size: str, arm: str) -> None:
    path = CONFIGS / size / f"{arm}.yaml"
    raw = yaml.safe_load(path.read_text())
    assert {"model", "data", "metadata"} <= raw.keys()
    assert not {"extends", "include", "base_config"} & raw.keys()
    config = load_train_config(path)
    metadata = config.metadata
    assert metadata["protocol"] == "qwen35-headline-v2-draft"
    assert metadata["readiness"] == (
        "requires_prepared_data_backbones_and_two_gpu_qualification"
    )
    assert metadata["arm"] == arm
    assert metadata["world_size"] == 2
    assert metadata["seed_count"] == 1
    assert Path(config.output_dir).parent.name == "qwen35-headline-v2"
    assert metadata["prepared_training_rows"] == 1_024_000
    assert metadata["planned_dataset_source"] == "nvidia/OpenMathInstruct-2"
    assert metadata["planned_dataset_source_split"] == "train_2M"
    assert metadata["pretrained_source"] == f"Qwen/Qwen3.5-{size.upper()}"
    assert "headline-draft" in config.wandb.tags
    assert config.wandb.group == metadata["protocol"]

    assert config.mode == config.data.kind == "sft"
    assert config.precision == "bf16_mixed"
    assert _model_dtype(config) == torch.float32
    assert config.sequence_len == 2048
    assert config.batch_size == 1
    assert config.global_batch_rows == 128
    assert config.total_batch_size == 0
    micro_rows = config.batch_size * metadata["world_size"]
    assert divmod(config.global_batch_rows, micro_rows) == (64, 0)
    assert config.seed == 24
    assert config.data.dataset == DATASET
    assert config.data.tokenizer == "hf"
    assert config.data.tokenizer_source == metadata["pretrained_source"]
    assert config.data.sft_loss_scope == "answer_only"
    assert config.data.sft_chat_template is True
    assert config.data.sft_enable_thinking is False
    assert config.data.prompt_suffix == config.data.response_prefix == ""
    assert config.logical_pair_grouping is False
    assert config.truncated_bptt_windows == 0
    assert config.activation_checkpointing == "none"
    assert config.activation_checkpoint_windows == 0
    assert config.performance.adaptive_batching.enabled is False
    assert config.init_checkpoint is config.resume is None
    assert config.freeze_backbone is False
    assert config.save_cache_processor_only is False
    assert config.checkpoint_format == "native_safetensors"
    assert config.save_final is True

    full_sft = arm in {"sft", "sft_stage1", "sft_then_sft"}
    assert config.distributed == ("fsdp" if full_sft else "ddp")
    assert config.learning_rate == (1e-5 if full_sft else 1e-4)
    assert config.trainable == (
        "all" if full_sft else "lora" if "lora" in arm else "cache_processor"
    )
    stage1 = load_train_config(CONFIGS / size / "sft_stage1.yaml")
    if arm.startswith("sft_then_"):
        assert metadata["stage"] == "stage2"
        assert config.max_steps == 4000
        assert config.data.sft_skip_rows == 512_000
        assert config.data.sft_take_rows == 512_000
        assert Path(config.backbone_checkpoint) == (
            Path(stage1.output_dir) / "checkpoint"
        )
        assert metadata["parent_run"] == Path(stage1.output_dir).name
        assert config.data.sft_skip_rows == stage1.data.sft_take_rows
        assert config.max_steps + stage1.max_steps == 8000
    else:
        assert metadata["stage"] == (
            "stage1" if arm == "sft_stage1" else "direct"
        )
        assert config.max_steps == (4000 if arm == "sft_stage1" else 8000)
        assert config.data.sft_skip_rows == 0
        assert config.data.sft_take_rows == (
            512_000 if arm == "sft_stage1" else 1_024_000
        )
        assert config.backbone_checkpoint == stage1.backbone_checkpoint
        assert Path(config.backbone_checkpoint).name == (
            f"qwen35-{size}-tf-base-native"
        )
        assert metadata["parent_run"] is None
    assert config.max_steps * config.global_batch_rows == (
        config.data.sft_take_rows
    )
    assert config.data.sft_skip_rows + config.data.sft_take_rows <= (
        metadata["prepared_training_rows"]
    )

    model_resolution = resolve_model(config.model)
    assert model_resolution.report.compatible, model_resolution.report
    assert model_resolution.receipt is not None
    experiment = resolve_experiment(config, model_resolution.receipt)
    assert experiment.report.compatible, experiment.report
    assert experiment.receipt.objective_protocol == "gcw.sft.answer_only_ce.v1"
    assert resolve_execution_plan(config).strategy == "eager_reference"


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize(
    "size,hidden,intermediate,depth,q_heads,kv_heads,value_heads,tied,"
    "native_count,cp_hidden,cp_ff,cp_heads,cp_count,kv_selected,"
    "linear_selected,rank,lora_count",
    [
        (
            "2b",
            2048,
            6144,
            24,
            8,
            2,
            16,
            True,
            1_881_825_088,
            384,
            1536,
            3,
            22_427_520,
            3,
            72,
            21,
            22_075_200,
        ),
        (
            "4b",
            2560,
            9216,
            32,
            16,
            4,
            32,
            True,
            4_205_751_296,
            384,
            2720,
            3,
            50_173_824,
            8,
            192,
            25,
            50_726_400,
        ),
        (
            "9b",
            4096,
            12288,
            32,
            16,
            4,
            32,
            False,
            8_953_803_264,
            768,
            3168,
            6,
            106_619_904,
            8,
            192,
            39,
            105_490_944,
        ),
    ],
    ids=["2b", "4b", "9b"],
)
def test_native_meta_geometry_counts_and_trainable_ownership(
    arm,
    size,
    hidden,
    intermediate,
    depth,
    q_heads,
    kv_heads,
    value_heads,
    tied,
    native_count,
    cp_hidden,
    cp_ff,
    cp_heads,
    cp_count,
    kv_selected,
    linear_selected,
    rank,
    lora_count,
    meta_rotary,
) -> None:
    config = load_train_config(CONFIGS / size / f"{arm}.yaml")
    geometry = config.model
    assert geometry.family == "qwen3_5"
    assert geometry.hidden_size == hidden
    assert geometry.intermediate_size == intermediate
    assert geometry.num_layers == depth
    assert geometry.num_heads == q_heads
    assert geometry.num_kv_heads == kv_heads
    assert geometry.tie_embeddings is tied
    assert geometry.vocab_size == 248320
    assert geometry.sequence_len == 262144
    assert geometry.head_dim == 256
    assert geometry.hidden_act == "silu"
    assert geometry.norm_eps == 1e-6
    assert geometry.qk_norm is True
    assert geometry.rotary_style == "llama"
    assert geometry.rope_theta == 10_000_000
    assert geometry.partial_rotary_factor == 0.25
    assert geometry.linear_conv_kernel_dim == 4
    assert geometry.linear_key_head_dim == geometry.linear_value_head_dim == 128
    assert geometry.linear_num_key_heads == 16
    assert geometry.linear_num_value_heads == value_heads
    assert geometry.layer_types == [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ] * (depth // 4)
    assert geometry.processor_layer_indices == list(range(3, depth, 4))

    with torch.device("meta"):
        model = build_model(geometry).to(dtype=_model_dtype(config))
    assert all(
        p.is_meta and p.dtype == torch.float32 for p in model.parameters()
    )
    assert all(buffer.is_meta for buffer in model.buffers())
    backbone = getattr(model, "backbone", model)
    assert isinstance(backbone, Qwen35TF)
    assert len(backbone.layers) == depth
    assert backbone.embed_tokens.weight.shape == (248320, hidden)
    assert (backbone.lm_head is None) is tied
    if not tied:
        assert backbone.lm_head.weight.shape == (248320, hidden)
        assert backbone.lm_head.weight is not backbone.embed_tokens.weight
    native = {
        name: p
        for name, p in backbone.named_parameters()
        if name.rsplit(".", 1)[-1] not in {"lora_a", "lora_b"}
    }
    assert len({id(p) for p in native.values()}) == len(native)
    assert not any("vision" in name or "mtp" in name for name in native)
    assert sum(p.numel() for p in native.values()) == native_count
    assert config.metadata["backbone_text_parameters"] == native_count

    adapters = {
        name: module
        for name, module in backbone.named_modules()
        if isinstance(module, LoRALinear)
    }
    _apply_trainable_policy(model, config.trainable)
    if config.trainable == "all":
        assert geometry.variant == "tf"
        assert geometry.lora is geometry.cache_processor is None
        assert geometry.diffusion_processor is None
        assert not adapters
        assert all(p.requires_grad for p in native.values())
        expected_trainable = native_count
    elif config.trainable == "lora":
        assert geometry.variant == "tf"
        assert geometry.cache_processor is geometry.diffusion_processor is None
        assert geometry.lora.rank == rank
        assert geometry.lora.alpha == rank
        assert geometry.lora.dropout == 0
        assert len(geometry.lora.targets) == len(LORA_TARGETS)
        assert set(geometry.lora.targets) == LORA_TARGETS
        assert {name.rsplit(".", 1)[-1] for name in adapters} == LORA_TARGETS
        linear_projections = {
            name
            for name, module in backbone.named_modules()
            if isinstance(module, torch.nn.Linear) and name != "lm_head"
        }
        assert set(adapters) == linear_projections
        assert (
            len(adapters) == depth * 3 + (depth // 4) * 4 + (depth * 3 // 4) * 5
        )
        for adapter in adapters.values():
            assert type(adapter) is LoRALinear
            assert adapter.lora_a.shape == (rank, adapter.in_features)
            assert adapter.lora_b.shape == (adapter.out_features, rank)
            assert adapter.scaling == 1
            assert adapter.dropout == 0
        assert not any(p.requires_grad for p in native.values())
        assert all(
            p.requires_grad == name.endswith((".lora_a", ".lora_b"))
            for name, p in model.named_parameters()
        )
        expected_trainable = lora_count
        params_per_rank, remainder = divmod(lora_count, rank)
        assert remainder == 0
        assert all(
            abs(lora_count - cp_count)
            <= abs(neighbor * params_per_rank - cp_count)
            for neighbor in (rank - 1, rank + 1)
        )
    else:
        assert geometry.variant == "longshot"
        assert geometry.lora is None
        assert not adapters
        processor = geometry.cache_processor
        assert (
            processor.hidden_size,
            processor.intermediate_size,
            processor.num_heads,
            processor.num_layers,
        ) == (cp_hidden, cp_ff, cp_heads, 3)
        assert processor.num_kv_heads == cp_heads
        assert processor.topk_heads == kv_selected
        assert kv_selected * 4 == len(geometry.cache_layer_indices) * kv_heads
        assert processor.topk_linear_heads == linear_selected
        assert linear_selected * 4 == depth * 3 // 4 * value_heads
        assert processor.process_every == 64
        assert processor.topk == 32
        assert processor.head_selection_method == "magnitude"
        assert processor.encoder_method == "query_attention"
        assert processor.encoder_query_source == "final_token"
        assert processor.unrope_keys is True
        assert processor.update_stream == "kv"
        assert processor.linear_state_cache is True
        assert processor.linear_state_query_mode == "per_linear_state"
        assert processor.linear_state_update_mode == "outer_product"
        assert processor.gate_linear_state_delta is True
        assert processor.live_rank1_linear_writer is True
        assert processor.out_proj_init == "zero"
        assert processor.output_scale_mode == "learned"
        assert processor.update_gate_type == "contextual_vector"
        assert processor.gate_mode == "sigmoid"
        assert processor.gate_init == -2.0
        assert processor.iterations == 1
        assert processor.iteration_mode == "internal"
        block = model.cache_processor.block
        assert block.gate is None
        assert block.gate_dim == block.output_dim == 512
        assert block.out_proj.out_features == 1024
        assert block.diffusion.refinement_gate is None
        assert not any(
            p.numel() == 1 for p in model.cache_processor.parameters()
        )
        assert block.linear_state_adapter.output_dim == 256
        assert block.linear_state_adapter.gate_dim == 256
        assert block.linear_state_adapter.output_proj.out_features == 512
        assert len(block.linear_state_adapter.query_projs) == depth * 3 // 4
        assert (
            sum(p.numel() for p in model.cache_processor.parameters())
            == cp_count
        )
        fraction = cp_count / native_count
        assert fraction == pytest.approx(TARGET_FRACTION, rel=0.001, abs=0)
        assert config.metadata["target_processor_fraction"] == TARGET_FRACTION
        assert config.metadata["actual_processor_fraction"] == fraction
        assert config.metadata["role"] == "contextual_vector_writeback"
        assert not any(p.requires_grad for p in native.values())
        assert all(
            p.requires_grad == name.startswith("cache_processor.")
            for name, p in model.named_parameters()
        )
        expected_trainable = cp_count
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == (
        expected_trainable
    )
    assert sum(p.numel() for p in model.parameters()) == (
        native_count + (0 if config.trainable == "all" else expected_trainable)
    )
    assert config.metadata["trainable_parameters"] == expected_trainable


@pytest.mark.parametrize("size", ["2b", "4b", "9b"])
@pytest.mark.parametrize(
    "arm",
    ["diffusion_h50_k4_vector_gate", "sft_then_diffusion_h50_k4_vector_gate"],
)
def test_vector_writeback_h50_k4_meta_recurrence(
    size, arm, monkeypatch, meta_rotary
) -> None:
    config = load_train_config(CONFIGS / size / f"{arm}.yaml")
    diffusion = config.model.diffusion_processor
    assert diffusion.training_method == "scheduled"
    assert diffusion.scheduled_horizon == 50
    assert diffusion.num_steps == 4
    assert diffusion.scheduled_training_solver_steps == 4
    assert diffusion.scheduled_inference_solver_steps == 4
    assert diffusion.time_conditioning is True
    assert diffusion.noise_scale == 0.1
    assert diffusion.perturbation_enabled is True
    assert diffusion.inference_mode == "deterministic"
    with torch.device("meta"):
        model = build_model(config.model)
        bottleneck = torch.zeros(1, 2, config.model.cache_processor.hidden_size)
    block = model.cache_processor.block
    # Meta tensors have no values: trace delta/gate slices and live-key initialization.
    initializers = Mock()
    initializers.attach_mock(Mock(wraps=torch.nn.init.zeros_), "zero")
    initializers.attach_mock(Mock(wraps=torch.nn.init.normal_), "normal")
    initializers.attach_mock(Mock(wraps=torch.nn.init.constant_), "constant")
    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.init, "zeros_", initializers.zero)
        patch.setattr(torch.nn.init, "normal_", initializers.normal)
        patch.setattr(torch.nn.init, "constant_", initializers.constant)
        block.init_weights()
        adapter = block.linear_state_adapter
        adapter.init_weights(config.model.cache_processor)
    calls = initializers.mock_calls
    for projection, delta_dim in (
        (block.out_proj, block.output_dim),
        (adapter.output_proj, adapter.output_dim),
    ):
        zero_weights = [
            call.args[0]
            for call in calls
            if call[0] == "zero" and call.args[0]._base is projection.weight
        ]
        assert {tensor.storage_offset() for tensor in zero_weights} == {
            0,
            delta_dim * config.model.cache_processor.hidden_size,
        }
        assert all(
            tensor.shape
            == (delta_dim, config.model.cache_processor.hidden_size)
            for tensor in zero_weights
        )
        assert any(
            call[0] == "zero"
            and call.args[0]._base is projection.bias
            and call.args[0].storage_offset() == 0
            and call.args[0].shape == (delta_dim,)
            for call in calls
        )
        assert any(
            call[0] == "constant"
            and call.args[0]._base is projection.bias
            and call.args[0].storage_offset() == delta_dim
            and call.args[0].shape == (delta_dim,)
            and call.args[1] == -2.0
            for call in calls
        )
    assert calls[-1][0] == "normal"
    live_keys = calls[-1].args[0]
    assert live_keys._base is adapter.output_proj.weight
    assert live_keys.shape == (128, config.model.cache_processor.hidden_size)
    assert live_keys.storage_offset() == 0

    runtime = block.diffusion
    assert runtime.scheduled_sigma(0) == 0
    assert runtime.scheduled_sigma(50) == 0.1
    noise = Mock(wraps=torch.randn_like)
    monkeypatch.setattr(torch, "randn_like", noise)
    field = Mock(side_effect=lambda state, time: torch.zeros_like(state))
    runtime.train()
    result = runtime(bottleneck, field, start_level=50)
    assert result.is_meta
    noise.assert_called_once()
    assert field.call_count == runtime.pop_field_evaluation_count() == 4
    field.reset_mock()
    noise.reset_mock()
    runtime.eval()
    assert runtime._resolve_start_level(None) == 50
    result = runtime(bottleneck, field)
    assert result.is_meta
    noise.assert_not_called()
    assert field.call_count == runtime.pop_field_evaluation_count() == 4


@pytest.mark.parametrize("size", ["2b", "4b", "9b"])
@pytest.mark.parametrize("legacy_freeze", [False, True])
def test_cp_checkpoint_dispatch_preserves_complete_saves(
    size, legacy_freeze, monkeypatch, tmp_path
) -> None:
    config = load_train_config(
        CONFIGS / size / "diffusion_h50_k4_vector_gate.yaml"
    )
    config = replace(config, freeze_backbone=legacy_freeze)
    save = Mock()
    monkeypatch.setattr("gcw.training.state.save_training_checkpoint", save)
    model, optimizer, resume_state = Mock(), Mock(), Mock()
    destination = tmp_path / "checkpoint"
    _save_training_state(
        destination, model, optimizer, config, resume_state, {}, rank=0
    )
    save.assert_called_once_with(
        destination,
        model,
        config.model,
        optimizer,
        resume_state,
        {},
        **({"cache_processor_only": True} if legacy_freeze else {}),
    )
    assert not destination.exists()
