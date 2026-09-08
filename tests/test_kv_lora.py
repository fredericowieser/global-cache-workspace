"""KV-only adapter isolation, matched initialization and joint ownership."""

from dataclasses import replace

import pytest
import torch
from torch import nn

from gcw.config import (
    CacheProcessorConfig,
    LoRAConfig,
    ModelConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.models.lora import KVLoRALinear, LoRALinear, apply_lora
from gcw.training.model_setup import _apply_trainable_policy


def config():
    return ModelConfig(
        family="qwen3_5",
        variant="longshot",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=128,
        layer_types=["linear_attention", "full_attention"],
        processor_layer_indices=[1],
        partial_rotary_factor=0.5,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        cache_processor=CacheProcessorConfig(
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
            num_layers=1,
            process_every=64,
            topk=8,
            topk_heads=1,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            score="kv_norm",
            unrope_keys=True,
            out_proj_init="zero",
            gradient_checkpointing=False,
        ),
    )


def kv_config(rank=4):
    return LoRAConfig(
        rank=rank,
        alpha=rank,
        dropout=0,
        targets=("k_proj", "v_proj", "in_proj_kv"),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_adapters_never_modify_query_outputs(dtype):
    torch.manual_seed(24)
    base = nn.Linear(12, 22, bias=False, dtype=dtype)
    m = KVLoRALinear(base, kv_config(), 6, 10)
    x = torch.randn(2, 3, 12, dtype=dtype)
    assert torch.equal(m(x), base(x))
    with torch.no_grad():
        m.lora_b.normal_(0, 0.1)
    actual = m(x)
    assert torch.equal(actual[..., :6], base(x)[..., :6])
    assert not torch.equal(actual[..., 6:12], base(x)[..., 6:12])
    assert not torch.equal(actual[..., 12:], base(x)[..., 12:])
    m.weight.requires_grad_(False)
    actual.float().square().mean().backward()
    for p in (m.lora_a, m.lora_b):
        assert p.grad is not None and torch.isfinite(p.grad).all()
        assert all(torch.count_nonzero(g) for g in p.grad.chunk(2, dim=0))
    m.zero_grad()
    m(x)[..., :6].float().sum().backward()
    assert torch.count_nonzero(m.lora_a.grad) == 0
    assert torch.count_nonzero(m.lora_b.grad) == 0
    assert m.weight.grad is None


def test_joint_model_preserves_all_shared_weights_rng_and_noop():
    torch.manual_seed(24)
    baseline = build_model(config())
    baseline.init_weights()
    expected_rng = torch.random.get_rng_state()
    torch.manual_seed(24)
    joint = build_model(replace(config(), lora=kv_config()))
    joint.init_weights()
    assert torch.equal(expected_rng, torch.random.get_rng_state())
    for name, tensor in baseline.state_dict().items():
        assert torch.equal(tensor, joint.state_dict()[name]), name
    adapters = {
        n: m for n, m in joint.named_modules() if isinstance(m, LoRALinear)
    }
    assert len(adapters) == 3
    assert sum(isinstance(m, KVLoRALinear) for m in adapters.values()) == 1
    assert all(
        n.startswith("backbone.")
        and n.endswith(("k_proj", "v_proj", "in_proj_qkv"))
        for n in adapters
    )
    _apply_trainable_policy(joint, "lora_and_cache_processor")
    for n, p in joint.named_parameters():
        assert p.requires_grad == (
            n.startswith("cache_processor.")
            or n.endswith((".lora_a", ".lora_b"))
        ), n
    ids = torch.randint(0, 64, (1, 70))
    baseline.eval()
    joint.eval()
    with torch.no_grad():
        assert torch.equal(baseline(ids), joint(ids))
    opt = torch.optim.AdamW(
        (p for p in joint.parameters() if p.requires_grad),
        lr=1e-4,
        weight_decay=0,
    )
    before = {n: p.detach().clone() for n, p in joint.named_parameters()}
    for _ in range(2):
        opt.zero_grad()
        joint(ids, targets=ids).backward()
        opt.step()
    for n, p in joint.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, before[n]), n
    assert all(torch.count_nonzero(m.lora_b) for m in adapters.values())
    assert all(
        not torch.equal(m.lora_a, before[n + ".lora_a"])
        for n, m in adapters.items()
    )
    assert any(
        not torch.equal(p, before[n])
        for n, p in joint.named_parameters()
        if n.startswith("cache_processor.")
    )
    original = {
        n: id(m)
        for n, m in joint.backbone.named_modules()
        if isinstance(m, LoRALinear)
    }
    apply_lora(joint.backbone, kv_config())
    assert original == {
        n: id(m)
        for n, m in joint.backbone.named_modules()
        if isinstance(m, LoRALinear)
    }


def test_kv_adapters_compose_with_scheduled_diffusion():
    from gcw.config import DiffusionProcessorConfig

    cfg = replace(
        config(),
        lora=kv_config(),
        diffusion_processor=DiffusionProcessorConfig(
            training_method="scheduled",
            schedule_horizon=50,
            training_solver_steps=2,
            noise_scale=0.1,
        ),
    )
    model = build_model(cfg)
    model.init_weights()
    _apply_trainable_policy(model, "lora_and_cache_processor")
    tokens = torch.randint(0, cfg.vocab_size, (1, 70))
    model(tokens, tokens).backward()
    for module in model.modules():
        if isinstance(module, LoRALinear):
            assert module.lora_b.grad is not None
            assert torch.isfinite(module.lora_b.grad).all()
            assert torch.count_nonzero(module.lora_b.grad) > 0
    assert (
        torch.count_nonzero(model.cache_processor.block.out_proj.weight.grad)
        > 0
    )


def test_plain_lora_receipt_matches_adapter_only_ownership():
    from gcw.architecture import resolve_experiment, resolve_model
    from gcw.config import DataConfig

    cfg = replace(config(), lora=kv_config())
    train = TrainConfig(
        model=cfg, data=DataConfig(kind="random"), trainable="lora"
    )
    _, receipt = resolve_model(cfg).require()
    resolved, _ = resolve_experiment(train, receipt).require()
    assert resolved.parameter_policy == "lora_only"
    model = build_model(cfg)
    _apply_trainable_policy(model, "lora")
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == name.endswith(
            (".lora_a", ".lora_b")
        ), name


def test_runtime_count_includes_adapters_but_preserves_core_estimate():
    from gcw.params import (
        runtime_non_embedding_params,
        total_non_embedding_params,
        trainable_adapter_params,
    )

    for lora in (None, kv_config()):
        cfg = replace(config(), lora=lora)
        m = build_model(cfg)
        base = total_non_embedding_params(cfg)
        assert runtime_non_embedding_params(
            cfg, m
        ) == base + trainable_adapter_params(m)
        assert (runtime_non_embedding_params(cfg, m) > base) == (
            lora is not None
        )


def test_virtual_target_rejects_ambiguous_or_absent_geometry():
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_lora(
            nn.Linear(3, 4), LoRAConfig(targets=("in_proj_qkv", "in_proj_kv"))
        )
    with pytest.raises(ValueError, match="matched no"):
        apply_lora(nn.Linear(3, 4), kv_config())
    with pytest.raises(ValueError, match="geometry"):
        KVLoRALinear(nn.Linear(3, 8), kv_config(), 2, 3)
    parent = nn.ModuleDict({"in_proj_qkv": nn.Linear(3, 8)})
    with pytest.raises(ValueError, match="geometry"):
        apply_lora(parent, kv_config())


def test_explicit_full_fused_target_remains_ordinary_lora():
    parent = nn.ModuleDict({"in_proj_qkv": nn.Linear(3, 8)})
    apply_lora(parent, LoRAConfig(rank=2, targets=("in_proj_qkv",)))
    adapter = parent["in_proj_qkv"]
    assert type(adapter) is LoRALinear
    assert adapter.lora_a.shape == (2, 3)
    assert adapter.lora_b.shape == (8, 2)


@pytest.mark.parametrize("key_dim,value_dim", [(0, 4), (-1, 6), (2, 0)])
def test_invalid_native_dimensions(key_dim, value_dim):
    with pytest.raises(ValueError, match="geometry"):
        KVLoRALinear(nn.Linear(3, 4), kv_config(), key_dim, value_dim)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_fused_adapter_retains_base_parameters_device_dtype_and_names(device):
    base = nn.Linear(12, 22, bias=True, device=device, dtype=torch.bfloat16)
    base.requires_grad_(False)
    adapter = KVLoRALinear(base, kv_config(), 6, 10)
    assert adapter.weight is base.weight
    assert adapter.bias is base.bias
    assert not adapter.weight.requires_grad and not adapter.bias.requires_grad
    assert adapter.lora_a.requires_grad and adapter.lora_b.requires_grad
    assert set(adapter.state_dict()) == {"weight", "bias", "lora_a", "lora_b"}
    assert all(p.device.type == device for p in adapter.parameters())
    assert all(p.dtype == torch.bfloat16 for p in adapter.parameters())
    assert adapter.lora_a.shape == (8, 12)
    assert adapter.lora_b.shape == (16, 4)


def test_fused_key_and_value_factors_are_independent():
    base = nn.Linear(12, 22, bias=True)
    adapter = KVLoRALinear(base, kv_config(), 6, 10)
    x = torch.randn(2, 3, 12)
    with torch.no_grad():
        adapter.lora_b[:6].normal_()
    output = adapter(x)
    assert torch.equal(output[..., :6], base(x)[..., :6])
    assert not torch.equal(output[..., 6:12], base(x)[..., 6:12])
    assert torch.equal(output[..., 12:], base(x)[..., 12:])
    output[..., 6:12].square().sum().backward()
    assert torch.count_nonzero(adapter.lora_a.grad[:4])
    assert not torch.count_nonzero(adapter.lora_a.grad[4:])
    assert not torch.count_nonzero(adapter.lora_b.grad[6:])
    with torch.no_grad():
        adapter.lora_b[:6].zero_()
        adapter.lora_b[6:].normal_()
    output = adapter(x)
    assert torch.equal(output[..., :12], base(x)[..., :12])
    assert not torch.equal(output[..., 12:], base(x)[..., 12:])


def test_ordinary_targets_preserve_initialization_rng_and_idempotence():
    import copy
    import math

    cfg = LoRAConfig(rank=2)
    assert cfg.targets == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    base = nn.ModuleDict(
        {
            name: nn.Linear(4, 6)
            for name in (*cfg.targets, "in_proj_qkv", "other")
        }
    )
    expected = copy.deepcopy(base)
    actual = copy.deepcopy(base)
    torch.manual_seed(123)
    # Historical constructor operations, including the discarded Linear init.
    for name in cfg.targets:
        child = expected[name]
        nn.Linear(4, 6)
        child.lora_a = nn.Parameter(torch.empty(2, 4))
        child.lora_b = nn.Parameter(torch.zeros(6, 2))
        nn.init.kaiming_uniform_(child.lora_a, a=math.sqrt(5))
    rng = torch.random.get_rng_state()
    torch.manual_seed(123)
    assert apply_lora(actual, cfg) is actual
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[name]), name
    assert {
        name for name, child in actual.items() if isinstance(child, LoRALinear)
    } == set(cfg.targets)
    identities = {name: id(child) for name, child in actual.items()}
    apply_lora(actual, cfg)
    assert identities == {name: id(child) for name, child in actual.items()}
    assert torch.equal(rng, torch.random.get_rng_state())
    assert apply_lora(base, None) is base


@pytest.mark.parametrize("lora", [kv_config(), LoRAConfig(rank=2)])
def test_joint_policy_admission_and_full_checkpoint_roundtrip(tmp_path, lora):
    from gcw.architecture import resolve_experiment, resolve_model
    from gcw.checkpoint import load_checkpoint, save_checkpoint
    from gcw.config import DataConfig, to_dict
    from gcw.training.model_setup import _load_initial_model

    cfg = replace(config(), lora=lora)
    train_cfg = TrainConfig(
        model=cfg,
        data=DataConfig(kind="random"),
        trainable="lora_and_cache_processor",
    )
    _, receipt = resolve_model(cfg).require()
    resolved, _ = resolve_experiment(train_cfg, receipt).require()
    assert resolved.parameter_policy == "lora_and_processor"
    model = build_model(cfg)
    model.init_weights()
    _apply_trainable_policy(model, train_cfg.trainable)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (
            name.startswith("cache_processor.")
            or name.endswith((".lora_a", ".lora_b"))
        ), name
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_b.normal_(0, 0.1)
    path = tmp_path / "full"
    save_checkpoint(path, model, cfg)
    loaded, loaded_cfg, _ = load_checkpoint(path)
    assert to_dict(loaded_cfg) == to_dict(cfg)
    target = build_model(cfg)
    _load_initial_model(target, path)
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
        assert torch.equal(tensor, target.state_dict()[name]), name
    ids = torch.randint(0, 64, (1, 70))
    model.eval()
    loaded.eval()
    with torch.no_grad():
        assert torch.equal(model(ids), loaded(ids))


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"freeze_backbone": True}, "freeze_backbone=false"),
        (
            {"save_cache_processor_only": True},
            "save_cache_processor_only=false",
        ),
        ({"model": config()}, "requires model.lora"),
        (
            {"model": ModelConfig(lora=LoRAConfig())},
            "requires a cache processor",
        ),
    ],
)
def test_joint_policy_rejects_incomplete_or_lossy_configuration(
    overrides, message
):
    kwargs = dict(
        model=replace(config(), lora=kv_config()),
        trainable="lora_and_cache_processor",
    )
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        TrainConfig(**kwargs)


def test_runtime_assembly_counts_adapters_and_discloses_missing_flops(
    tmp_path, monkeypatch
):
    from gcw.architecture import resolve_execution_plan
    from gcw.config import DataConfig
    from gcw.params import (
        runtime_non_embedding_params,
        trainable_adapter_params,
    )
    from gcw.training import assembly

    monkeypatch.setattr(
        assembly, "init_distributed", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        assembly.RunTracker, "for_training", lambda *args, **kwargs: None
    )
    cfg = TrainConfig(
        model=replace(config(), lora=kv_config()),
        data=DataConfig(kind="random"),
        trainable="lora_and_cache_processor",
        distributed="none",
        output_dir=str(tmp_path / "run"),
        sequence_len=70,
        batch_size=1,
        max_steps=1,
    )
    plan = assembly.assemble_training_execution(
        cfg, resolve_execution_plan(cfg)
    )
    assert plan is not None
    assert trainable_adapter_params(plan.model) > 0
    assert plan.run_meta[
        "non_embedding_params"
    ] == runtime_non_embedding_params(cfg.model, plan.model)
    assert plan.run_meta["adapter_flops_included"] is False
    assert {
        id(p) for group in plan.opt.param_groups for p in group["params"]
    } == {id(p) for p in plan.model.parameters() if p.requires_grad}


def test_backbone_checkpoint_keeps_zero_initialized_kv_adapters(tmp_path):
    from gcw.checkpoint import load_backbone_checkpoint, save_checkpoint

    cfg = config()
    baseline = build_model(cfg)
    baseline.init_weights()
    save_checkpoint(tmp_path / "base", baseline, cfg)
    adapted = build_model(replace(cfg, lora=kv_config()))
    adapted.init_weights()
    before = {
        n: p.detach().clone()
        for n, p in adapted.named_parameters()
        if n.endswith((".lora_a", ".lora_b"))
    }
    load_backbone_checkpoint(adapted, tmp_path / "base")
    for name, value in baseline.backbone.state_dict().items():
        assert torch.equal(value, adapted.backbone.state_dict()[name]), name
    for name, value in before.items():
        assert torch.equal(value, adapted.state_dict()[name]), name


def test_joint_runtime_policy_requires_both_components():
    with pytest.raises(ValueError, match="requires attached LoRA"):
        _apply_trainable_policy(
            build_model(config()), "lora_and_cache_processor"
        )
    model = build_model(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            lora=LoRAConfig(rank=2),
        )
    )
    with pytest.raises(ValueError, match="requires a cache processor"):
        _apply_trainable_policy(model, "lora_and_cache_processor")
