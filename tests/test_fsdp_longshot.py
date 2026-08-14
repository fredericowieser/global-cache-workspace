from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gcw.config import CacheProcessorConfig, ModelConfig, TrainConfig
from gcw.train import (
    _apply_activation_checkpointing,
    _build_training_model,
    _unwrap,
    _wrap_fsdp,
)


def _tiny_qwen35_longshot_config() -> ModelConfig:
    return ModelConfig(
        variant="longshot",
        family="qwen3_5",
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ],
        processor_layer_indices=[1, 3],
        partial_rotary_factor=0.5,
        linear_conv_kernel_dim=3,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            topk_linear_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            linear_state_cache=True,
            linear_state_query_mode="per_linear_state",
            linear_state_update_mode="gated_delta",
        ),
    )


def _run_longshot_fsdp(
    rank: int,
    world_size: int,
    init_file: str,
    freeze_backbone: bool,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        device = torch.device("cuda", rank)
        config = TrainConfig(
            model=_tiny_qwen35_longshot_config(),
            distributed="fsdp",
            freeze_backbone=freeze_backbone,
            precision="bf16_mixed",
        )
        model = _build_training_model(
            config, device, rank=rank, use_fsdp=True
        )
        if freeze_backbone:
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(False)
        _apply_activation_checkpointing(model, family=config.model.family)
        model = _wrap_fsdp(model, config, device)

        input_ids = torch.randint(0, config.model.vocab_size, (1, 9), device=device)
        loss = model(input_ids, input_ids)
        loss.backward()
        assert torch.isfinite(loss)

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        dynamics_by_layer = _unwrap(model).cache_processor.linear_state_dynamics
        assert dynamics_by_layer
        assert all(
            isinstance(dynamics, FSDP)
            for dynamics in dynamics_by_layer.values()
        )
        for dynamics in dynamics_by_layer.values():
            if freeze_backbone:
                assert not any(
                    parameter.requires_grad
                    for parameter in dynamics.parameters()
                )
                continue
            with FSDP.summon_full_params(
                dynamics, recurse=False, with_grads=True
            ):
                assert dynamics.module.A_log.grad is not None
                assert dynamics.module.dt_bias.grad is not None
                assert torch.isfinite(dynamics.module.A_log.grad).all()
                assert torch.isfinite(dynamics.module.dt_bias.grad).all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
@pytest.mark.parametrize("freeze_backbone", [True, False])
def test_qwen35_longshot_two_rank_fsdp_gated_delta_dynamics(
    tmp_path: Path,
    freeze_backbone: bool,
):
    init_file = tmp_path / f"fsdp-{int(freeze_backbone)}"
    mp.spawn(
        _run_longshot_fsdp,
        args=(2, str(init_file), freeze_backbone),
        nprocs=2,
        join=True,
    )
