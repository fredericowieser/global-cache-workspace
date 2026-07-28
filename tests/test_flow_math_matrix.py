from __future__ import annotations

from pathlib import Path

import pytest

from gcw.config import load_train_config
from gcw.params import processor_params


CONFIG_DIR = Path("configs/train/flow_math_llama32_1b")


def test_flow_bgt_seed24_config_is_capacity_matched_and_frozen():
    path = CONFIG_DIR / "llama32_1b_flow_bgt_seed24.yaml"

    assert path.is_file()
    config = load_train_config(path)

    assert config.model.variant == "bgt"
    assert config.model.architecture_name == "flow_bgt"
    assert config.model.legacy_bottleneck_generate is False
    assert config.model.flow_processor is not None
    assert config.model.flow_processor.num_steps == 2
    assert config.model.flow_processor.time_conditioning is True
    assert config.model.flow_processor.eta_max == 1.0
    assert config.model.flow_processor.flow_gate_init == -4.0
    assert config.model.cache_processor is not None
    assert config.model.cache_processor.hidden_size == 960
    assert config.model.cache_processor.intermediate_size == 3840
    assert config.model.cache_processor.num_heads == 15
    assert config.model.cache_processor.num_kv_heads == 15
    assert processor_params(config.model) == 46_242_370
    assert config.freeze_backbone is True
    assert config.seed == 24
    assert config.data.response_prefix == "\n"


def test_flow_math_matrix_has_exact_configs_and_shared_protocol():
    variants = {
        "bgt": ("flow_bgt", 46_242_370),
        "icarus": ("flow_icarus", 46_348_034),
        "longshot": ("flow_longshot", 47_659_650),
    }
    seeds = (24, 25, 26)
    expected = {
        CONFIG_DIR / f"llama32_1b_flow_{variant}_seed{seed}.yaml"
        for variant in variants
        for seed in seeds
    }

    assert set(CONFIG_DIR.glob("*.yaml")) == expected

    output_dirs: set[str] = set()
    wandb_names: set[str] = set()
    wandb_ids: set[str] = set()
    for path in sorted(expected):
        variant = path.stem.removeprefix("llama32_1b_flow_").split("_seed", maxsplit=1)[0]
        seed = int(path.stem.rsplit("seed", maxsplit=1)[1])
        architecture_name, expected_params = variants[variant]
        config = load_train_config(path)

        assert config.model.variant == variant
        assert config.model.architecture_name == architecture_name
        assert config.model.legacy_bottleneck_generate is False
        assert config.model.flow_processor is not None
        assert config.model.flow_processor.num_steps == 2
        assert config.model.flow_processor.time_conditioning is True
        assert config.model.flow_processor.eta_max == 1.0
        assert config.model.flow_processor.flow_gate_init == -4.0
        assert processor_params(config.model) == expected_params
        assert config.freeze_backbone is True
        assert config.backbone_checkpoint == "/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/llama32_1b/tf"
        assert config.data.dataset == "/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl"
        assert config.data.tokenizer_source == "/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model"
        assert config.data.sft_skip_rows == 128_000
        assert config.data.sft_take_rows == 128_000
        assert config.data.response_prefix == "\n"
        assert config.mode == "sft"
        assert config.distributed == "ddp"
        assert config.batch_size == 16
        assert config.sequence_len == 512
        assert config.max_steps == 1_000
        assert config.total_batch_size == 65_536
        assert config.learning_rate == 1e-4
        assert config.adam_beta2 == 0.999
        assert config.warmup_steps == 0
        assert config.warmdown_ratio == 0.0
        assert config.weight_decay == 0.0
        assert config.grad_clip == 1.0
        assert config.bf16 is True
        assert config.save_every == 100
        assert config.seed == seed
        assert config.output_dir == (
            f"/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1/runs/{variant}/seed-{seed}"
        )
        assert config.wandb.group == "flow-math-llama32-1b-v1"
        assert f"flow-{variant}" in config.wandb.tags
        assert f"seed-{seed}" in config.wandb.tags

        output_dirs.add(config.output_dir)
        wandb_names.add(config.wandb.name)
        wandb_id = config.wandb.id
        assert wandb_id is not None
        wandb_ids.add(wandb_id)

    assert len(output_dirs) == 9
    assert len(wandb_names) == 9
    assert len(wandb_ids) == 9


@pytest.mark.parametrize(
    ("variant", "steps"),
    [("bgt", 2), ("icarus", 2), ("longshot", 2), ("bgt", 100)],
)
def test_flow_math_pilot_config_uses_full_model_and_separate_output(variant, steps):
    suffix = f"{variant}_steps{steps}"
    path = CONFIG_DIR / "pilots" / f"llama32_1b_flow_{suffix}.yaml"

    assert path.is_file()
    pilot = load_train_config(path)
    full = load_train_config(CONFIG_DIR / f"llama32_1b_flow_{variant}_seed24.yaml")

    assert pilot.model == full.model
    assert pilot.data == full.data
    assert pilot.freeze_backbone is True
    assert pilot.seed == 24
    assert pilot.max_steps == steps
    assert pilot.total_batch_size == full.total_batch_size
    assert pilot.output_dir == (
        f"/mnt/jfs-01/shared/gcw/experiments/flow_math_llama32_1b_v1/pilots/{variant}/steps-{steps}"
    )
    assert pilot.wandb.enabled is False
