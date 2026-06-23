import json
from pathlib import Path

from gcw.config import load_train_config
from gcw.data.sft import prepare_openmath
from gcw.params import processor_params, total_non_embedding_params
from scripts import publish_bottleneck_llm_artifacts as publish


def _openmath_sft_config_paths() -> list[Path]:
    paths: list[Path] = []
    for root in [Path("configs/train/bottleneck_llm"), Path("configs/train/bottleneck_compat")]:
        for path in sorted(root.glob("*.yaml")):
            text = path.read_text(encoding="utf-8")
            if "kind: sft" in text and "OpenMathInstruct-2/train.jsonl" in text:
                paths.append(path)
    return paths


def test_publication_spec_has_expected_artifact_counts():
    legacy = publish.legacy_artifacts()
    sft = publish.sft_artifacts()

    assert len(legacy) == 8
    assert len(sft) == 4
    assert all("/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/" in str(item[3]) for item in legacy)
    assert all("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/" in str(item[2]) for item in sft)


def test_prepare_openmath_normalizes_jsonl(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"problem": "2+2?", "solution": "4"}),
                json.dumps({"problem": "3+3?", "generated_solution": "6", "expected_answer": "six"}),
                json.dumps({"messages": [{"content": "Q"}, {"content": "A"}]}),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "train.jsonl"

    assert prepare_openmath(source, out) == 3

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"prompt": "2+2?", "answer": "4"},
        {"prompt": "3+3?", "answer": "6"},
        {"prompt": "Q", "answer": "A"},
    ]


def test_bottleneck_llm_training_configs_load_with_expected_modes():
    frozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_frozen.yaml")
    head_mask = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_head_mask_from_sft_frozen.yaml")
    head_mask_fair = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_head_mask_fair_from_sft_frozen.yaml")
    unfrozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_unfrozen.yaml")
    icarus_frozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen.yaml")
    icarus_shared = load_train_config("configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen_shared.yaml")
    icarus_unfrozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_unfrozen.yaml")
    tf_sft = load_train_config("configs/train/bottleneck_llm/llama32_1b_tf_sft.yaml")

    assert frozen.freeze_backbone
    assert frozen.model.variant == "bgt"
    assert frozen.backbone_checkpoint.startswith("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/")
    assert head_mask.freeze_backbone
    assert head_mask.model.variant == "bgt"
    assert head_mask.model.cache_processor.processor_mode == "head_mask_broadcast"
    assert head_mask.model.cache_processor.score == "kv_norm"
    assert head_mask.model.cache_processor.update_stream == "kv"
    assert head_mask.model.cache_processor.writer_head_fraction == 0.2
    assert head_mask.data.dataset == "/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl"
    assert head_mask_fair.model.tie_embeddings == frozen.model.tie_embeddings
    assert head_mask_fair.model.cache_processor.processor_mode == "head_mask_broadcast"
    assert head_mask_fair.model.cache_processor.score == "kv_norm"
    assert head_mask_fair.model.cache_processor.writer_head_fraction == 0.2
    assert head_mask_fair.model.cache_processor.out_proj_init == frozen.model.cache_processor.out_proj_init
    assert head_mask_fair.data.sft_skip_rows == 128000
    assert head_mask_fair.data.sft_take_rows == 128000
    assert head_mask_fair.data.response_prefix == "\n"
    assert head_mask_fair.adam_beta2 == 0.999
    assert not unfrozen.freeze_backbone
    assert unfrozen.model.variant == "bgt"
    assert icarus_frozen.freeze_backbone
    assert icarus_frozen.model.variant == "icarus"
    assert icarus_frozen.data.dataset == "/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl"
    assert icarus_frozen.backbone_checkpoint.startswith("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/")
    assert icarus_shared.freeze_backbone
    assert icarus_shared.model.variant == "icarus"
    assert icarus_shared.model.cache_processor.per_cache_head_projections is False
    assert icarus_shared.model.legacy_bottleneck_generate
    assert icarus_shared.data.sft_skip_rows == 128000
    assert icarus_shared.data.sft_take_rows == 128000
    assert icarus_shared.data.response_prefix == "\n"
    assert icarus_shared.adam_beta2 == 0.999
    assert icarus_shared.metadata["processor_params"] == processor_params(icarus_shared.model)
    assert icarus_shared.metadata["total_non_embedding_params"] == total_non_embedding_params(icarus_shared.model)
    assert not icarus_unfrozen.freeze_backbone
    assert icarus_unfrozen.model.variant == "icarus"
    assert tf_sft.model.variant == "tf"
    assert tf_sft.model.cache_processor is None


def test_martin_launch_script_uses_shared_two_gpu_icarus_recipe():
    script = Path("scripts/martin.sh").read_text(encoding="utf-8")

    assert 'llama32_1b_icarus_from_sft_frozen_shared.yaml' in script
    assert 'GPUS="${GPUS:-2}"' in script


def test_openmath_sft_configs_use_legacy_formatting_and_slices():
    paths = _openmath_sft_config_paths()

    assert len(paths) == 40
    for path in paths:
        config = load_train_config(path)
        assert config.data.response_prefix == "\n", path
        assert config.data.sft_take_rows == 128000, path
        assert config.adam_beta2 == 0.999, path
        if config.model.variant == "tf":
            assert config.data.sft_skip_rows == 0, path
        else:
            assert config.data.sft_skip_rows == 128000, path


def test_head_mask_fair_config_matches_bgt_recipe_except_masking():
    baseline = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_frozen.yaml")
    head_mask = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_head_mask_fair_from_sft_frozen.yaml")

    assert head_mask.model.variant == baseline.model.variant == "bgt"
    assert head_mask.model.tie_embeddings == baseline.model.tie_embeddings
    assert head_mask.data == baseline.data
    assert head_mask.mode == baseline.mode
    assert head_mask.distributed == baseline.distributed
    assert head_mask.freeze_backbone == baseline.freeze_backbone
    assert head_mask.backbone_checkpoint == baseline.backbone_checkpoint
    assert head_mask.batch_size == baseline.batch_size
    assert head_mask.sequence_len == baseline.sequence_len
    assert head_mask.max_steps == baseline.max_steps
    assert head_mask.total_batch_size == baseline.total_batch_size
    assert head_mask.learning_rate == baseline.learning_rate
    assert head_mask.adam_beta1 == baseline.adam_beta1
    assert head_mask.adam_beta2 == baseline.adam_beta2 == 0.999
    assert head_mask.adam_eps == baseline.adam_eps
    assert head_mask.warmup_steps == baseline.warmup_steps
    assert head_mask.weight_decay == baseline.weight_decay
    assert head_mask.grad_clip == baseline.grad_clip
    assert head_mask.bf16 == baseline.bf16
    assert head_mask.seed == baseline.seed

    base_cp = baseline.model.cache_processor
    mask_cp = head_mask.model.cache_processor
    assert base_cp is not None
    assert mask_cp is not None
    assert mask_cp.process_every == base_cp.process_every
    assert mask_cp.topk == base_cp.topk
    assert mask_cp.hidden_size == base_cp.hidden_size
    assert mask_cp.intermediate_size == base_cp.intermediate_size
    assert mask_cp.num_layers == base_cp.num_layers
    assert mask_cp.num_heads == base_cp.num_heads
    assert mask_cp.num_kv_heads == base_cp.num_kv_heads
    assert mask_cp.gate_init == base_cp.gate_init
    assert mask_cp.update_stream == base_cp.update_stream == "kv"
    assert mask_cp.out_proj_init == base_cp.out_proj_init == "normal"
    assert mask_cp.gate_mode == base_cp.gate_mode == "sigmoid"
    assert mask_cp.processor_mode == "head_mask_broadcast"
    assert mask_cp.score == "kv_norm"
    assert mask_cp.writer_head_fraction == 0.2
