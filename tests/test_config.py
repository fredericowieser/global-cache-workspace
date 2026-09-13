import pytest

from gcw.config import (
    DataConfig,
    EvalConfig,
    LoRAConfig,
    LongshotSelectorTrainingConfig,
    ModelConfig,
    ResumeConfig,
    TrainConfig,
    WandbConfig,
    load_train_config,
)


def test_eval_config_defaults_to_plain_thinking_disabled() -> None:
    config = EvalConfig()
    assert config.batch_size == 8
    assert config.limit is None
    assert config.apply_chat_template is False
    assert config.enable_thinking is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"apply_chat_template": 1},
        {"enable_thinking": 1},
        {"enable_thinking": True},
    ],
)
def test_eval_config_rejects_invalid_thinking_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        EvalConfig(**kwargs)


def test_qwen35_h50_k4_recipe_preserves_frozen_training_protocol() -> None:
    config = load_train_config(
        "configs/train/qwen35_2b_diffusion_longshot_h50_k4.yaml"
    )
    assert config.model.family == "qwen3_5"
    assert config.model.variant == "longshot"
    assert config.freeze_backbone and config.trainable == "cache_processor"
    assert config.model.lora is None
    assert config.init_checkpoint is None and config.resume is None
    assert config.backbone_checkpoint is not None
    assert config.backbone_checkpoint.endswith("qwen35-2b-tf-base-native")

    processor = config.model.cache_processor
    assert processor is not None
    assert processor.linear_state_cache and processor.live_rank1_linear_writer
    assert processor.linear_state_update_mode == "outer_product"
    assert processor.out_proj_init == "zero"
    assert processor.output_scale_mode == "fixed_one"
    assert processor.process_every == 64 and processor.iterations == 1
    diffusion = config.model.diffusion_processor
    assert diffusion is not None and diffusion.training_method == "scheduled"
    assert diffusion.schedule_horizon == 50
    assert (
        diffusion.training_solver_steps == diffusion.inference_solver_steps == 4
    )
    assert diffusion.time_conditioning and diffusion.perturbation_enabled
    assert diffusion.noise_scale == 0.1
    assert diffusion.inference_mode == "deterministic"

    assert config.mode == "sft" and config.seed == 24
    assert config.data.sft_loss_scope == "answer_only"
    assert config.data.sft_chat_template and not config.data.sft_enable_thinking
    assert config.data.prompt_suffix == config.data.response_prefix == ""
    assert config.data.sft_skip_rows == config.data.sft_take_rows == 128000
    assert config.precision == "bf16" and config.loss_backend == "liger"
    assert config.distributed == "none" and config.batch_size == 16
    assert config.sequence_len == 512 and config.total_batch_size == 65536
    assert config.max_steps == 2000 and config.learning_rate == 1e-4
    assert (config.adam_beta1, config.adam_beta2, config.adam_eps) == (
        0.9,
        0.999,
        1e-8,
    )
    assert (
        config.weight_decay == config.warmup_steps == config.warmdown_ratio == 0
    )
    assert config.truncated_bptt_windows == 0
    assert config.longshot_gradient_checkpointing_seq_len_threshold == 513
    assert config.performance.noisy_diffusion.recurrence_backend == "eager"
    assert not config.performance.adaptive_batching.enabled


def test_train_config_builds_explicit_resume_and_wandb_state():
    config = TrainConfig(
        resume={
            "checkpoint": "runs/example/checkpoint",
            "topology": "elastic_ddp",
        },
        wandb={"enabled": True, "id": "run-123", "resume": "must"},
    )

    assert config.resume == ResumeConfig(
        checkpoint="runs/example/checkpoint", topology="elastic_ddp"
    )
    assert config.wandb == WandbConfig(
        enabled=True, id="run-123", resume="must"
    )


def test_longshot_selector_training_config_serializes_and_validates():
    config = TrainConfig(
        model=ModelConfig(
            variant="longshot",
            cache_processor={
                "head_selection_method": "learned_contextual",
            },
        ),
        longshot_selector_training={},
    )

    assert config.longshot_selector_training == LongshotSelectorTrainingConfig()

    with pytest.raises(ValueError, match="phase boundaries"):
        LongshotSelectorTrainingConfig(soft_end=0.7, second_hardening_start=0.6)
    with pytest.raises(ValueError, match="temperatures"):
        LongshotSelectorTrainingConfig(final_temperature=0.0)
    with pytest.raises(ValueError, match="suppressions"):
        LongshotSelectorTrainingConfig(final_suppression=1.0)


def test_longshot_selector_training_rejects_unsupported_execution_paths():
    with pytest.raises(ValueError, match="cannot be combined with flow"):
        TrainConfig(
            model=ModelConfig(
                variant="longshot",
                flow_processor={},
                cache_processor={
                    "head_selection_method": "learned_contextual",
                },
            ),
            longshot_selector_training={},
        )

    with pytest.raises(ValueError, match="multiple internal iterations"):
        TrainConfig(
            model=ModelConfig(
                variant="longshot",
                cache_processor={
                    "head_selection_method": "learned_contextual",
                    "iterations": 2,
                    "internal_iteration_score": "attn_l2",
                },
            ),
            longshot_selector_training={},
        )


@pytest.mark.parametrize(
    "cache_processor",
    [
        {
            "head_selection_method": "magnitude",
            "internal_iteration_score": "learned_contextual",
            "iteration_mode": "external",
            "iterations": 2,
        },
        {
            "head_selection_method": "magnitude",
            "internal_iteration_score": "learned_contextual",
            "iteration_mode": "internal",
            "iterations": 1,
        },
    ],
)
def test_longshot_selector_training_rejects_unreachable_iterative_selection(
    cache_processor,
):
    with pytest.raises(ValueError, match="multiple internal iterations"):
        TrainConfig(
            model=ModelConfig(
                variant="longshot",
                cache_processor=cache_processor,
            ),
            longshot_selector_training={},
        )


def test_longshot_selector_training_accepts_reachable_iterative_selection():
    config = TrainConfig(
        model=ModelConfig(
            variant="longshot",
            cache_processor={
                "head_selection_method": "magnitude",
                "internal_iteration_score": "learned_contextual",
                "iteration_mode": "internal",
                "iterations": 2,
            },
        ),
        longshot_selector_training={},
    )

    assert config.longshot_selector_training is not None


def test_row_and_token_batch_contracts_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one"):
        TrainConfig(total_batch_size=1024, global_batch_rows=64)


def test_train_config_allows_backbone_with_resume_for_fingerprint_stability():
    # Resuming must preserve the original backbone_checkpoint so the training
    # fingerprint matches, while init_checkpoint remains mutually exclusive.
    config = TrainConfig(
        backbone_checkpoint="runs/backbone/checkpoint",
        resume={"checkpoint": "runs/training/checkpoint"},
    )
    assert config.backbone_checkpoint == "runs/backbone/checkpoint"
    assert config.resume is not None


def test_train_config_rejects_init_checkpoint_with_resume():
    with pytest.raises(
        ValueError, match="init_checkpoint and resume are mutually exclusive"
    ):
        TrainConfig(
            init_checkpoint="runs/init/checkpoint",
            resume={"checkpoint": "runs/training/checkpoint"},
        )


def test_train_config_rejects_backbone_with_init_checkpoint():
    with pytest.raises(
        ValueError,
        match="backbone_checkpoint and init_checkpoint are mutually exclusive",
    ):
        TrainConfig(
            backbone_checkpoint="runs/backbone/checkpoint",
            init_checkpoint="runs/init/checkpoint",
        )


@pytest.mark.parametrize("grad_clip", [0.0, -1.0, float("inf"), float("nan")])
def test_train_config_requires_finite_positive_grad_clip(grad_clip):
    with pytest.raises(
        ValueError, match="grad_clip must be finite and positive"
    ):
        TrainConfig(grad_clip=grad_clip)


@pytest.mark.parametrize("trainable", ["all", "backbone", "cache_processor"])
def test_train_config_accepts_explicit_trainable_policy(trainable):
    config = TrainConfig(
        model=ModelConfig(variant="bgt"),
        data=DataConfig(kind="random"),
        trainable=trainable,
    )
    assert config.trainable == trainable


def test_lora_policy_requires_adapters():
    with pytest.raises(ValueError, match="requires model.lora"):
        TrainConfig(trainable="lora")
    config = TrainConfig(
        model=ModelConfig(lora=LoRAConfig(rank=8)), trainable="lora"
    )
    assert config.model.lora == LoRAConfig(rank=8)


def test_processor_only_policy_requires_a_processor():
    with pytest.raises(ValueError, match="requires a cache processor"):
        TrainConfig(trainable="cache_processor")


def test_wandb_must_resume_requires_run_id():
    with pytest.raises(ValueError, match="wandb.id"):
        WandbConfig(enabled=True, resume="must")


@pytest.mark.parametrize("tags", [[""], ["x" * 65], [1], "not-a-list"])
def test_wandb_tags_match_service_constraints(tags):
    with pytest.raises(ValueError, match="between 1 and 64 characters"):
        WandbConfig(enabled=True, tags=tags)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "topology"),
    [
        (
            "configs/train/resume/smollm360m_matched_bgt_20b_pe512_resume.yaml",
            "same",
        ),
        (
            "configs/train/resume/smollm360m_matched_bt_20b_pe512_resume.yaml",
            "same",
        ),
        (
            "configs/train/resume/smollm360m_matched_bt_20b_pe512_resume_2gpu.yaml",
            "elastic_ddp",
        ),
    ],
)
def test_resume_configs_use_versioned_resume_interface(path, topology):
    config = load_train_config(path)

    assert config.resume is not None
    assert config.resume.topology == topology
    assert config.wandb.id
    assert config.wandb.resume == "must"


@pytest.mark.parametrize("indices", ([1, 0], [0, 0]))
def test_processor_layer_indices_must_be_sorted_and_unique(indices):
    with pytest.raises(ValueError, match="sorted and unique"):
        ModelConfig(num_layers=2, processor_layer_indices=indices)


@pytest.mark.parametrize(
    ("artifact_hash", "source_ids_hash"),
    [
        (None, "1" * 64),
        ("1" * 64, None),
        ("0" * 64, "1" * 64),
        ("1" * 64, "placeholder"),
    ],
)
def test_canonical_data_config_requires_non_placeholder_artifact_identity_hashes(
    artifact_hash,
    source_ids_hash,
):
    with pytest.raises(
        ValueError, match="canonical data requires authenticated"
    ):
        DataConfig(
            canonical_execution=True,
            expected_artifact_sha256=artifact_hash,
            expected_source_ids_sha256=source_ids_hash,
        )


def test_load_train_config_bytes_uses_captured_snapshot(tmp_path) -> None:
    from gcw.config import load_train_config_bytes

    path = tmp_path / "train.yaml"
    path.write_text("mode: pretrain\nseed: 17\n")
    snapshot = path.read_bytes()
    path.write_text("mode: sft\nseed: 999\n")

    loaded = load_train_config_bytes(snapshot)
    assert loaded.mode == "pretrain"
    assert loaded.seed == 17
