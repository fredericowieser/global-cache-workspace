from pathlib import Path
import os
import shutil

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import gcw.training.checkpoint as checkpoint_module
import gcw.training.state as state_module
from gcw.checkpoint import load_checkpoint
from gcw.config import DataConfig, ModelConfig, ResumeConfig, TrainConfig
from gcw.data.shards import ShardWriter
from gcw.data.training import RandomCursor
from gcw.models import build_model
from gcw.training.core import train
from gcw.training.checkpoint import ResumeState, TrainingCheckpointError
from gcw.training.checkpoint import save_training_checkpoint as real_save_training_checkpoint


def test_two_process_ddp_resume_matches_uninterrupted_training(tmp_path):
    baseline_dir = tmp_path / "baseline"
    resumed_dir = tmp_path / "resumed"
    mp.spawn(
        _train_worker,
        args=(2, str(tmp_path / "baseline-init"), str(baseline_dir), False),
        nprocs=2,
        join=True,
    )
    mp.spawn(
        _train_worker,
        args=(2, str(tmp_path / "resumed-init"), str(resumed_dir), True),
        nprocs=2,
        join=True,
    )

    baseline, _, baseline_state = load_checkpoint(baseline_dir / "checkpoint")
    resumed, _, resumed_state = load_checkpoint(resumed_dir / "checkpoint")
    assert baseline_state["step"] == resumed_state["step"] == 2
    for expected, actual in zip(baseline.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_one_to_two_process_elastic_fineweb_resume_is_deterministic(tmp_path):
    data_dir = tmp_path / "data"
    writer = ShardWriter(data_dir, "train", shard_tokens=1000)
    writer.add_tokens([index % 64 for index in range(513)])
    writer.close()
    seed_dir = tmp_path / "seed"
    config = _fineweb_config(str(seed_dir), str(data_dir), batch_size=2)

    def save_then_interrupt(path, model, model_config, optimizer, resume_state, training_fingerprint):
        real_save_training_checkpoint(path, model, model_config, optimizer, resume_state, training_fingerprint)
        if resume_state.step == 1 and Path(path).name == "checkpoint":
            raise RuntimeError("simulated elastic interruption")

    state_module.save_training_checkpoint = save_then_interrupt
    try:
        train(config)
    except RuntimeError as exc:
        if "simulated elastic interruption" not in str(exc):
            raise
    finally:
        state_module.save_training_checkpoint = real_save_training_checkpoint

    outputs = [tmp_path / "elastic-a", tmp_path / "elastic-b"]
    for index, output in enumerate(outputs):
        shutil.copytree(seed_dir / "checkpoint", output / "checkpoint")
        mp.spawn(
            _elastic_worker,
            args=(2, str(tmp_path / f"elastic-init-{index}"), str(output), str(data_dir)),
            nprocs=2,
            join=True,
        )

    first, _, first_state = load_checkpoint(outputs[0] / "checkpoint")
    second, _, second_state = load_checkpoint(outputs[1] / "checkpoint")
    assert first_state["step"] == second_state["step"] == 2
    assert first_state["resume_exact_topology"] is False
    for expected, actual in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_rank_zero_checkpoint_failure_reaches_every_ddp_rank(tmp_path):
    mp.spawn(
        _checkpoint_failure_worker,
        args=(2, str(tmp_path / "failure-init"), str(tmp_path / "failure")),
        nprocs=2,
        join=True,
    )


def _train_worker(rank: int, world_size: int, init_file: str, output_dir: str, interrupt: bool) -> None:
    _init_process_group(rank, world_size, init_file)
    config = _config(output_dir)
    if not interrupt:
        train(config)
        return

    def save_then_interrupt(path, model, model_config, optimizer, resume_state, training_fingerprint):
        real_save_training_checkpoint(path, model, model_config, optimizer, resume_state, training_fingerprint)
        if resume_state.step == 1 and Path(path).name == "checkpoint":
            raise RuntimeError("simulated distributed interruption")

    state_module.save_training_checkpoint = save_then_interrupt
    try:
        train(config)
    except RuntimeError as exc:
        if "simulated distributed interruption" not in str(exc):
            raise
    finally:
        state_module.save_training_checkpoint = real_save_training_checkpoint
    config.resume = ResumeConfig(checkpoint=str(Path(output_dir) / "checkpoint"))
    train(config)


def _elastic_worker(rank: int, world_size: int, init_file: str, output_dir: str, data_dir: str) -> None:
    _init_process_group(rank, world_size, init_file)
    config = _fineweb_config(output_dir, data_dir, batch_size=1)
    config.resume = ResumeConfig(checkpoint=str(Path(output_dir) / "checkpoint"), topology="elastic_ddp")
    train(config)


def _checkpoint_failure_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    _init_process_group(rank, world_size, init_file)
    config = _config(output_dir)
    model = build_model(config.model)
    model.init_weights()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = ResumeState(
        step=1,
        smooth_loss=0.0,
        loader_cursor=RandomCursor(rng_state=(1, 2, 3)),
        saved_world_size=world_size,
        exact_topology=True,
    )
    original_write = checkpoint_module._write_checkpoint
    if rank == 0:
        def fail_write(*args, **kwargs):
            raise OSError("simulated disk failure")

        checkpoint_module._write_checkpoint = fail_write
    try:
        try:
            checkpoint_module.save_training_checkpoint(
                Path(output_dir) / "checkpoint",
                model,
                config.model,
                optimizer,
                state,
                {"distributed": "ddp", "data_kind": "random", "seed": 24},
            )
        except TrainingCheckpointError as exc:
            assert "simulated disk failure" in str(exc)
        else:
            raise AssertionError("checkpoint failure was not broadcast")
    finally:
        checkpoint_module._write_checkpoint = original_write
        dist.destroy_process_group()


def _init_process_group(rank: int, world_size: int, init_file: str) -> None:
    use_cuda = os.environ.get("GCW_DDP_TEST_DEVICE") == "cuda"
    if use_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
            raise RuntimeError(f"CUDA DDP test requires {world_size} visible GPUs")
        os.environ["LOCAL_RANK"] = str(rank)
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl" if use_cuda else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )


def _config(output_dir: str) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            variant="tf",
            family="smollm",
            sequence_len=8,
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            tie_embeddings=True,
        ),
        data=DataConfig(kind="random"),
        distributed="ddp",
        output_dir=output_dir,
        batch_size=1,
        sequence_len=8,
        max_steps=2,
        total_batch_size=16,
        learning_rate=1e-3,
        warmup_steps=1,
        save_every=1,
        eval_every=0,
        eval_on_start=False,
    )


def _fineweb_config(output_dir: str, data_dir: str, *, batch_size: int) -> TrainConfig:
    config = _config(output_dir)
    config.data = DataConfig(kind="fineweb", data_dir=data_dir)
    config.batch_size = batch_size
    config.sequence_len = 4
    config.model.sequence_len = 4
    config.total_batch_size = 8
    return config
