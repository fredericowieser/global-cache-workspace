"""Subprocess worker for the production Icarus checkpoint/restart acceptance test."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from gcw.data.training import RandomCursor, cursor_to_dict
from gcw.models import build_model
from gcw.models.model import LanguageModel
from gcw.performance.backends.icarus import (
    configure_icarus_training_execution,
    icarus_processor_backend_metadata,
    icarus_training_execution,
    warmup_icarus_training_execution,
)
from gcw.training.checkpoint import (
    ResumeState,
    TrainingCheckpoint,
    save_training_checkpoint,
)
from test_icarus_compile_backend import _config


def _fingerprint(model: LanguageModel) -> dict[str, Any]:
    return {
        "schema": "gcw.icarus_compile_restart_acceptance.v1",
        "distributed": "none",
        "icarus_processor_backend_fingerprint": icarus_processor_backend_metadata(
            model.config, "compile"
        )[
            "fingerprint_sha256"
        ],
    }


def _new_training() -> tuple[LanguageModel, torch.optim.Optimizer]:
    torch.manual_seed(4521)
    torch.cuda.manual_seed_all(4521)
    config = _config(gradient_checkpointing=True)
    config.sequence_len = 192
    model = build_model(config).to("cuda")
    assert isinstance(model, LanguageModel)
    model.init_weights()
    configure_icarus_training_execution(model, (2,))
    warmup_icarus_training_execution(
        model,
        (2,),
        sequence_length=config.sequence_len,
        device=torch.device("cuda"),
        bf16=True,
    )
    return model, torch.optim.AdamW(model.parameters(), lr=3e-4)


def _step(model: LanguageModel, optimizer: torch.optim.Optimizer) -> None:
    tokens = torch.randint(0, model.config.vocab_size, (2, 192), device="cuda")
    targets = tokens.roll(-1, dims=1)
    optimizer.zero_grad(set_to_none=True)
    with (
        torch.autocast("cuda", dtype=torch.bfloat16),
        icarus_training_execution(model, 2),
    ):
        loss = model(tokens, targets)
        loss.backward()
    optimizer.step()


def _optimizer_slots(
    model: LanguageModel, optimizer: torch.optim.Optimizer
) -> dict[str, dict[str, torch.Tensor | float]]:
    result: dict[str, dict[str, torch.Tensor | float]] = {}
    for name, parameter in model.named_parameters():
        state = optimizer.state.get(parameter)
        if not state:
            continue
        result[name] = {
            key: (
                value.detach().cpu()
                if isinstance(value, torch.Tensor)
                else float(value)
            )
            for key, value in state.items()
        }
    return result


def _save_result(
    model: LanguageModel,
    optimizer: torch.optim.Optimizer,
    cursor: RandomCursor,
    path: Path,
) -> None:
    torch.save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "optimizer_slots": _optimizer_slots(model, optimizer),
            "cpu_rng": torch.random.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "cursor": cursor_to_dict(cursor),
        },
        path,
    )


def _initial_cursor() -> RandomCursor:
    generator = torch.Generator().manual_seed(7219)
    return RandomCursor(rng_state=tuple(generator.get_state().tolist()))


def _advance_cursor(cursor: RandomCursor) -> RandomCursor:
    generator = torch.Generator()
    generator.set_state(torch.tensor(cursor.rng_state, dtype=torch.uint8))
    torch.randint(0, 64, (2, 192), generator=generator)
    return RandomCursor(rng_state=tuple(generator.get_state().tolist()))


def main() -> None:
    mode = sys.argv[1]
    if mode == "continuous":
        model, optimizer = _new_training()
        cursor = _initial_cursor()
        _step(model, optimizer)
        cursor = _advance_cursor(cursor)
        _step(model, optimizer)
        cursor = _advance_cursor(cursor)
        _save_result(model, optimizer, cursor, Path(sys.argv[2]))
        return
    if mode == "save":
        model, optimizer = _new_training()
        cursor = _initial_cursor()
        _step(model, optimizer)
        cursor = _advance_cursor(cursor)
        resume = ResumeState(
            step=1,
            smooth_loss=0.5,
            loader_cursor=cursor,
            saved_world_size=1,
            exact_topology=True,
            metadata={"acceptance": "icarus_compile"},
        )
        save_training_checkpoint(
            Path(sys.argv[2]),
            model,
            model.config,
            optimizer,
            resume,
            _fingerprint(model),
        )
        return
    if mode == "resume":
        model, optimizer = _new_training()
        checkpoint = TrainingCheckpoint.open(
            Path(sys.argv[2]),
            expected_fingerprint=_fingerprint(model),
            topology="same",
            current_world_size=1,
        )
        checkpoint.restore_model(model)
        checkpoint.restore_optimizer(optimizer, device="cuda")
        checkpoint.restore_rng(rank=0)
        resumed_cursor = checkpoint.loader_cursor(rank=0)
        assert isinstance(resumed_cursor, RandomCursor)
        assert resumed_cursor == _advance_cursor(_initial_cursor())
        assert checkpoint.resume_state.metadata == {
            "acceptance": "icarus_compile"
        }
        _step(model, optimizer)
        _save_result(
            model, optimizer, _advance_cursor(resumed_cursor), Path(sys.argv[3])
        )
        return
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
