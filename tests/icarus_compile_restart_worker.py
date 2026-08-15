"""Subprocess worker for the opt-in Icarus process-restart acceptance test."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from gcw.models import build_model
from gcw.models.model import LanguageModel
from gcw.performance.backends.icarus import (
    configure_icarus_training_execution,
    icarus_training_execution,
)
from test_icarus_compile_backend import _config


def _new_training() -> tuple[LanguageModel, torch.optim.Optimizer]:
    torch.manual_seed(4521)
    model = build_model(_config()).to("cuda")
    assert isinstance(model, LanguageModel)
    model.init_weights()
    configure_icarus_training_execution(model, (2,))
    return model, torch.optim.AdamW(model.parameters(), lr=3e-4)


def _step(model: LanguageModel, optimizer: torch.optim.Optimizer, step: int) -> None:
    generator = torch.Generator().manual_seed(900 + step)
    tokens = torch.randint(0, model.config.vocab_size, (2, 128), generator=generator).cuda()
    targets = tokens.roll(-1, dims=1)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16), icarus_training_execution(model, 2):
        loss = model(tokens, targets)
        loss.backward()
    optimizer.step()


def _save_parameters(model: LanguageModel, path: Path) -> None:
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)


def main() -> None:
    mode = sys.argv[1]
    if mode == "continuous":
        model, optimizer = _new_training()
        _step(model, optimizer, 0)
        _step(model, optimizer, 1)
        _save_parameters(model, Path(sys.argv[2]))
        return
    if mode == "save":
        model, optimizer = _new_training()
        _step(model, optimizer, 0)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, Path(sys.argv[2]))
        return
    if mode == "resume":
        model, optimizer = _new_training()
        checkpoint = torch.load(Path(sys.argv[2]), map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        _step(model, optimizer, 1)
        _save_parameters(model, Path(sys.argv[3]))
        return
    raise ValueError(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
