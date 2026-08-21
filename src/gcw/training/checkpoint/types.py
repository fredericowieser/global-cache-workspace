from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gcw.data.training import LoaderCursor

FORMAT_VERSION = 2
PAYLOAD_FILES = frozenset(
    {
        "model.safetensors",
        "optimizer.pt",
        "rng.pt",
        "config.json",
        "train_state.json",
    }
)
INITIAL_PROCESSOR_FILE = "initial_processor.safetensors"
OPTIONAL_PAYLOAD_FILES = frozenset({INITIAL_PROCESSOR_FILE})


class TrainingCheckpointError(RuntimeError):
    pass


class LegacyCheckpointError(TrainingCheckpointError):
    pass


class CheckpointCompatibilityError(TrainingCheckpointError):
    pass


class UnsupportedTopologyError(CheckpointCompatibilityError):
    pass


@dataclass(frozen=True)
class ResumeState:
    step: int
    smooth_loss: float
    loader_cursor: LoaderCursor
    saved_world_size: int
    exact_topology: bool
    metadata: dict[str, Any] = field(default_factory=dict)
