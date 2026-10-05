# Adapted from the pinned Reasoning Gym source; see SOURCES.json and NOTICE.txt.
"""Base class for procedural dataset generators"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sized
from random import Random
from typing import Any, Optional


class ProceduralDataset(ABC, Sized, Iterable[dict[str, Any]]):
    """Abstract base class for procedural dataset generators"""

    def __init__(self, config: Any, seed: Optional[int] = None, size: int = 500):
        """Initialize the dataset with config, optional seed and size"""
        if hasattr(config, "validate") and callable(config.validate):
            config.validate()

        self.config = config
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)

    @property
    def category(self) -> str:
        """Extract category from the module name."""
        module_name = self.__class__.__module__
        parts = module_name.split(".")
        if len(parts) >= 3:
            return parts[1]  # reasoning_gym.{category}.dataset_name
        return "other"

    def __len__(self) -> int:
        """Return the virtual size of the dataset"""
        return self.size

    def __iter__(self):
        """Make the dataset iterable"""
        self._current_idx = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next item in iteration"""
        if self._current_idx >= self.size:
            raise StopIteration
        item = self[self._current_idx]
        self._current_idx += 1
        return item

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Generate a single dataset item

        Args:
            idx: Index of the item to generate

        Returns:
            dict containing at least:
                - question: str
                - answer: str
                - metadata: dict
        """
        raise NotImplementedError

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Overwrite this method in derived classes if a single oracle answer is not available."""
        oracle_answer = entry["answer"]
        reward = 0.0
        if isinstance(answer, str) and len(answer) > 0:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = len(oracle_answer) / len(answer)
        return reward
