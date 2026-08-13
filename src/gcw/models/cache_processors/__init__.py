"""Cache state, selection, processing, recurrence, and writeback mechanisms.

Import concrete processors from their owning modules. The package initializer
stays dependency-light because transformer backbones also depend on cache-state
records during generation.
"""

from gcw.models.cache_processors.base import CacheProcessorBase

__all__ = ["CacheProcessorBase"]
