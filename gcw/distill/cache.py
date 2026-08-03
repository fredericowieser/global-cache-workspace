from __future__ import annotations

from gcw.models.cache_state import CacheState


def truncate_last_cache_token(cache: CacheState) -> CacheState:
    """Drop one cached token while preserving autograd links to prior positions."""
    if cache.scores.shape[1] == 0:
        raise ValueError("cannot truncate an empty cache")
    if cache.linear_states is not None:
        raise ValueError("cache refresh does not support recurrent linear-attention state")
    return CacheState(
        [key[:, :, :-1, :] for key in cache.keys],
        [value[:, :, :-1, :] for value in cache.values],
        cache.scores[:, :-1],
    )
