from __future__ import annotations


def processor_event_positions(max_response_tokens: int, process_every: int) -> tuple[int, ...]:
    """Return response-relative cache-processing events, including the boundary at zero."""
    if max_response_tokens < 0:
        raise ValueError("max_response_tokens must be non-negative")
    if process_every <= 0:
        raise ValueError("process_every must be positive")
    return tuple(range(0, max_response_tokens + 1, process_every))
