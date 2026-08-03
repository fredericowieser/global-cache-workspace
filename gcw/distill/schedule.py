from __future__ import annotations


def recurrent_event_due(*, consumed_response_tokens: int, process_every: int, will_continue: bool) -> bool:
    if consumed_response_tokens <= 0:
        raise ValueError("consumed_response_tokens must be positive")
    if process_every <= 0:
        raise ValueError("process_every must be positive")
    return will_continue and consumed_response_tokens % process_every == 0


def processor_event_positions(max_response_tokens: int, process_every: int) -> tuple[int, ...]:
    """Return response-relative events that can affect an answer distribution."""
    if max_response_tokens < 0:
        raise ValueError("max_response_tokens must be non-negative")
    if process_every <= 0:
        raise ValueError("process_every must be positive")
    return (0, *range(process_every, max_response_tokens, process_every))
