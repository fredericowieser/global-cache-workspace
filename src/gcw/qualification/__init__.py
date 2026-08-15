"""Private qualification primitives.

The supported package API remains :mod:`gcw`; this namespace is intentionally
private and may evolve with qualification milestones.
"""

from .state import (
    ComparisonResult,
    GradientSnapshot,
    QualificationStateError,
    StateMismatch,
    TrainingState,
    capture_gradient_snapshot,
    capture_training_state,
    compare_authority_local,
    compare_semantic_state,
    deserialize_training_state,
    encode_value,
    semantic_digest,
    serialize_training_state,
    validate_training_state,
)

__all__ = (
    "ComparisonResult",
    "GradientSnapshot",
    "QualificationStateError",
    "StateMismatch",
    "TrainingState",
    "capture_gradient_snapshot",
    "capture_training_state",
    "compare_authority_local",
    "compare_semantic_state",
    "deserialize_training_state",
    "encode_value",
    "semantic_digest",
    "serialize_training_state",
    "validate_training_state",
)
