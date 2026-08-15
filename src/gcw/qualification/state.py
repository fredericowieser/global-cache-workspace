"""Canonical, fail-closed capture of complete training state for qualification.

This module deliberately has no integration with the training loop.  Qualification
callers take gradient snapshots and capture state outside measured regions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
import math
import random
import struct
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

import torch
from torch import nn

_SCHEMA_VERSION = 1
_ARCHIVE_MAGIC = b"GCWQSTATE\x00\x01"


class QualificationStateError(ValueError):
    """Raised when state cannot be represented canonically and safely."""


@dataclass(frozen=True)
class ClassIdentity:
    module: str
    qualname: str


@dataclass(frozen=True)
class TensorMetadata:
    tensor_class: ClassIdentity
    dtype: str
    device_type: str
    device_index: int | None
    element_size: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    layout: str
    requires_grad: bool
    is_conj: bool
    is_neg: bool


@dataclass(frozen=True)
class TensorRecord:
    metadata: TensorMetadata
    content: bytes
    content_sha256: str


@dataclass(frozen=True)
class NullValue:
    tag: Literal["null"] = "null"


@dataclass(frozen=True)
class BoolValue:
    value: bool
    tag: Literal["bool"] = "bool"


@dataclass(frozen=True)
class IntValue:
    value: int
    tag: Literal["int"] = "int"


@dataclass(frozen=True)
class FloatValue:
    hexadecimal: str
    tag: Literal["float"] = "float"


@dataclass(frozen=True)
class StringValue:
    value: str
    tag: Literal["string"] = "string"


@dataclass(frozen=True)
class BytesValue:
    value: bytes
    tag: Literal["bytes"] = "bytes"


@dataclass(frozen=True)
class TorchDTypeValue:
    name: str
    tag: Literal["torch_dtype"] = "torch_dtype"


@dataclass(frozen=True)
class TorchDeviceValue:
    device_type: str
    device_index: int | None
    tag: Literal["torch_device"] = "torch_device"


@dataclass(frozen=True)
class SequenceValue:
    sequence_type: Literal["list", "tuple"]
    items: tuple[CanonicalValue, ...]
    tag: Literal["sequence"] = "sequence"


@dataclass(frozen=True)
class MappingEntry:
    key: str
    value: CanonicalValue


@dataclass(frozen=True)
class MappingValue:
    entries: tuple[MappingEntry, ...]
    tag: Literal["mapping"] = "mapping"


@dataclass(frozen=True)
class TensorValue:
    tensor: TensorRecord
    tag: Literal["tensor"] = "tensor"


@dataclass(frozen=True)
class NumpyArrayValue:
    dtype: str
    shape: tuple[int, ...]
    content: bytes
    content_sha256: str
    tag: Literal["numpy_array"] = "numpy_array"


CanonicalValue: TypeAlias = (
    NullValue
    | BoolValue
    | IntValue
    | FloatValue
    | StringValue
    | BytesValue
    | TorchDTypeValue
    | TorchDeviceValue
    | SequenceValue
    | MappingValue
    | TensorValue
    | NumpyArrayValue
)


@dataclass(frozen=True)
class ModuleRecord:
    name: str
    object_alias: str
    class_identity: ClassIdentity
    training: bool


@dataclass(frozen=True)
class ParameterSlot:
    name: str
    present: bool
    frozen: bool | None
    trainable: bool | None
    requires_grad: bool | None
    object_alias: str | None
    storage_alias: str | None
    tensor: TensorRecord | None


@dataclass(frozen=True)
class BufferSlot:
    name: str
    present: bool
    persistent: bool
    object_alias: str | None
    storage_alias: str | None
    tensor: TensorRecord | None


@dataclass(frozen=True)
class GradientRecord:
    parameter_alias: str
    frozen: bool
    tensor: TensorRecord | None


@dataclass(frozen=True)
class GradientSnapshot:
    gradients: tuple[GradientRecord, ...]


@dataclass(frozen=True)
class OptimizerGroup:
    parameter_aliases: tuple[str, ...]
    options: MappingValue


@dataclass(frozen=True)
class OptimizerStateEntry:
    parameter_alias: str
    state: CanonicalValue


@dataclass(frozen=True)
class OptimizerRecord:
    class_identity: ClassIdentity
    defaults: MappingValue
    groups: tuple[OptimizerGroup, ...]
    state: tuple[OptimizerStateEntry, ...]


@dataclass(frozen=True)
class CudaRngState:
    disposition: Literal["captured", "unavailable", "omitted"]
    device_states: tuple[TensorRecord, ...]


@dataclass(frozen=True)
class RngState:
    python: CanonicalValue
    numpy: CanonicalValue | None
    torch_cpu: TensorRecord
    cuda: CudaRngState


@dataclass(frozen=True)
class RuntimeRecord:
    semantic: CanonicalValue
    authority_local: CanonicalValue


@dataclass(frozen=True)
class TrainingState:
    schema_version: int
    modules: tuple[ModuleRecord, ...]
    parameters: tuple[ParameterSlot, ...]
    buffers: tuple[BufferSlot, ...]
    raw_gradients: GradientSnapshot
    post_clip_gradients: GradientSnapshot
    optimizer: OptimizerRecord
    rng: RngState
    cursor_before: CanonicalValue
    cursor_after: CanonicalValue
    step_ordinal: int
    lr_schedule_values: CanonicalValue
    weight_decay_schedule_values: CanonicalValue
    reservation_manifest: CanonicalValue
    runtime: RuntimeRecord

    def to_bytes(self) -> bytes:
        """Serialize into a deterministic manifest plus binary blob archive."""
        return serialize_training_state(self)

    @classmethod
    def from_bytes(cls, payload: bytes) -> TrainingState:
        """Load and integrity-check a deterministic qualification archive."""
        return deserialize_training_state(payload)

    def semantic_digest(self) -> str:
        """Return the digest used for semantic equality (authority-local excluded)."""
        return semantic_digest(self)


MismatchKind = Literal["structural", "metadata", "alias", "bytes", "value"]


@dataclass(frozen=True)
class StateMismatch:
    path: str
    kind: MismatchKind
    expected: str
    actual: str


@dataclass(frozen=True)
class ComparisonResult:
    mismatches: tuple[StateMismatch, ...]

    @property
    def equal(self) -> bool:
        return not self.mismatches

    def require_equal(self) -> None:
        if self.mismatches:
            details = "; ".join(f"{item.path} ({item.kind})" for item in self.mismatches)
            raise AssertionError(f"qualification states differ: {details}")


@dataclass(frozen=True)
class _ParameterInventory:
    slots: tuple[tuple[str, nn.Parameter | None], ...]
    alias_to_parameter: tuple[tuple[str, nn.Parameter], ...]
    object_alias_by_id: Mapping[int, str]


@dataclass(frozen=True)
class _BufferInventory:
    slots: tuple[tuple[str, torch.Tensor | None, bool], ...]


def _identity(value: object) -> ClassIdentity:
    cls = type(value)
    return ClassIdentity(module=cls.__module__, qualname=cls.__qualname__)


def _qualified(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


def _module_inventory(model: nn.Module) -> tuple[tuple[str, nn.Module], ...]:
    found: list[tuple[str, nn.Module]] = []

    def visit(name: str, module: nn.Module, ancestors: frozenset[int]) -> None:
        module_id = id(module)
        if module_id in ancestors:
            raise QualificationStateError(f"module registration cycle at modules.{name or '<root>'}")
        found.append((name, module))
        next_ancestors = ancestors | {module_id}
        for child_name in sorted(module._modules):
            child = module._modules[child_name]
            if child is not None:
                visit(_qualified(name, child_name), child, next_ancestors)

    visit("", model, frozenset())
    return tuple(found)


def _collect_inventories(
    modules: tuple[tuple[str, nn.Module], ...],
) -> tuple[_ParameterInventory, _BufferInventory, Mapping[int, str], Mapping[tuple[str, int | None, int], str]]:
    parameters: list[tuple[str, nn.Parameter | None]] = []
    buffers: list[tuple[str, torch.Tensor | None, bool]] = []
    tensor_slots: list[tuple[str, torch.Tensor]] = []
    for module_name, module in modules:
        for local_name in sorted(module._parameters):
            name = _qualified(module_name, local_name)
            parameter = module._parameters[local_name]
            parameters.append((name, parameter))
            if parameter is not None:
                tensor_slots.append((f"parameter:{name}", parameter))
        for local_name in sorted(module._buffers):
            name = _qualified(module_name, local_name)
            buffer = module._buffers[local_name]
            persistent = local_name not in module._non_persistent_buffers_set
            buffers.append((name, buffer, persistent))
            if buffer is not None:
                tensor_slots.append((f"buffer:{name}", buffer))

    names_by_object: dict[int, list[str]] = {}
    names_by_storage: dict[tuple[str, int | None, int], list[str]] = {}
    for typed_name, tensor in tensor_slots:
        _ensure_supported_tensor(tensor, typed_name)
        names_by_object.setdefault(id(tensor), []).append(typed_name)
        storage = tensor.untyped_storage()
        storage_key = (tensor.device.type, tensor.device.index, int(storage._cdata))
        names_by_storage.setdefault(storage_key, []).append(typed_name)
    object_aliases = {key: min(names) for key, names in names_by_object.items()}
    storage_aliases = {key: min(names) for key, names in names_by_storage.items()}

    parameter_alias_to_value: dict[str, nn.Parameter] = {}
    parameter_alias_by_id: dict[int, str] = {}
    for name, parameter in parameters:
        if parameter is None:
            continue
        typed_alias = object_aliases[id(parameter)]
        if not typed_alias.startswith("parameter:"):
            raise QualificationStateError(f"parameter {name} aliases a buffer before any parameter slot")
        alias = typed_alias.removeprefix("parameter:")
        parameter_alias_by_id[id(parameter)] = alias
        parameter_alias_to_value.setdefault(alias, parameter)
    return (
        _ParameterInventory(
            slots=tuple(parameters),
            alias_to_parameter=tuple(sorted(parameter_alias_to_value.items())),
            object_alias_by_id=parameter_alias_by_id,
        ),
        _BufferInventory(slots=tuple(buffers)),
        object_aliases,
        storage_aliases,
    )


def _ensure_supported_tensor(tensor: torch.Tensor, path: str) -> None:
    if tensor.layout != torch.strided:
        raise QualificationStateError(f"unsupported non-strided tensor at {path}: {tensor.layout}")
    if tensor.is_quantized:
        raise QualificationStateError(f"unsupported quantized tensor at {path}")
    if tensor.device.type == "meta":
        raise QualificationStateError(f"unsupported meta tensor at {path}")


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # Copy only logical elements.  Never read the source storage span directly: a view
    # may have holes, overlap, or a non-zero offset.
    detached = tensor.detach().resolve_conj().resolve_neg()
    cpu_copy = detached.to(device=torch.device("cpu"), copy=True).contiguous()
    if cpu_copy.numel() == 0:
        return b""
    byte_view = cpu_copy.reshape(-1).view(torch.uint8)
    return byte_view.numpy().tobytes(order="C")


def _capture_tensor(tensor: torch.Tensor, path: str) -> TensorRecord:
    _ensure_supported_tensor(tensor, path)
    content = _tensor_bytes(tensor)
    metadata = TensorMetadata(
        tensor_class=_identity(tensor),
        dtype=str(tensor.dtype),
        device_type=tensor.device.type,
        device_index=tensor.device.index,
        element_size=tensor.element_size(),
        shape=tuple(int(dimension) for dimension in tensor.shape),
        stride=tuple(int(value) for value in tensor.stride()),
        storage_offset=int(tensor.storage_offset()),
        layout=str(tensor.layout),
        requires_grad=tensor.requires_grad,
        is_conj=tensor.is_conj(),
        is_neg=tensor.is_neg(),
    )
    return TensorRecord(metadata=metadata, content=content, content_sha256=sha256(content).hexdigest())


def encode_value(value: object, *, path: str = "value") -> CanonicalValue:
    """Recursively encode a deterministic value, rejecting ambiguity and cycles."""
    return _encode_value(value, path=path, active=set())


def _encode_value(value: object, *, path: str, active: set[int]) -> CanonicalValue:
    if value is None:
        return NullValue()
    if isinstance(value, bool):
        return BoolValue(value)
    if isinstance(value, int):
        return IntValue(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QualificationStateError(f"nonfinite float at {path}")
        return FloatValue(value.hex())
    if isinstance(value, str):
        return StringValue(value)
    if isinstance(value, bytes):
        return BytesValue(value)
    if isinstance(value, torch.dtype):
        return TorchDTypeValue(str(value))
    if isinstance(value, torch.device):
        return TorchDeviceValue(value.type, value.index)
    if isinstance(value, torch.Tensor):
        return TensorValue(_capture_tensor(value, path))

    # NumPy is optional to the qualification substrate at import time.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised in environments without NumPy
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise QualificationStateError(f"object NumPy array at {path} is unsupported")
        copied = np.ascontiguousarray(value).copy()
        content = copied.tobytes(order="C")
        return NumpyArrayValue(copied.dtype.str, tuple(copied.shape), content, sha256(content).hexdigest())
    if np is not None and isinstance(value, np.generic):
        return _encode_value(value.item(), path=path, active=active)

    if isinstance(value, Mapping):
        marker = id(value)
        if marker in active:
            raise QualificationStateError(f"cycle at {path}")
        active.add(marker)
        try:
            entries: list[MappingEntry] = []
            for key in value:
                if not isinstance(key, str):
                    raise QualificationStateError(f"non-string mapping key at {path}")
                entries.append(MappingEntry(key, _encode_value(value[key], path=f"{path}.{key}", active=active)))
            entries.sort(key=lambda item: item.key)
            return MappingValue(tuple(entries))
        finally:
            active.remove(marker)
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in active:
            raise QualificationStateError(f"cycle at {path}")
        active.add(marker)
        try:
            sequence_type: Literal["list", "tuple"] = "list" if isinstance(value, list) else "tuple"
            return SequenceValue(
                sequence_type,
                tuple(_encode_value(item, path=f"{path}[{index}]", active=active) for index, item in enumerate(value)),
            )
        finally:
            active.remove(marker)
    raise QualificationStateError(f"unsupported non-deterministic value at {path}: {_identity(value)}")


def capture_gradient_snapshot(model: nn.Module) -> GradientSnapshot:
    """Clone the current complete canonical gradient inventory.

    Call this once before clipping and again after clipping; the state capture API
    intentionally cannot derive either phase from the other.
    """
    modules = _module_inventory(model)
    parameter_inventory, _, _, _ = _collect_inventories(modules)
    records: list[GradientRecord] = []
    for alias, parameter in parameter_inventory.alias_to_parameter:
        frozen = not parameter.requires_grad
        gradient = None if frozen else parameter.grad
        captured = None if gradient is None else _capture_tensor(gradient, f"gradients.{alias}")
        records.append(GradientRecord(parameter_alias=alias, frozen=frozen, tensor=captured))
    return GradientSnapshot(tuple(records))


def _validate_gradient_snapshot(snapshot: GradientSnapshot, inventory: _ParameterInventory, path: str) -> None:
    expected = tuple(alias for alias, _ in inventory.alias_to_parameter)
    actual = tuple(record.parameter_alias for record in snapshot.gradients)
    if actual != expected:
        raise QualificationStateError(f"{path} does not match canonical parameter aliases: {actual!r} != {expected!r}")
    parameters = dict(inventory.alias_to_parameter)
    for record in snapshot.gradients:
        frozen = not parameters[record.parameter_alias].requires_grad
        if record.frozen != frozen:
            raise QualificationStateError(f"{path}.{record.parameter_alias}.frozen is inconsistent")
        if frozen and record.tensor is not None:
            raise QualificationStateError(f"{path}.{record.parameter_alias} must be None for a frozen parameter")
        if record.tensor is not None:
            _validate_tensor_record(record.tensor, f"{path}.{record.parameter_alias}.tensor")


def _capture_optimizer(optimizer: torch.optim.Optimizer, inventory: _ParameterInventory) -> OptimizerRecord:
    aliases = inventory.object_alias_by_id
    seen: set[str] = set()
    groups: list[OptimizerGroup] = []
    for group_index, group in enumerate(optimizer.param_groups):
        membership: list[str] = []
        for parameter_index, parameter in enumerate(group.get("params", ())):
            if not isinstance(parameter, torch.Tensor) or id(parameter) not in aliases:
                raise QualificationStateError(
                    f"optimizer.groups[{group_index}].params[{parameter_index}] is not owned by the model"
                )
            alias = aliases[id(parameter)]
            if alias in seen:
                raise QualificationStateError(f"ambiguous duplicate optimizer membership for parameter {alias}")
            seen.add(alias)
            membership.append(alias)
        options = {key: value for key, value in group.items() if key != "params"}
        encoded_options = encode_value(options, path=f"optimizer.groups[{group_index}].options")
        if not isinstance(encoded_options, MappingValue):  # pragma: no cover - construction invariant
            raise AssertionError("optimizer options must encode as a mapping")
        groups.append(OptimizerGroup(tuple(membership), encoded_options))

    state_entries: list[OptimizerStateEntry] = []
    for parameter, value in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or id(parameter) not in aliases:
            raise QualificationStateError("optimizer.state contains a parameter not owned by the model")
        alias = aliases[id(parameter)]
        state_entries.append(
            OptimizerStateEntry(alias, encode_value(value, path=f"optimizer.state.{alias}"))
        )
    state_entries.sort(key=lambda item: item.parameter_alias)
    if len({item.parameter_alias for item in state_entries}) != len(state_entries):
        raise QualificationStateError("optimizer.state has ambiguous duplicate parameter keys")
    defaults = encode_value(optimizer.defaults, path="optimizer.defaults")
    if not isinstance(defaults, MappingValue):  # pragma: no cover - construction invariant
        raise AssertionError("optimizer defaults must encode as a mapping")
    return OptimizerRecord(_identity(optimizer), defaults, tuple(groups), tuple(state_entries))


def _capture_rng(*, capture_cuda_rng: bool) -> RngState:
    python_state = encode_value(random.getstate(), path="rng.python")
    try:
        import numpy as np
    except ImportError:
        numpy_state = None
    else:
        numpy_state = encode_value(np.random.get_state(), path="rng.numpy")
    torch_cpu = _capture_tensor(torch.random.get_rng_state(), "rng.torch_cpu")
    if not capture_cuda_rng:
        cuda = CudaRngState("omitted", ())
    elif not torch.cuda.is_available():
        cuda = CudaRngState("unavailable", ())
    else:
        cuda = CudaRngState(
            "captured",
            tuple(
                _capture_tensor(state, f"rng.cuda[{index}]")
                for index, state in enumerate(torch.cuda.get_rng_state_all())
            ),
        )
    return RngState(python_state, numpy_state, torch_cpu, cuda)


def capture_training_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    raw_gradients: GradientSnapshot,
    post_clip_gradients: GradientSnapshot,
    cursor_before: object,
    cursor_after: object,
    step_ordinal: int,
    lr_schedule_values: object,
    weight_decay_schedule_values: object,
    reservation_manifest: object,
    runtime_semantic: object,
    runtime_authority_local: object,
    capture_cuda_rng: bool = True,
) -> TrainingState:
    """Capture canonical qualification state without changing live training objects."""
    if isinstance(step_ordinal, bool) or not isinstance(step_ordinal, int) or step_ordinal < 0:
        raise QualificationStateError("step_ordinal must be a non-negative integer")
    modules = _module_inventory(model)
    parameter_inventory, buffer_inventory, object_aliases, storage_aliases = _collect_inventories(modules)
    _validate_gradient_snapshot(raw_gradients, parameter_inventory, "raw_gradients")
    _validate_gradient_snapshot(post_clip_gradients, parameter_inventory, "post_clip_gradients")

    names_by_module_id: dict[int, list[str]] = {}
    for name, module in modules:
        names_by_module_id.setdefault(id(module), []).append(name or "<root>")
    module_records = tuple(
        ModuleRecord(
            name=name,
            object_alias=min(names_by_module_id[id(module)]),
            class_identity=_identity(module),
            training=module.training,
        )
        for name, module in modules
    )

    parameter_records: list[ParameterSlot] = []
    for name, parameter in parameter_inventory.slots:
        if parameter is None:
            parameter_records.append(ParameterSlot(name, False, None, None, None, None, None, None))
            continue
        storage = parameter.untyped_storage()
        storage_key = (parameter.device.type, parameter.device.index, int(storage._cdata))
        parameter_records.append(
            ParameterSlot(
                name=name,
                present=True,
                trainable=parameter.requires_grad,
                frozen=not parameter.requires_grad,
                requires_grad=parameter.requires_grad,
                object_alias=object_aliases[id(parameter)],
                storage_alias=storage_aliases[storage_key],
                tensor=_capture_tensor(parameter, f"parameters.{name}"),
            )
        )

    buffer_records: list[BufferSlot] = []
    for name, buffer, persistent in buffer_inventory.slots:
        if buffer is None:
            buffer_records.append(BufferSlot(name, False, persistent, None, None, None))
            continue
        storage = buffer.untyped_storage()
        storage_key = (buffer.device.type, buffer.device.index, int(storage._cdata))
        buffer_records.append(
            BufferSlot(
                name=name,
                present=True,
                persistent=persistent,
                object_alias=object_aliases[id(buffer)],
                storage_alias=storage_aliases[storage_key],
                tensor=_capture_tensor(buffer, f"buffers.{name}"),
            )
        )

    state = TrainingState(
        schema_version=_SCHEMA_VERSION,
        modules=module_records,
        parameters=tuple(parameter_records),
        buffers=tuple(buffer_records),
        raw_gradients=raw_gradients,
        post_clip_gradients=post_clip_gradients,
        optimizer=_capture_optimizer(optimizer, parameter_inventory),
        rng=_capture_rng(capture_cuda_rng=capture_cuda_rng),
        cursor_before=encode_value(cursor_before, path="cursor_before"),
        cursor_after=encode_value(cursor_after, path="cursor_after"),
        step_ordinal=step_ordinal,
        lr_schedule_values=encode_value(lr_schedule_values, path="lr_schedule_values"),
        weight_decay_schedule_values=encode_value(
            weight_decay_schedule_values, path="weight_decay_schedule_values"
        ),
        reservation_manifest=encode_value(reservation_manifest, path="reservation_manifest"),
        runtime=RuntimeRecord(
            semantic=encode_value(runtime_semantic, path="runtime.semantic"),
            authority_local=encode_value(runtime_authority_local, path="runtime.authority_local"),
        ),
    )
    validate_training_state(state)
    return state


def _validate_tensor_record(record: TensorRecord, path: str) -> None:
    metadata = record.metadata
    if (
        metadata.element_size <= 0
        or any(dimension < 0 for dimension in metadata.shape)
        or len(metadata.shape) != len(metadata.stride)
        or metadata.layout != str(torch.strided)
    ):
        raise QualificationStateError(f"invalid tensor metadata at {path}.metadata")
    expected_size = math.prod(metadata.shape) * metadata.element_size
    if len(record.content) != expected_size:
        raise QualificationStateError(f"tensor byte length mismatch at {path}.content")
    digest = sha256(record.content).hexdigest()
    if digest != record.content_sha256:
        raise QualificationStateError(f"content digest mismatch at {path}.content_sha256")


def validate_training_state(state: TrainingState) -> None:
    """Validate archive integrity and canonical uniqueness invariants."""
    if state.schema_version != _SCHEMA_VERSION:
        raise QualificationStateError(f"unsupported qualification schema version: {state.schema_version}")
    module_names = tuple(item.name for item in state.modules)
    parameter_names = tuple(item.name for item in state.parameters)
    buffer_names = tuple(item.name for item in state.buffers)
    if len(set(module_names)) != len(module_names) or not module_names or module_names[0] != "":
        raise QualificationStateError("module names are not a unique rooted inventory")
    if len(set(parameter_names)) != len(parameter_names):
        raise QualificationStateError("duplicate parameter slot names")
    if len(set(buffer_names)) != len(buffer_names):
        raise QualificationStateError("duplicate buffer slot names")

    parameter_aliases: set[str] = set()
    frozen_by_alias: dict[str, bool] = {}
    for index, parameter_record in enumerate(state.parameters):
        path = f"parameters[{index}]"
        if not parameter_record.present:
            if any(
                value is not None
                for value in (
                    parameter_record.frozen,
                    parameter_record.trainable,
                    parameter_record.requires_grad,
                    parameter_record.object_alias,
                    parameter_record.storage_alias,
                    parameter_record.tensor,
                )
            ):
                raise QualificationStateError(f"inconsistent absent slot at {path}")
            continue
        if (
            parameter_record.tensor is None
            or parameter_record.object_alias is None
            or parameter_record.storage_alias is None
            or parameter_record.requires_grad is None
            or parameter_record.trainable != parameter_record.requires_grad
            or parameter_record.frozen != (not parameter_record.requires_grad)
        ):
            raise QualificationStateError(f"inconsistent present slot at {path}")
        if not parameter_record.object_alias.startswith("parameter:"):
            raise QualificationStateError(f"invalid parameter object alias at {path}.object_alias")
        alias = parameter_record.object_alias.removeprefix("parameter:")
        if alias in frozen_by_alias and frozen_by_alias[alias] != parameter_record.frozen:
            raise QualificationStateError(f"inconsistent tied parameter status at {path}")
        parameter_aliases.add(alias)
        frozen_by_alias[alias] = cast(bool, parameter_record.frozen)
        if parameter_record.tensor.metadata.requires_grad != parameter_record.requires_grad:
            raise QualificationStateError(f"inconsistent tensor requires_grad at {path}")
        _validate_tensor_record(parameter_record.tensor, f"{path}.tensor")

    for index, buffer_record in enumerate(state.buffers):
        path = f"buffers[{index}]"
        if not buffer_record.present:
            if any(
                value is not None
                for value in (buffer_record.object_alias, buffer_record.storage_alias, buffer_record.tensor)
            ):
                raise QualificationStateError(f"inconsistent absent slot at {path}")
            continue
        if (
            buffer_record.tensor is None
            or buffer_record.object_alias is None
            or buffer_record.storage_alias is None
        ):
            raise QualificationStateError(f"inconsistent present slot at {path}")
        _validate_tensor_record(buffer_record.tensor, f"{path}.tensor")

    expected_gradient_aliases = tuple(sorted(parameter_aliases))
    for name, snapshot in (
        ("raw_gradients", state.raw_gradients),
        ("post_clip_gradients", state.post_clip_gradients),
    ):
        actual_aliases = tuple(record.parameter_alias for record in snapshot.gradients)
        if actual_aliases != expected_gradient_aliases:
            raise QualificationStateError(f"{name} is not the complete canonical parameter alias inventory")
        for index, gradient_record in enumerate(snapshot.gradients):
            if gradient_record.frozen != frozen_by_alias[gradient_record.parameter_alias]:
                raise QualificationStateError(f"inconsistent frozen gradient status at {name}[{index}]")
            if gradient_record.frozen and gradient_record.tensor is not None:
                raise QualificationStateError(f"frozen gradient is not None at {name}[{index}]")
            if gradient_record.tensor is not None:
                _validate_tensor_record(gradient_record.tensor, f"{name}[{index}].tensor")

    memberships = tuple(alias for group in state.optimizer.groups for alias in group.parameter_aliases)
    if len(memberships) != len(set(memberships)):
        raise QualificationStateError("optimizer groups contain ambiguous duplicate membership")
    if not set(memberships).issubset(parameter_aliases):
        raise QualificationStateError("optimizer groups contain a foreign parameter alias")
    state_aliases = tuple(entry.parameter_alias for entry in state.optimizer.state)
    if state_aliases != tuple(sorted(state_aliases)) or len(state_aliases) != len(set(state_aliases)):
        raise QualificationStateError("optimizer state aliases are not canonical and unique")
    if not set(state_aliases).issubset(parameter_aliases):
        raise QualificationStateError("optimizer state contains a foreign parameter alias")
    for entry_index, entry in enumerate(state.optimizer.state):
        _validate_value(entry.state, f"optimizer.state[{entry_index}].state")
    _validate_value(state.optimizer.defaults, "optimizer.defaults")
    for index, group in enumerate(state.optimizer.groups):
        _validate_value(group.options, f"optimizer.groups[{index}].options")

    _validate_value(state.rng.python, "rng.python")
    if state.rng.numpy is not None:
        _validate_value(state.rng.numpy, "rng.numpy")
    _validate_tensor_record(state.rng.torch_cpu, "rng.torch_cpu")
    if state.rng.cuda.disposition != "captured" and state.rng.cuda.device_states:
        raise QualificationStateError("uncaptured CUDA RNG disposition has device states")
    for index, tensor in enumerate(state.rng.cuda.device_states):
        _validate_tensor_record(tensor, f"rng.cuda.device_states[{index}]")
    for path, value in (
        ("cursor_before", state.cursor_before),
        ("cursor_after", state.cursor_after),
        ("lr_schedule_values", state.lr_schedule_values),
        ("weight_decay_schedule_values", state.weight_decay_schedule_values),
        ("reservation_manifest", state.reservation_manifest),
        ("runtime.semantic", state.runtime.semantic),
        ("runtime.authority_local", state.runtime.authority_local),
    ):
        _validate_value(value, path)


def _validate_value(value: CanonicalValue, path: str) -> None:
    expected_tags: dict[type[object], str] = {
        NullValue: "null",
        BoolValue: "bool",
        IntValue: "int",
        FloatValue: "float",
        StringValue: "string",
        BytesValue: "bytes",
        TorchDTypeValue: "torch_dtype",
        TorchDeviceValue: "torch_device",
        SequenceValue: "sequence",
        MappingValue: "mapping",
        TensorValue: "tensor",
        NumpyArrayValue: "numpy_array",
    }
    expected_tag = expected_tags.get(type(value))
    if expected_tag is None or getattr(value, "tag", None) != expected_tag:
        raise QualificationStateError(f"invalid tagged value at {path}")
    if isinstance(value, FloatValue):
        try:
            decoded = float.fromhex(value.hexadecimal)
        except ValueError as error:
            raise QualificationStateError(f"invalid float at {path}") from error
        if not math.isfinite(decoded) or decoded.hex() != value.hexadecimal:
            raise QualificationStateError(f"noncanonical float at {path}")
    elif isinstance(value, TensorValue):
        _validate_tensor_record(value.tensor, f"{path}.tensor")
    elif isinstance(value, NumpyArrayValue):
        if any(dimension < 0 for dimension in value.shape):
            raise QualificationStateError(f"invalid NumPy shape at {path}.shape")
        if sha256(value.content).hexdigest() != value.content_sha256:
            raise QualificationStateError(f"content digest mismatch at {path}.content_sha256")
    elif isinstance(value, SequenceValue):
        if value.sequence_type not in ("list", "tuple"):
            raise QualificationStateError(f"invalid sequence type at {path}")
        for index, item in enumerate(value.items):
            _validate_value(item, f"{path}.items[{index}]")
    elif isinstance(value, MappingValue):
        keys = tuple(entry.key for entry in value.entries)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise QualificationStateError(f"noncanonical mapping entries at {path}")
        for entry in value.entries:
            if not isinstance(entry.key, str):
                raise QualificationStateError(f"non-string mapping key at {path}")
            _validate_value(entry.value, f"{path}.{entry.key}")


def _summary(value: object) -> str:
    if isinstance(value, bytes):
        return f"bytes(len={len(value)},sha256={sha256(value).hexdigest()})"
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


def _kind_for(path: str, *, structural: bool = False, byte_value: bool = False) -> MismatchKind:
    if structural:
        return "structural"
    if byte_value or path.endswith(".content") or path.endswith(".value") and "BytesValue" in path:
        return "bytes"
    leaf = path.rsplit(".", 1)[-1]
    if "alias" in leaf:
        return "alias"
    metadata_fields = {field.name for field in fields(TensorMetadata)} | {
        "class_identity",
        "training",
        "trainable",
        "requires_grad",
        "persistent",
        "present",
        "content_sha256",
    }
    if leaf in metadata_fields or ".metadata." in path or ".class_identity." in path:
        return "metadata"
    return "value"


def _compare(expected: object, actual: object, path: str, output: list[StateMismatch]) -> None:
    if type(expected) is not type(actual):
        output.append(StateMismatch(path, "structural", type(expected).__name__, type(actual).__name__))
        return
    if isinstance(expected, bytes):
        if expected != actual:
            output.append(StateMismatch(path, "bytes", _summary(expected), _summary(actual)))
        return
    if is_dataclass(expected) and not isinstance(expected, type):
        for field in fields(expected):
            _compare(getattr(expected, field.name), getattr(actual, field.name), f"{path}.{field.name}", output)
        return
    if isinstance(expected, tuple):
        actual_tuple = cast(tuple[object, ...], actual)
        if len(expected) != len(actual_tuple):
            output.append(StateMismatch(path, "structural", f"length={len(expected)}", f"length={len(actual_tuple)}"))
        for index, (left, right) in enumerate(zip(expected, actual_tuple)):
            _compare(left, right, f"{path}[{index}]", output)
        return
    if expected != actual:
        output.append(StateMismatch(path, _kind_for(path), _summary(expected), _summary(actual)))


def _semantic_projection(state: TrainingState) -> TrainingState:
    return replace(state, runtime=replace(state.runtime, authority_local=NullValue()))


def compare_semantic_state(expected: TrainingState, actual: TrainingState) -> ComparisonResult:
    """Compare semantic state; authority-local runtime data is excluded by API."""
    mismatches: list[StateMismatch] = []
    _compare(_semantic_projection(expected), _semantic_projection(actual), "state", mismatches)
    return ComparisonResult(tuple(mismatches))


def compare_authority_local(expected: TrainingState, actual: TrainingState) -> ComparisonResult:
    """Compare only explicitly non-semantic, authority-local runtime data."""
    mismatches: list[StateMismatch] = []
    _compare(expected.runtime.authority_local, actual.runtime.authority_local, "authority_local", mismatches)
    return ComparisonResult(tuple(mismatches))


def semantic_digest(state: TrainingState) -> str:
    """Digest semantic state without authority-local runtime fields."""
    return sha256(serialize_training_state(_semantic_projection(state))).hexdigest()


_SERIALIZABLE_CLASSES: tuple[type[Any], ...] = (
    ClassIdentity,
    TensorMetadata,
    TensorRecord,
    NullValue,
    BoolValue,
    IntValue,
    FloatValue,
    StringValue,
    BytesValue,
    TorchDTypeValue,
    TorchDeviceValue,
    SequenceValue,
    MappingEntry,
    MappingValue,
    TensorValue,
    NumpyArrayValue,
    ModuleRecord,
    ParameterSlot,
    BufferSlot,
    GradientRecord,
    GradientSnapshot,
    OptimizerGroup,
    OptimizerStateEntry,
    OptimizerRecord,
    CudaRngState,
    RngState,
    RuntimeRecord,
    TrainingState,
)


def _manifest_encode(value: object, blobs: list[bytes]) -> object:
    if isinstance(value, bytes):
        index = len(blobs)
        blobs.append(value)
        return {"$blob": index, "length": len(value), "sha256": sha256(value).hexdigest()}
    if is_dataclass(value) and not isinstance(value, type):
        output: dict[str, object] = {"$type": type(value).__name__}
        for field in fields(value):
            output[field.name] = _manifest_encode(getattr(value, field.name), blobs)
        return output
    if isinstance(value, tuple):
        return {"$tuple": [_manifest_encode(item, blobs) for item in value]}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise QualificationStateError(f"internal unsupported archive value: {type(value).__name__}")


def serialize_training_state(state: TrainingState) -> bytes:
    """Serialize state deterministically, keeping binary payloads outside JSON."""
    validate_training_state(state)
    blobs: list[bytes] = []
    manifest_object = _manifest_encode(state, blobs)
    manifest = json.dumps(manifest_object, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    archive = bytearray(_ARCHIVE_MAGIC)
    archive.extend(struct.pack(">Q", len(manifest)))
    archive.extend(manifest)
    archive.extend(struct.pack(">Q", len(blobs)))
    for blob in blobs:
        archive.extend(struct.pack(">Q", len(blob)))
        archive.extend(blob)
    return bytes(archive)


def _manifest_decode(value: object, blobs: Sequence[bytes]) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if not isinstance(value, dict):
        raise QualificationStateError("malformed qualification manifest")
    if "$blob" in value:
        index = value["$blob"]
        length = value.get("length")
        digest = value.get("sha256")
        if not isinstance(index, int) or not 0 <= index < len(blobs):
            raise QualificationStateError("invalid blob index in qualification manifest")
        blob = blobs[index]
        if length != len(blob) or digest != sha256(blob).hexdigest():
            raise QualificationStateError("qualification blob integrity failure")
        return blob
    if "$tuple" in value:
        items = value["$tuple"]
        if not isinstance(items, list):
            raise QualificationStateError("malformed tuple in qualification manifest")
        return tuple(_manifest_decode(item, blobs) for item in items)
    type_name = value.get("$type")
    cls = (
        next((candidate for candidate in _SERIALIZABLE_CLASSES if candidate.__name__ == type_name), None)
        if isinstance(type_name, str)
        else None
    )
    if cls is None:
        raise QualificationStateError(f"unknown qualification manifest type: {type_name!r}")
    expected_fields = {field.name for field in fields(cls)}
    supplied_fields = set(value) - {"$type"}
    if supplied_fields != expected_fields:
        raise QualificationStateError(f"wrong fields for qualification manifest type {type_name}")
    arguments = {name: _manifest_decode(value[name], blobs) for name in expected_fields}
    return cast(Any, cls)(**arguments)


def deserialize_training_state(payload: bytes) -> TrainingState:
    """Deserialize a deterministic archive and fail closed on any corruption."""
    if not payload.startswith(_ARCHIVE_MAGIC):
        raise QualificationStateError("not a GCW qualification state archive")
    position = len(_ARCHIVE_MAGIC)

    def take_u64() -> int:
        nonlocal position
        if position + 8 > len(payload):
            raise QualificationStateError("truncated qualification state archive")
        result = struct.unpack(">Q", payload[position : position + 8])[0]
        position += 8
        return result

    manifest_length = take_u64()
    if position + manifest_length > len(payload):
        raise QualificationStateError("truncated qualification manifest")
    manifest_bytes = payload[position : position + manifest_length]
    position += manifest_length
    blob_count = take_u64()
    blobs: list[bytes] = []
    for _ in range(blob_count):
        length = take_u64()
        if position + length > len(payload):
            raise QualificationStateError("truncated qualification blob")
        blobs.append(payload[position : position + length])
        position += length
    if position != len(payload):
        raise QualificationStateError("trailing bytes in qualification state archive")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationStateError("invalid qualification manifest JSON") from error
    decoded = _manifest_decode(manifest, blobs)
    if not isinstance(decoded, TrainingState):
        raise QualificationStateError("qualification archive root is not TrainingState")
    validate_training_state(decoded)
    return decoded
