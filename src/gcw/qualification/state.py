"""Canonical, fail-closed capture of complete training state for qualification.

This module deliberately has no integration with the training loop.  Qualification
callers take gradient snapshots and capture state outside measured regions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
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
    storage_nbytes: int
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
    object_alias: str = ""
    tag: Literal["sequence"] = "sequence"


@dataclass(frozen=True)
class MappingEntry:
    key: str
    value: CanonicalValue


@dataclass(frozen=True)
class MappingValue:
    entries: tuple[MappingEntry, ...]
    object_alias: str = ""
    tag: Literal["mapping"] = "mapping"


@dataclass(frozen=True)
class EnumValue:
    class_identity: ClassIdentity
    member_name: str
    value: CanonicalValue
    object_alias: str = ""
    tag: Literal["enum"] = "enum"


@dataclass(frozen=True)
class NamedTupleValue:
    class_identity: ClassIdentity
    fields: tuple[MappingEntry, ...]
    object_alias: str = ""
    tag: Literal["namedtuple"] = "namedtuple"


@dataclass(frozen=True)
class DataclassValue:
    class_identity: ClassIdentity
    fields: tuple[MappingEntry, ...]
    object_alias: str = ""
    tag: Literal["dataclass"] = "dataclass"


@dataclass(frozen=True)
class TensorValue:
    tensor: TensorRecord
    object_alias: str = ""
    storage_alias: str = ""
    tag: Literal["tensor"] = "tensor"


@dataclass(frozen=True)
class NumpyScalarValue:
    dtype: str
    content: bytes
    content_sha256: str
    tag: Literal["numpy_scalar"] = "numpy_scalar"


@dataclass(frozen=True)
class NumpyArrayValue:
    dtype: str
    shape: tuple[int, ...]
    content: bytes
    content_sha256: str
    strides: tuple[int, ...] = ()
    storage_offset: int = 0
    storage_nbytes: int = 0
    writeable: bool = True
    object_alias: str = ""
    storage_alias: str = ""
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
    | EnumValue
    | NamedTupleValue
    | DataclassValue
    | TensorValue
    | NumpyScalarValue
    | NumpyArrayValue
)


@dataclass(frozen=True)
class ModuleRecord:
    name: str
    object_alias: str
    class_identity: ClassIdentity
    training: bool
    has_extra_state: bool


@dataclass(frozen=True)
class ModuleExtraStateRecord:
    module_alias: str
    state: CanonicalValue


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
    module_extra_states: tuple[ModuleExtraStateRecord, ...]
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


_TensorStorageKey: TypeAlias = tuple[str, int | None, int]


@dataclass
class _ValueEncodingContext:
    active: set[int]
    object_alias_by_id: dict[int, str]
    object_references: dict[int, object]
    tensor_storage_aliases: dict[_TensorStorageKey, str]
    tensor_storage_references: dict[_TensorStorageKey, torch.Tensor]
    numpy_storage_aliases: dict[int, str]
    numpy_storage_references: dict[int, object]
    next_object_alias: int = 0
    next_storage_alias: int = 0

    def object_alias(self, value: object, path: str) -> str:
        marker = id(value)
        reference = self.object_references.get(marker)
        if reference is not None and reference is not value:  # pragma: no cover - retained refs prevent reuse
            raise QualificationStateError(f"object identity reuse while encoding {path}")
        alias = self.object_alias_by_id.get(marker)
        if alias is None:
            alias = f"value-object:{self.next_object_alias:08d}"
            self.next_object_alias += 1
            self.object_alias_by_id[marker] = alias
        self.object_references[marker] = value
        return alias

    def _new_storage_alias(self) -> str:
        alias = f"value-storage:{self.next_storage_alias:08d}"
        self.next_storage_alias += 1
        return alias

    def tensor_storage_alias(self, tensor: torch.Tensor) -> str:
        storage = tensor.untyped_storage()
        key: _TensorStorageKey = (tensor.device.type, tensor.device.index, int(storage._cdata))
        alias = self.tensor_storage_aliases.get(key)
        if alias is None:
            alias = self._new_storage_alias()
            self.tensor_storage_aliases[key] = alias
        self.tensor_storage_references[key] = tensor
        return alias

    def numpy_storage_alias(self, owner: object) -> str:
        marker = id(owner)
        reference = self.numpy_storage_references.get(marker)
        if reference is not None and reference is not owner:  # pragma: no cover - retained refs prevent reuse
            raise QualificationStateError("NumPy storage owner identity reuse")
        alias = self.numpy_storage_aliases.get(marker)
        if alias is None:
            alias = self._new_storage_alias()
            self.numpy_storage_aliases[marker] = alias
        self.numpy_storage_references[marker] = owner
        return alias


def _new_value_encoding_context(
    *,
    object_aliases: Mapping[int, str] | None = None,
    object_references: Mapping[int, object] | None = None,
    tensor_storage_aliases: Mapping[_TensorStorageKey, str] | None = None,
    tensor_storage_references: Mapping[_TensorStorageKey, torch.Tensor] | None = None,
) -> _ValueEncodingContext:
    aliases = dict(object_aliases or {})
    references = dict(object_references or {})
    if aliases.keys() != references.keys():
        raise QualificationStateError("value context object aliases and references differ")
    storage_aliases = dict(tensor_storage_aliases or {})
    storage_references = dict(tensor_storage_references or {})
    if storage_aliases.keys() != storage_references.keys():
        raise QualificationStateError("value context storage aliases and references differ")
    return _ValueEncodingContext(
        active=set(),
        object_alias_by_id=aliases,
        object_references=references,
        tensor_storage_aliases=storage_aliases,
        tensor_storage_references=storage_references,
        numpy_storage_aliases={},
        numpy_storage_references={},
    )


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
) -> tuple[_ParameterInventory, _BufferInventory, Mapping[int, str], Mapping[_TensorStorageKey, str]]:
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
    names_by_storage: dict[_TensorStorageKey, list[str]] = {}
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
        storage_nbytes=int(tensor.untyped_storage().nbytes()),
        layout=str(tensor.layout),
        requires_grad=tensor.requires_grad,
        is_conj=tensor.is_conj(),
        is_neg=tensor.is_neg(),
    )
    return TensorRecord(metadata=metadata, content=content, content_sha256=sha256(content).hexdigest())


def _numpy_view_bounds(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    storage_offset: int,
    itemsize: int,
    storage_nbytes: int,
    path: str,
) -> None:
    if len(shape) != len(strides):
        raise QualificationStateError(f"NumPy rank mismatch at {path}")
    if storage_offset < 0 or storage_nbytes < 0:
        raise QualificationStateError(f"invalid NumPy storage metadata at {path}")
    if any(dimension == 0 for dimension in shape):
        if storage_offset > storage_nbytes:
            raise QualificationStateError(f"NumPy empty view exceeds storage at {path}")
        return
    lower = storage_offset
    upper = storage_offset
    for dimension, stride in zip(shape, strides):
        extent = (dimension - 1) * stride
        lower += min(0, extent)
        upper += max(0, extent)
    if lower < 0 or upper + itemsize > storage_nbytes:
        raise QualificationStateError(f"NumPy view exceeds captured storage at {path}")


def _numpy_storage_details(array: object, np: Any, path: str) -> tuple[object, int, int]:
    root = array
    seen: set[int] = set()
    while isinstance(getattr(root, "base", None), np.ndarray):
        marker = id(root)
        if marker in seen:
            raise QualificationStateError(f"NumPy base cycle at {path}")
        seen.add(marker)
        root = getattr(root, "base")
    base = getattr(root, "base", None)
    if base is None:
        owner = root
        lower, upper = np.lib.array_utils.byte_bounds(root)
        return owner, int(lower), int(upper - lower)

    owner = base
    while isinstance(owner, memoryview):
        next_owner = owner.obj
        if next_owner is owner:
            break
        owner = next_owner
    try:
        owner_bytes = np.frombuffer(owner, dtype=np.uint8)
    except (BufferError, TypeError, ValueError) as error:
        raise QualificationStateError(f"unsupported NumPy storage owner at {path}") from error
    if not owner_bytes.flags.c_contiguous:  # pragma: no cover - frombuffer guarantees this
        raise QualificationStateError(f"noncontiguous NumPy storage owner at {path}")
    pointer = int(owner_bytes.__array_interface__["data"][0])
    return owner, pointer, int(owner_bytes.nbytes)


def encode_value(value: object, *, path: str = "value") -> CanonicalValue:
    """Recursively encode a deterministic value, preserving aliases and rejecting cycles."""
    return _encode_value(value, path=path, context=_new_value_encoding_context())


def _encode_value(value: object, *, path: str, context: _ValueEncodingContext) -> CanonicalValue:
    if value is None:
        return NullValue()
    if isinstance(value, Enum):
        marker = id(value)
        if marker in context.active:
            raise QualificationStateError(f"cycle at {path}")
        context.active.add(marker)
        object_alias = context.object_alias(value, path)
        try:
            return EnumValue(
                class_identity=_identity(value),
                member_name=value.name,
                value=_encode_value(value.value, path=f"{path}.value", context=context),
                object_alias=object_alias,
            )
        finally:
            context.active.remove(marker)
    if type(value) is bool:
        return BoolValue(value)
    if type(value) is int:
        return IntValue(value)
    if type(value) is float:
        return FloatValue(struct.pack(">d", value).hex())
    if type(value) is str:
        return StringValue(value)
    if type(value) is bytes:
        return BytesValue(value)
    if isinstance(value, torch.dtype):
        return TorchDTypeValue(str(value))
    if isinstance(value, torch.device):
        return TorchDeviceValue(value.type, value.index)
    if isinstance(value, torch.Tensor):
        return TensorValue(
            tensor=_capture_tensor(value, path),
            object_alias=context.object_alias(value, path),
            storage_alias=context.tensor_storage_alias(value),
        )

    # NumPy is optional to the qualification substrate at import time.
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised in environments without NumPy
        np = None  # type: ignore[assignment]
    if np is not None and isinstance(value, np.ndarray):
        if type(value) is not np.ndarray:
            raise QualificationStateError(f"NumPy array subclass at {path} is unsupported")
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise QualificationStateError(f"object or structured NumPy array at {path} is unsupported")
        owner, storage_pointer, storage_nbytes = _numpy_storage_details(value, np, path)
        pointer = int(value.__array_interface__["data"][0])
        shape = tuple(int(dimension) for dimension in value.shape)
        strides = tuple(int(stride) for stride in value.strides)
        storage_offset = pointer - storage_pointer
        _numpy_view_bounds(shape, strides, storage_offset, int(value.dtype.itemsize), storage_nbytes, path)
        copied = np.ascontiguousarray(value).copy()
        content = copied.tobytes(order="C")
        return NumpyArrayValue(
            dtype=copied.dtype.str,
            shape=shape,
            content=content,
            content_sha256=sha256(content).hexdigest(),
            strides=strides,
            storage_offset=storage_offset,
            storage_nbytes=storage_nbytes,
            writeable=bool(value.flags.writeable),
            object_alias=context.object_alias(value, path),
            storage_alias=context.numpy_storage_alias(owner),
        )
    if np is not None and isinstance(value, np.generic):
        scalar = np.asarray(value)
        if scalar.dtype.hasobject or scalar.dtype.fields is not None:
            raise QualificationStateError(f"object or structured NumPy scalar at {path} is unsupported")
        content = scalar.tobytes(order="C")
        return NumpyScalarValue(scalar.dtype.str, content, sha256(content).hexdigest())

    named_fields = getattr(type(value), "_fields", None)
    if isinstance(value, tuple) and isinstance(named_fields, tuple):
        if any(not isinstance(name, str) for name in named_fields) or len(set(named_fields)) != len(named_fields):
            raise QualificationStateError(f"invalid namedtuple fields at {path}")
        marker = id(value)
        if marker in context.active:
            raise QualificationStateError(f"cycle at {path}")
        context.active.add(marker)
        object_alias = context.object_alias(value, path)
        try:
            encoded_fields = tuple(
                MappingEntry(
                    name,
                    _encode_value(getattr(value, name), path=f"{path}.{name}", context=context),
                )
                for name in sorted(named_fields)
            )
            return NamedTupleValue(_identity(value), encoded_fields, object_alias=object_alias)
        finally:
            context.active.remove(marker)
    if is_dataclass(value) and not isinstance(value, type):
        marker = id(value)
        if marker in context.active:
            raise QualificationStateError(f"cycle at {path}")
        context.active.add(marker)
        object_alias = context.object_alias(value, path)
        try:
            encoded_fields = tuple(
                MappingEntry(
                    field.name,
                    _encode_value(getattr(value, field.name), path=f"{path}.{field.name}", context=context),
                )
                for field in sorted(fields(value), key=lambda item: item.name)
            )
            return DataclassValue(_identity(value), encoded_fields, object_alias=object_alias)
        finally:
            context.active.remove(marker)

    if isinstance(value, Mapping):
        if type(value) is not dict:
            raise QualificationStateError(f"mapping subclass at {path} is unsupported")
        marker = id(value)
        if marker in context.active:
            raise QualificationStateError(f"cycle at {path}")
        keys = tuple(value)
        if any(type(key) is not str for key in keys):
            raise QualificationStateError(f"non-string mapping key at {path}")
        if any(not key for key in keys):
            raise QualificationStateError(f"empty mapping key at {path}")
        context.active.add(marker)
        object_alias = context.object_alias(value, path)
        try:
            entries = tuple(
                MappingEntry(key, _encode_value(value[key], path=f"{path}.{key}", context=context))
                for key in sorted(keys)
            )
            return MappingValue(entries, object_alias=object_alias)
        finally:
            context.active.remove(marker)
    if type(value) in {list, tuple}:
        marker = id(value)
        if marker in context.active:
            raise QualificationStateError(f"cycle at {path}")
        context.active.add(marker)
        object_alias = context.object_alias(value, path)
        try:
            items = cast(list[object] | tuple[object, ...], value)
            sequence_type: Literal["list", "tuple"] = "list" if type(value) is list else "tuple"
            return SequenceValue(
                sequence_type,
                tuple(
                    _encode_value(item, path=f"{path}[{index}]", context=context) for index, item in enumerate(items)
                ),
                object_alias=object_alias,
            )
        finally:
            context.active.remove(marker)
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
        gradient = parameter.grad
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
        if record.tensor is not None:
            _validate_tensor_record(record.tensor, f"{path}.{record.parameter_alias}.tensor")


def _capture_optimizer(
    optimizer: torch.optim.Optimizer,
    inventory: _ParameterInventory,
    value_context: _ValueEncodingContext,
) -> OptimizerRecord:
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
        encoded_options = _encode_value(options, path=f"optimizer.groups[{group_index}].options", context=value_context)
        if not isinstance(encoded_options, MappingValue):  # pragma: no cover - construction invariant
            raise AssertionError("optimizer options must encode as a mapping")
        groups.append(OptimizerGroup(tuple(membership), encoded_options))

    state_values: list[tuple[str, object]] = []
    for parameter, value in optimizer.state.items():
        if not isinstance(parameter, torch.Tensor) or id(parameter) not in aliases:
            raise QualificationStateError("optimizer.state contains a parameter not owned by the model")
        state_values.append((aliases[id(parameter)], value))
    state_values.sort(key=lambda item: item[0])
    if len({alias for alias, _ in state_values}) != len(state_values):
        raise QualificationStateError("optimizer.state has ambiguous duplicate parameter keys")
    state_entries = tuple(
        OptimizerStateEntry(
            alias,
            _encode_value(value, path=f"optimizer.state.{alias}", context=value_context),
        )
        for alias, value in state_values
    )
    defaults = _encode_value(optimizer.defaults, path="optimizer.defaults", context=value_context)
    if not isinstance(defaults, MappingValue):  # pragma: no cover - construction invariant
        raise AssertionError("optimizer defaults must encode as a mapping")
    return OptimizerRecord(_identity(optimizer), defaults, tuple(groups), state_entries)


def _capture_rng(*, capture_cuda_rng: bool, value_context: _ValueEncodingContext) -> RngState:
    python_state = _encode_value(random.getstate(), path="rng.python", context=value_context)
    try:
        import numpy as np
    except ImportError:
        numpy_state = None
    else:
        numpy_state = _encode_value(np.random.get_state(), path="rng.numpy", context=value_context)
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
    registered_tensors = [tensor for _, tensor in parameter_inventory.slots if tensor is not None] + [
        tensor for _, tensor, _ in buffer_inventory.slots if tensor is not None
    ]
    object_references = {id(tensor): tensor for tensor in registered_tensors}
    storage_references: dict[_TensorStorageKey, torch.Tensor] = {
        (tensor.device.type, tensor.device.index, int(tensor.untyped_storage()._cdata)): tensor
        for tensor in registered_tensors
    }
    value_context = _new_value_encoding_context(
        object_aliases=object_aliases,
        object_references=object_references,
        tensor_storage_aliases=storage_aliases,
        tensor_storage_references=storage_references,
    )

    names_by_module_id: dict[int, list[str]] = {}
    for name, module in modules:
        names_by_module_id.setdefault(id(module), []).append(name or "<root>")
    has_extra_state_by_id = {
        id(module): type(module).get_extra_state is not nn.Module.get_extra_state for _, module in modules
    }
    module_records = tuple(
        ModuleRecord(
            name=name,
            object_alias=min(names_by_module_id[id(module)]),
            class_identity=_identity(module),
            training=module.training,
            has_extra_state=has_extra_state_by_id[id(module)],
        )
        for name, module in modules
    )
    canonical_modules: dict[str, nn.Module] = {}
    for _, module in modules:
        canonical_modules.setdefault(min(names_by_module_id[id(module)]), module)
    module_extra_states: list[ModuleExtraStateRecord] = []
    for module_alias, module in sorted(canonical_modules.items()):
        if type(module).get_extra_state is nn.Module.get_extra_state:
            continue
        try:
            extra_state = module.get_extra_state()
        except Exception as error:
            raise QualificationStateError(f"module extra state capture failed at modules.{module_alias}") from error
        module_extra_states.append(
            ModuleExtraStateRecord(
                module_alias,
                _encode_value(
                    extra_state,
                    path=f"module_extra_states.{module_alias}",
                    context=value_context,
                ),
            )
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
        module_extra_states=tuple(module_extra_states),
        parameters=tuple(parameter_records),
        buffers=tuple(buffer_records),
        raw_gradients=raw_gradients,
        post_clip_gradients=post_clip_gradients,
        optimizer=_capture_optimizer(optimizer, parameter_inventory, value_context),
        rng=_capture_rng(capture_cuda_rng=capture_cuda_rng, value_context=value_context),
        cursor_before=_encode_value(cursor_before, path="cursor_before", context=value_context),
        cursor_after=_encode_value(cursor_after, path="cursor_after", context=value_context),
        step_ordinal=step_ordinal,
        lr_schedule_values=_encode_value(lr_schedule_values, path="lr_schedule_values", context=value_context),
        weight_decay_schedule_values=_encode_value(
            weight_decay_schedule_values,
            path="weight_decay_schedule_values",
            context=value_context,
        ),
        reservation_manifest=_encode_value(reservation_manifest, path="reservation_manifest", context=value_context),
        runtime=RuntimeRecord(
            semantic=_encode_value(runtime_semantic, path="runtime.semantic", context=value_context),
            authority_local=_encode_value(
                runtime_authority_local,
                path="runtime.authority_local",
                context=value_context,
            ),
        ),
    )
    validate_training_state(state)
    return state


def _validate_class_identity(identity: ClassIdentity, path: str) -> None:
    if (
        type(identity) is not ClassIdentity
        or type(identity.module) is not str
        or not identity.module
        or type(identity.qualname) is not str
        or not identity.qualname
    ):
        raise QualificationStateError(f"invalid class identity at {path}")


def _validate_tensor_record(record: TensorRecord, path: str) -> None:
    if type(record) is not TensorRecord or type(record.metadata) is not TensorMetadata:
        raise QualificationStateError(f"invalid tensor record at {path}")
    metadata = record.metadata
    _validate_class_identity(metadata.tensor_class, f"{path}.metadata.tensor_class")
    dtype_name = metadata.dtype.removeprefix("torch.") if type(metadata.dtype) is str else ""
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype) or str(dtype) != metadata.dtype:
        raise QualificationStateError(f"invalid tensor dtype at {path}.metadata.dtype")
    if type(metadata.device_type) is not str or not metadata.device_type:
        raise QualificationStateError(f"invalid tensor device type at {path}.metadata.device_type")
    if metadata.device_index is not None and (type(metadata.device_index) is not int or metadata.device_index < 0):
        raise QualificationStateError(f"invalid tensor device index at {path}.metadata.device_index")
    try:
        (
            torch.device(metadata.device_type)
            if metadata.device_index is None
            else torch.device(metadata.device_type, metadata.device_index)
        )
    except (RuntimeError, ValueError) as error:
        raise QualificationStateError(f"invalid tensor device at {path}.metadata") from error
    if type(metadata.element_size) is not int or metadata.element_size <= 0:
        raise QualificationStateError(f"invalid tensor element size at {path}.metadata.element_size")
    if dtype.itemsize != metadata.element_size:
        raise QualificationStateError(f"tensor dtype/element-size mismatch at {path}.metadata")
    if type(metadata.shape) is not tuple or any(
        type(dimension) is not int or dimension < 0 for dimension in metadata.shape
    ):
        raise QualificationStateError(f"invalid tensor shape at {path}.metadata.shape")
    if type(metadata.stride) is not tuple or any(type(value) is not int or value < 0 for value in metadata.stride):
        raise QualificationStateError(f"invalid tensor stride at {path}.metadata.stride")
    if len(metadata.shape) != len(metadata.stride):
        raise QualificationStateError(f"tensor rank mismatch at {path}.metadata")
    if type(metadata.storage_offset) is not int or metadata.storage_offset < 0:
        raise QualificationStateError(f"invalid tensor storage offset at {path}.metadata.storage_offset")
    if type(metadata.storage_nbytes) is not int or metadata.storage_nbytes < 0:
        raise QualificationStateError(f"invalid tensor storage size at {path}.metadata.storage_nbytes")
    if any(dimension == 0 for dimension in metadata.shape):
        if metadata.storage_offset * metadata.element_size > metadata.storage_nbytes:
            raise QualificationStateError(f"empty tensor view exceeds storage at {path}.metadata")
    else:
        final_element = metadata.storage_offset + sum(
            (dimension - 1) * stride for dimension, stride in zip(metadata.shape, metadata.stride)
        )
        if (final_element + 1) * metadata.element_size > metadata.storage_nbytes:
            raise QualificationStateError(f"tensor view exceeds storage at {path}.metadata")
    if metadata.layout != str(torch.strided):
        raise QualificationStateError(f"invalid tensor layout at {path}.metadata.layout")
    if any(type(value) is not bool for value in (metadata.requires_grad, metadata.is_conj, metadata.is_neg)):
        raise QualificationStateError(f"invalid tensor flags at {path}.metadata")
    if type(record.content) is not bytes or type(record.content_sha256) is not str:
        raise QualificationStateError(f"invalid tensor content fields at {path}")
    expected_size = math.prod(metadata.shape) * metadata.element_size
    if len(record.content) != expected_size:
        raise QualificationStateError(f"tensor byte length mismatch at {path}.content")
    digest = sha256(record.content).hexdigest()
    if digest != record.content_sha256:
        raise QualificationStateError(f"content digest mismatch at {path}.content_sha256")


def _validate_slot_aliases(
    object_groups: Mapping[str, list[str]],
    storage_groups: Mapping[str, list[str]],
    typed_names: set[str],
) -> None:
    for kind, groups in (("object", object_groups), ("storage", storage_groups)):
        for alias, members in groups.items():
            if alias not in typed_names or alias != min(members):
                raise QualificationStateError(f"noncanonical {kind} alias {alias!r}")


def validate_training_state(state: TrainingState) -> None:
    """Validate archive integrity, exact types, and canonical uniqueness invariants."""
    if type(state) is not TrainingState:
        raise QualificationStateError("qualification state root has an invalid type")
    if type(state.schema_version) is not int or state.schema_version != _SCHEMA_VERSION:
        raise QualificationStateError(f"unsupported qualification schema version: {state.schema_version!r}")
    if type(state.step_ordinal) is not int or state.step_ordinal < 0:
        raise QualificationStateError("step_ordinal must be a non-negative integer")
    if any(
        type(value) is not tuple
        for value in (
            state.modules,
            state.module_extra_states,
            state.parameters,
            state.buffers,
        )
    ):
        raise QualificationStateError("qualification inventories must be tuples")

    record: Any
    module_names: list[str] = []
    module_groups: dict[str, list[ModuleRecord]] = {}
    for index, record in enumerate(state.modules):
        path = f"modules[{index}]"
        if (
            type(record) is not ModuleRecord
            or type(record.name) is not str
            or type(record.object_alias) is not str
            or not record.object_alias
            or type(record.training) is not bool
            or type(record.has_extra_state) is not bool
        ):
            raise QualificationStateError(f"invalid module record at {path}")
        _validate_class_identity(record.class_identity, f"{path}.class_identity")
        module_names.append(record.name)
        module_groups.setdefault(record.object_alias, []).append(record)
    if len(set(module_names)) != len(module_names) or not module_names or module_names[0] != "":
        raise QualificationStateError("module names are not a unique rooted inventory")
    module_name_set = set(module_names)
    children: dict[str, list[str]] = {name: [] for name in module_names}
    for name in module_names[1:]:
        if any(not component for component in name.split(".")):
            raise QualificationStateError(f"invalid qualified module name {name!r}")
        parent = name.rpartition(".")[0]
        if parent not in module_name_set:
            raise QualificationStateError(f"module inventory is not prefix-closed at {name!r}")
        children[parent].append(name)
    expected_module_order: list[str] = []

    def visit_module(name: str) -> None:
        expected_module_order.append(name)
        for child in sorted(children[name]):
            visit_module(child)

    visit_module("")
    if module_names != expected_module_order:
        raise QualificationStateError("module inventory is not in canonical depth-first name order")
    canonical_module_names = {name or "<root>" for name in module_names}
    for alias, records in module_groups.items():
        names = [record.name or "<root>" for record in records]
        if alias not in canonical_module_names or alias != min(names):
            raise QualificationStateError(f"noncanonical module alias {alias!r}")
        first = records[0]
        if any(
            record.class_identity != first.class_identity
            or record.training != first.training
            or record.has_extra_state != first.has_extra_state
            for record in records[1:]
        ):
            raise QualificationStateError(f"inconsistent aliased module metadata for {alias!r}")

    expected_extra_aliases = tuple(
        sorted(alias for alias, records in module_groups.items() if records[0].has_extra_state)
    )
    actual_extra_aliases: list[str] = []
    for index, record in enumerate(state.module_extra_states):
        path = f"module_extra_states[{index}]"
        if type(record) is not ModuleExtraStateRecord or type(record.module_alias) is not str:
            raise QualificationStateError(f"invalid module extra-state record at {path}")
        actual_extra_aliases.append(record.module_alias)
    if tuple(actual_extra_aliases) != expected_extra_aliases:
        raise QualificationStateError("module extra state is not the complete canonical module inventory")

    parameter_names: set[str] = set()
    buffer_names: set[str] = set()
    parameter_name_order: list[str] = []
    buffer_name_order: list[str] = []
    parameter_aliases: set[str] = set()
    frozen_by_alias: dict[str, bool] = {}
    object_groups: dict[str, list[str]] = {}
    storage_groups: dict[str, list[str]] = {}
    typed_names: set[str] = set()
    for index, record in enumerate(state.parameters):
        path = f"parameters[{index}]"
        if type(record) is not ParameterSlot or type(record.name) is not str or not record.name:
            raise QualificationStateError(f"invalid parameter slot at {path}")
        if record.name in parameter_names:
            raise QualificationStateError("duplicate parameter slot names")
        parameter_names.add(record.name)
        parameter_name_order.append(record.name)
        if type(record.present) is not bool:
            raise QualificationStateError(f"invalid parameter presence at {path}.present")
        if not record.present:
            if any(
                value is not None
                for value in (
                    record.frozen,
                    record.trainable,
                    record.requires_grad,
                    record.object_alias,
                    record.storage_alias,
                    record.tensor,
                )
            ):
                raise QualificationStateError(f"inconsistent absent slot at {path}")
            continue
        if (
            type(record.requires_grad) is not bool
            or type(record.trainable) is not bool
            or type(record.frozen) is not bool
            or record.trainable != record.requires_grad
            or record.frozen != (not record.requires_grad)
            or type(record.object_alias) is not str
            or type(record.storage_alias) is not str
            or type(record.tensor) is not TensorRecord
        ):
            raise QualificationStateError(f"inconsistent present slot at {path}")
        typed_name = f"parameter:{record.name}"
        typed_names.add(typed_name)
        object_groups.setdefault(record.object_alias, []).append(typed_name)
        storage_groups.setdefault(record.storage_alias, []).append(typed_name)
        if not record.object_alias.startswith("parameter:"):
            raise QualificationStateError(f"invalid parameter object alias at {path}.object_alias")
        alias = record.object_alias.removeprefix("parameter:")
        if alias in frozen_by_alias and frozen_by_alias[alias] != record.frozen:
            raise QualificationStateError(f"inconsistent tied parameter status at {path}")
        parameter_aliases.add(alias)
        frozen_by_alias[alias] = record.frozen
        _validate_tensor_record(record.tensor, f"{path}.tensor")
        if record.tensor.metadata.requires_grad != record.requires_grad:
            raise QualificationStateError(f"inconsistent tensor requires_grad at {path}")

    for index, record in enumerate(state.buffers):
        path = f"buffers[{index}]"
        if (
            type(record) is not BufferSlot
            or type(record.name) is not str
            or not record.name
            or type(record.present) is not bool
            or type(record.persistent) is not bool
        ):
            raise QualificationStateError(f"invalid buffer slot at {path}")
        if record.name in buffer_names:
            raise QualificationStateError("duplicate buffer slot names")
        buffer_names.add(record.name)
        buffer_name_order.append(record.name)
        if not record.present:
            if any(value is not None for value in (record.object_alias, record.storage_alias, record.tensor)):
                raise QualificationStateError(f"inconsistent absent slot at {path}")
            continue
        if (
            type(record.object_alias) is not str
            or type(record.storage_alias) is not str
            or type(record.tensor) is not TensorRecord
        ):
            raise QualificationStateError(f"inconsistent present slot at {path}")
        typed_name = f"buffer:{record.name}"
        typed_names.add(typed_name)
        object_groups.setdefault(record.object_alias, []).append(typed_name)
        storage_groups.setdefault(record.storage_alias, []).append(typed_name)
        _validate_tensor_record(record.tensor, f"{path}.tensor")
    registration_conflicts = (parameter_names & buffer_names) | (
        set(module_names[1:]) & (parameter_names | buffer_names)
    )
    if registration_conflicts:
        raise QualificationStateError(f"module registration namespaces overlap: {sorted(registration_conflicts)!r}")

    def expected_slot_order(names: Sequence[str], kind: str) -> list[str]:
        by_module: dict[str, list[str]] = {name: [] for name in module_names}
        for name in names:
            if any(not component for component in name.split(".")):
                raise QualificationStateError(f"invalid qualified {kind} slot name: {name!r}")
            parent = name.rpartition(".")[0]
            if parent not in by_module:
                raise QualificationStateError(f"{kind} slot has an unknown module prefix: {name!r}")
            by_module[parent].append(name)
        return [name for module_name in module_names for name in sorted(by_module[module_name])]

    if parameter_name_order != expected_slot_order(parameter_name_order, "parameter"):
        raise QualificationStateError("parameter slots are not in canonical module/local-name order")
    if buffer_name_order != expected_slot_order(buffer_name_order, "buffer"):
        raise QualificationStateError("buffer slots are not in canonical module/local-name order")
    _validate_slot_aliases(object_groups, storage_groups, typed_names)
    external_value_objects: dict[str, TensorRecord] = {}
    external_value_storages: dict[str, _StorageDescriptor] = {}
    external_slots: tuple[ParameterSlot | BufferSlot, ...] = (*state.parameters, *state.buffers)
    for slot in external_slots:
        if slot.present:
            assert slot.object_alias is not None and slot.storage_alias is not None and slot.tensor is not None
            previous = external_value_objects.setdefault(slot.object_alias, slot.tensor)
            if previous != slot.tensor:
                raise QualificationStateError(f"inconsistent external tensor alias {slot.object_alias!r}")
            metadata = slot.tensor.metadata
            descriptor = _StorageDescriptor(
                kind="torch",
                device_type=metadata.device_type,
                device_index=metadata.device_index,
                nbytes=metadata.storage_nbytes,
            )
            previous_storage = external_value_storages.setdefault(slot.storage_alias, descriptor)
            if previous_storage != descriptor:
                raise QualificationStateError(f"inconsistent external storage alias {slot.storage_alias!r}")
    value_aliases = _ValueAliasValidation(
        external_objects=external_value_objects,
        external_storages=external_value_storages,
        objects={},
        storages={},
    )
    for extra_state in state.module_extra_states:
        _validate_value(extra_state.state, f"module_extra_states.{extra_state.module_alias}", value_aliases)

    expected_gradient_aliases = tuple(sorted(parameter_aliases))
    for name, snapshot in (
        ("raw_gradients", state.raw_gradients),
        ("post_clip_gradients", state.post_clip_gradients),
    ):
        if type(snapshot) is not GradientSnapshot or type(snapshot.gradients) is not tuple:
            raise QualificationStateError(f"invalid gradient snapshot at {name}")
        actual_aliases: list[str] = []
        for index, record in enumerate(snapshot.gradients):
            path = f"{name}[{index}]"
            if (
                type(record) is not GradientRecord
                or type(record.parameter_alias) is not str
                or type(record.frozen) is not bool
                or (record.tensor is not None and type(record.tensor) is not TensorRecord)
            ):
                raise QualificationStateError(f"invalid gradient record at {path}")
            actual_aliases.append(record.parameter_alias)
            if record.parameter_alias not in frozen_by_alias:
                raise QualificationStateError(f"foreign parameter alias at {path}")
            if record.frozen != frozen_by_alias[record.parameter_alias]:
                raise QualificationStateError(f"inconsistent frozen gradient status at {path}")
            if record.tensor is not None:
                _validate_tensor_record(record.tensor, f"{path}.tensor")
        if tuple(actual_aliases) != expected_gradient_aliases:
            raise QualificationStateError(f"{name} is not the complete canonical parameter alias inventory")

    if type(state.optimizer) is not OptimizerRecord:
        raise QualificationStateError("invalid optimizer record")
    _validate_class_identity(state.optimizer.class_identity, "optimizer.class_identity")
    if type(state.optimizer.groups) is not tuple or type(state.optimizer.state) is not tuple:
        raise QualificationStateError("optimizer inventories must be tuples")
    if type(state.optimizer.defaults) is not MappingValue:
        raise QualificationStateError("optimizer defaults must be a canonical mapping")
    memberships: list[str] = []
    for index, group in enumerate(state.optimizer.groups):
        path = f"optimizer.groups[{index}]"
        if type(group) is not OptimizerGroup or type(group.parameter_aliases) is not tuple:
            raise QualificationStateError(f"invalid optimizer group at {path}")
        if any(type(alias) is not str for alias in group.parameter_aliases):
            raise QualificationStateError(f"invalid optimizer membership at {path}")
        if type(group.options) is not MappingValue:
            raise QualificationStateError(f"invalid optimizer options at {path}")
        memberships.extend(group.parameter_aliases)
        _validate_value(group.options, f"{path}.options", value_aliases)
    if len(memberships) != len(set(memberships)):
        raise QualificationStateError("optimizer groups contain ambiguous duplicate membership")
    if not set(memberships).issubset(parameter_aliases):
        raise QualificationStateError("optimizer groups contain a foreign parameter alias")
    state_aliases: list[str] = []
    for index, entry in enumerate(state.optimizer.state):
        path = f"optimizer.state[{index}]"
        if type(entry) is not OptimizerStateEntry or type(entry.parameter_alias) is not str:
            raise QualificationStateError(f"invalid optimizer state entry at {path}")
        state_aliases.append(entry.parameter_alias)
        _validate_value(entry.state, f"optimizer.state.{entry.parameter_alias}", value_aliases)
    if tuple(state_aliases) != tuple(sorted(state_aliases)) or len(state_aliases) != len(set(state_aliases)):
        raise QualificationStateError("optimizer state aliases are not canonical and unique")
    if not set(state_aliases).issubset(parameter_aliases):
        raise QualificationStateError("optimizer state contains a foreign parameter alias")
    _validate_value(state.optimizer.defaults, "optimizer.defaults", value_aliases)

    if type(state.rng) is not RngState or type(state.rng.cuda) is not CudaRngState:
        raise QualificationStateError("invalid RNG record")
    _validate_value(state.rng.python, "rng.python", value_aliases)
    if state.rng.numpy is not None:
        _validate_value(state.rng.numpy, "rng.numpy", value_aliases)
    _validate_tensor_record(state.rng.torch_cpu, "rng.torch_cpu")
    if state.rng.cuda.disposition not in {"captured", "unavailable", "omitted"}:
        raise QualificationStateError("invalid CUDA RNG disposition")
    if type(state.rng.cuda.device_states) is not tuple:
        raise QualificationStateError("CUDA RNG device states must be a tuple")
    if state.rng.cuda.disposition == "captured" and not state.rng.cuda.device_states:
        raise QualificationStateError("captured CUDA RNG has no device states")
    if state.rng.cuda.disposition != "captured" and state.rng.cuda.device_states:
        raise QualificationStateError("uncaptured CUDA RNG disposition has device states")
    for index, tensor in enumerate(state.rng.cuda.device_states):
        _validate_tensor_record(tensor, f"rng.cuda.device_states[{index}]")

    if type(state.runtime) is not RuntimeRecord:
        raise QualificationStateError("invalid runtime record")
    for path, value in (
        ("cursor_before", state.cursor_before),
        ("cursor_after", state.cursor_after),
        ("lr_schedule_values", state.lr_schedule_values),
        ("weight_decay_schedule_values", state.weight_decay_schedule_values),
        ("reservation_manifest", state.reservation_manifest),
        ("runtime.semantic", state.runtime.semantic),
        ("runtime.authority_local", state.runtime.authority_local),
    ):
        _validate_value(value, path, value_aliases)


def _validate_numpy_payload(dtype_name: str, shape: tuple[int, ...], content: bytes, path: str) -> None:
    if type(dtype_name) is not str or not dtype_name or type(shape) is not tuple:
        raise QualificationStateError(f"invalid NumPy metadata at {path}")
    if any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise QualificationStateError(f"invalid NumPy shape at {path}.shape")
    if type(content) is not bytes:
        raise QualificationStateError(f"invalid NumPy content at {path}.content")
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - exercised without NumPy
        return
    try:
        dtype = np.dtype(dtype_name)
    except TypeError as error:
        raise QualificationStateError(f"invalid NumPy dtype at {path}.dtype") from error
    if dtype.hasobject or dtype.fields is not None or dtype.str != dtype_name:
        raise QualificationStateError(f"unsupported or noncanonical NumPy dtype at {path}.dtype")
    if len(content) != math.prod(shape) * dtype.itemsize:
        raise QualificationStateError(f"NumPy byte length mismatch at {path}.content")


@dataclass(frozen=True)
class _StorageDescriptor:
    kind: Literal["torch", "numpy"]
    device_type: str
    device_index: int | None
    nbytes: int


@dataclass
class _ValueAliasValidation:
    external_objects: Mapping[str, TensorRecord]
    external_storages: Mapping[str, _StorageDescriptor]
    objects: dict[str, tuple[str, CanonicalValue]]
    storages: dict[str, tuple[str, _StorageDescriptor]]
    next_object_alias: int = 0
    next_storage_alias: int = 0

    def register_object(self, value: CanonicalValue, path: str, alias: str) -> None:
        if type(alias) is not str or not alias:
            raise QualificationStateError(f"invalid object alias at {path}.object_alias")
        external = self.external_objects.get(alias)
        if external is not None:
            if not isinstance(value, TensorValue) or value.tensor != external:
                raise QualificationStateError(f"value object alias {alias!r} conflicts with model tensor state")
            return
        previous = self.objects.get(alias)
        if previous is None:
            expected = f"value-object:{self.next_object_alias:08d}"
            if alias != expected:
                raise QualificationStateError(f"noncanonical first object alias at {path}.object_alias")
            self.next_object_alias += 1
            self.objects[alias] = (path, value)
        elif previous[1] != value:
            raise QualificationStateError(
                f"inconsistent repeated object alias {alias!r} at {previous[0]!r} and {path!r}"
            )

    def register_storage(self, path: str, alias: str, descriptor: _StorageDescriptor) -> None:
        if type(alias) is not str or not alias:
            raise QualificationStateError(f"invalid storage alias at {path}.storage_alias")
        external = self.external_storages.get(alias)
        if external is not None:
            if descriptor != external:
                raise QualificationStateError(f"storage alias {alias!r} has incompatible metadata")
            return
        previous = self.storages.get(alias)
        if previous is None:
            expected = f"value-storage:{self.next_storage_alias:08d}"
            if alias != expected:
                raise QualificationStateError(f"noncanonical first storage alias at {path}.storage_alias")
            self.next_storage_alias += 1
            self.storages[alias] = (path, descriptor)
        elif previous[1] != descriptor:
            raise QualificationStateError(
                f"inconsistent repeated storage alias {alias!r} at {previous[0]!r} and {path!r}"
            )


def _validate_structured_entries(
    entries: tuple[MappingEntry, ...],
    path: str,
    aliases: _ValueAliasValidation,
    *,
    sorted_keys: bool,
) -> None:
    if type(entries) is not tuple:
        raise QualificationStateError(f"structured fields must be a tuple at {path}")
    keys: list[str] = []
    for index, entry in enumerate(entries):
        if type(entry) is not MappingEntry or type(entry.key) is not str or not entry.key:
            raise QualificationStateError(f"invalid structured field at {path}[{index}]")
        keys.append(entry.key)
        _validate_value(entry.value, f"{path}.{entry.key}", aliases)
    if len(keys) != len(set(keys)) or (sorted_keys and keys != sorted(keys)):
        raise QualificationStateError(f"noncanonical structured fields at {path}")


def _validate_value(
    value: CanonicalValue,
    path: str,
    aliases: _ValueAliasValidation | None = None,
) -> None:
    if aliases is None:
        aliases = _ValueAliasValidation({}, {}, {}, {})
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
        EnumValue: "enum",
        NamedTupleValue: "namedtuple",
        DataclassValue: "dataclass",
        TensorValue: "tensor",
        NumpyScalarValue: "numpy_scalar",
        NumpyArrayValue: "numpy_array",
    }
    expected_tag = expected_tags.get(type(value))
    if expected_tag is None or type(getattr(value, "tag", None)) is not str or value.tag != expected_tag:
        raise QualificationStateError(f"invalid tagged value at {path}")
    if isinstance(value, NullValue):
        return
    if isinstance(value, BoolValue):
        if type(value.value) is not bool:
            raise QualificationStateError(f"invalid bool at {path}")
    elif isinstance(value, IntValue):
        if type(value.value) is not int:
            raise QualificationStateError(f"invalid int at {path}")
    elif isinstance(value, FloatValue):
        if type(value.hexadecimal) is not str or len(value.hexadecimal) != 16:
            raise QualificationStateError(f"invalid float bits at {path}")
        try:
            raw = bytes.fromhex(value.hexadecimal)
            struct.unpack(">d", raw)
        except (ValueError, struct.error) as error:
            raise QualificationStateError(f"invalid float bits at {path}") from error
        if raw.hex() != value.hexadecimal:
            raise QualificationStateError(f"noncanonical float bits at {path}")
    elif isinstance(value, StringValue):
        if type(value.value) is not str:
            raise QualificationStateError(f"invalid string at {path}")
    elif isinstance(value, BytesValue):
        if type(value.value) is not bytes:
            raise QualificationStateError(f"invalid bytes at {path}")
    elif isinstance(value, TorchDTypeValue):
        dtype_name = value.name.removeprefix("torch.") if type(value.name) is str else ""
        dtype = getattr(torch, dtype_name, None)
        if not isinstance(dtype, torch.dtype) or str(dtype) != value.name:
            raise QualificationStateError(f"invalid torch dtype at {path}")
    elif isinstance(value, TorchDeviceValue):
        if type(value.device_type) is not str or not value.device_type:
            raise QualificationStateError(f"invalid torch device at {path}")
        if value.device_index is not None and (type(value.device_index) is not int or value.device_index < 0):
            raise QualificationStateError(f"invalid torch device index at {path}")
        try:
            (
                torch.device(value.device_type)
                if value.device_index is None
                else torch.device(value.device_type, value.device_index)
            )
        except (RuntimeError, ValueError) as error:
            raise QualificationStateError(f"invalid torch device at {path}") from error
    elif isinstance(value, TensorValue):
        _validate_tensor_record(value.tensor, f"{path}.tensor")
        aliases.register_object(value, path, value.object_alias)
        metadata = value.tensor.metadata
        aliases.register_storage(
            path,
            value.storage_alias,
            _StorageDescriptor(
                kind="torch",
                device_type=metadata.device_type,
                device_index=metadata.device_index,
                nbytes=metadata.storage_nbytes,
            ),
        )
    elif isinstance(value, NumpyScalarValue):
        _validate_numpy_payload(value.dtype, (), value.content, path)
        if type(value.content_sha256) is not str or sha256(value.content).hexdigest() != value.content_sha256:
            raise QualificationStateError(f"content digest mismatch at {path}.content_sha256")
    elif isinstance(value, NumpyArrayValue):
        _validate_numpy_payload(value.dtype, value.shape, value.content, path)
        if type(value.content_sha256) is not str or sha256(value.content).hexdigest() != value.content_sha256:
            raise QualificationStateError(f"content digest mismatch at {path}.content_sha256")
        if (
            type(value.strides) is not tuple
            or len(value.strides) != len(value.shape)
            or any(type(stride) is not int for stride in value.strides)
            or type(value.storage_offset) is not int
            or value.storage_offset < 0
            or type(value.storage_nbytes) is not int
            or value.storage_nbytes < 0
            or type(value.writeable) is not bool
        ):
            raise QualificationStateError(f"invalid NumPy view metadata at {path}")
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - exercised without NumPy
            pass
        else:
            _numpy_view_bounds(
                value.shape,
                value.strides,
                value.storage_offset,
                int(np.dtype(value.dtype).itemsize),
                value.storage_nbytes,
                path,
            )
        aliases.register_object(value, path, value.object_alias)
        aliases.register_storage(
            path,
            value.storage_alias,
            _StorageDescriptor("numpy", "cpu", None, value.storage_nbytes),
        )
    elif isinstance(value, SequenceValue):
        if value.sequence_type not in ("list", "tuple") or type(value.items) is not tuple:
            raise QualificationStateError(f"invalid sequence at {path}")
        aliases.register_object(value, path, value.object_alias)
        for index, item in enumerate(value.items):
            _validate_value(item, f"{path}[{index}]", aliases)
    elif isinstance(value, MappingValue):
        aliases.register_object(value, path, value.object_alias)
        _validate_structured_entries(value.entries, path, aliases, sorted_keys=True)
    elif isinstance(value, EnumValue):
        _validate_class_identity(value.class_identity, f"{path}.class_identity")
        if type(value.member_name) is not str or not value.member_name:
            raise QualificationStateError(f"invalid enum member at {path}")
        aliases.register_object(value, path, value.object_alias)
        _validate_value(value.value, f"{path}.value", aliases)
    elif isinstance(value, NamedTupleValue):
        _validate_class_identity(value.class_identity, f"{path}.class_identity")
        aliases.register_object(value, path, value.object_alias)
        _validate_structured_entries(value.fields, path, aliases, sorted_keys=True)
    elif isinstance(value, DataclassValue):
        _validate_class_identity(value.class_identity, f"{path}.class_identity")
        aliases.register_object(value, path, value.object_alias)
        _validate_structured_entries(value.fields, path, aliases, sorted_keys=True)


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
    EnumValue,
    NamedTupleValue,
    DataclassValue,
    TensorValue,
    NumpyScalarValue,
    NumpyArrayValue,
    ModuleRecord,
    ModuleExtraStateRecord,
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
        if set(value) != {"$blob", "length", "sha256"}:
            raise QualificationStateError("malformed blob reference in qualification manifest")
        index = value["$blob"]
        length = value["length"]
        digest = value["sha256"]
        if type(index) is not int or not 0 <= index < len(blobs):
            raise QualificationStateError("invalid blob index in qualification manifest")
        blob = blobs[index]
        if type(length) is not int or type(digest) is not str:
            raise QualificationStateError("malformed blob integrity fields")
        if length != len(blob) or digest != sha256(blob).hexdigest():
            raise QualificationStateError("qualification blob integrity failure")
        return blob
    if "$tuple" in value:
        if set(value) != {"$tuple"}:
            raise QualificationStateError("malformed tuple in qualification manifest")
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

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise QualificationStateError(f"duplicate qualification manifest key: {key!r}")
            output[key] = value
        return output

    try:
        manifest = json.loads(manifest_bytes, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationStateError("invalid qualification manifest JSON") from error
    canonical_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if canonical_manifest != manifest_bytes:
        raise QualificationStateError("qualification archive is not in canonical serialized form")

    blob_references: list[int] = []
    pending: list[object] = [manifest]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if "$blob" in item and type(item["$blob"]) is int:
                blob_references.append(item["$blob"])
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    if len(blob_references) != len(blobs) or sorted(blob_references) != list(range(len(blobs))):
        raise QualificationStateError("qualification blob references are not canonical and complete")

    decoded = _manifest_decode(manifest, blobs)
    if type(decoded) is not TrainingState:
        raise QualificationStateError("qualification archive root is not TrainingState")
    validate_training_state(decoded)
    if serialize_training_state(decoded) != payload:
        raise QualificationStateError("qualification archive is not in canonical serialized form")
    return decoded
