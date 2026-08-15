from __future__ import annotations

from dataclasses import replace
from enum import IntEnum, StrEnum
import json
import random
import struct
from typing import Any, NamedTuple

import numpy as np
import pytest
import torch
from torch import nn

from gcw.data.training import FineWebCursor, RandomCursor, SFTCursor, TrainingBatch
from gcw.performance.adjacent_pairs import (
    AdaptivePairStepPlan,
    PairGroupPlan,
    ReservedAdjacentPairs,
)
from gcw.performance.config import PhysicalBatchBandConfig
from gcw.qualification.state import (
    DataclassValue,
    EnumValue,
    FloatValue,
    IntValue,
    MappingEntry,
    MappingValue,
    NamedTupleValue,
    NumpyArrayValue,
    NumpyScalarValue,
    QualificationStateError,
    StringValue,
    SequenceValue,
    TensorValue,
    capture_gradient_snapshot,
    capture_training_state,
    compare_authority_local,
    compare_semantic_state,
    deserialize_training_state,
    encode_value,
    serialize_training_state,
    validate_training_state,
)


class RuntimeKind(IntEnum):
    REFERENCE = 1


class RuntimeLabel(StrEnum):
    EAGER = "eager"


class CursorTuple(NamedTuple):
    epoch: int
    row: int


class MixedChild(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("child_buffer", torch.tensor([9.0]))


class QualificationToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = nn.Parameter(torch.arange(6.0).reshape(2, 3))
        self.weight = base
        self.tied = base
        self.column = nn.Parameter(base.detach()[:, :1])
        self.frozen = nn.Parameter(torch.tensor([7.0]), requires_grad=False)
        self.register_parameter("absent_parameter", None)

        persistent = torch.tensor([1.0, 2.0, 3.0])
        self.register_buffer("persistent", persistent, persistent=True)
        self.register_buffer("persistent_alias", persistent, persistent=True)
        self.register_buffer("nonpersistent", persistent[1:], persistent=False)
        self.register_buffer("absent_buffer", None, persistent=False)
        self.child = MixedChild()
        self.train()
        self.child.eval()
        self.extra_generation = 5

    def forward(self) -> torch.Tensor:
        return self.weight.square().sum() + self.column.square().sum()

    def get_extra_state(self) -> object:
        return {"generation": self.extra_generation, "cursor": SFTCursor(epoch=2, row=10)}

    def set_extra_state(self, state: object) -> None:
        if not isinstance(state, dict) or type(state.get("generation")) is not int:
            raise ValueError("invalid toy extra state")
        self.extra_generation = state["generation"]


def _fixture() -> tuple[QualificationToy, torch.optim.SGD, Any, Any]:
    model = QualificationToy()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight, model.column], "lr": 0.25, "weight_decay": 0.1},
            {"params": [model.frozen], "lr": 0.05, "weight_decay": 0.0},
        ],
        momentum=0.9,
    )
    model.forward().backward()
    model.frozen.grad = torch.tensor([11.0])
    raw = capture_gradient_snapshot(model)
    assert model.weight.grad is not None
    assert model.column.grad is not None
    model.weight.grad.mul_(0.25)
    model.column.grad.add_(3.0)
    model.frozen.grad.add_(2.0)
    post = capture_gradient_snapshot(model)
    optimizer.state[model.weight] = {
        "nested": {"momentum": torch.tensor([1.0, 2.0]), "history": [1, {"accepted": True}]},
        "step": 4,
    }
    return model, optimizer, raw, post


def _capture(
    model: QualificationToy,
    optimizer: torch.optim.Optimizer,
    raw: Any,
    post: Any,
    **overrides: object,
) -> Any:
    values: dict[str, object] = {
        "cursor_before": {"epoch": 2, "sample": 10},
        "cursor_after": {"epoch": 2, "sample": 14},
        "step_ordinal": 7,
        "lr_schedule_values": [0.25, 0.05],
        "weight_decay_schedule_values": [0.1, 0.0],
        "reservation_manifest": {"reservation": "r-12", "indices": [10, 11, 12, 13]},
        "runtime_semantic": {"backend": "eager", "deterministic": True},
        "runtime_authority_local": {"pid": 123, "hostname": "worker-a"},
        "capture_cuda_rng": False,
    }
    values.update(overrides)
    return capture_training_state(
        model,
        optimizer,
        raw_gradients=raw,
        post_clip_gradients=post,
        **values,
    )


def test_complete_capture_aliases_gradients_optimizer_rng_and_context() -> None:
    model, optimizer, raw, post = _fixture()
    state = _capture(model, optimizer, raw, post)

    assert [(module.name, module.training) for module in state.modules] == [("", True), ("child", False)]
    assert state.modules[0].has_extra_state
    assert not state.modules[1].has_extra_state
    assert [record.module_alias for record in state.module_extra_states] == ["<root>"]
    assert isinstance(state.module_extra_states[0].state, MappingValue)
    parameters = {slot.name: slot for slot in state.parameters}
    assert parameters["absent_parameter"].present is False
    assert parameters["frozen"].trainable is False
    assert parameters["weight"].object_alias == parameters["tied"].object_alias
    assert parameters["weight"].storage_alias == parameters["column"].storage_alias
    assert parameters["weight"].tensor is not None
    assert parameters["weight"].tensor.metadata.shape == (2, 3)
    assert parameters["column"].tensor is not None
    assert parameters["column"].tensor.metadata.stride == (3, 1)
    assert parameters["column"].tensor.metadata.storage_offset == 0

    buffers = {slot.name: slot for slot in state.buffers}
    assert buffers["absent_buffer"].present is False
    assert buffers["absent_buffer"].persistent is False
    assert buffers["persistent"].persistent is True
    assert buffers["nonpersistent"].persistent is False
    assert buffers["persistent"].object_alias == buffers["persistent_alias"].object_alias
    assert buffers["persistent"].storage_alias == buffers["nonpersistent"].storage_alias

    raw_by_name = {gradient.parameter_alias: gradient for gradient in state.raw_gradients.gradients}
    post_by_name = {gradient.parameter_alias: gradient for gradient in state.post_clip_gradients.gradients}
    assert raw_by_name["frozen"].frozen and raw_by_name["frozen"].tensor is not None
    assert raw_by_name["frozen"].tensor != post_by_name["frozen"].tensor
    assert raw_by_name["tied"].tensor != post_by_name["tied"].tensor
    assert raw_by_name["column"].tensor != post_by_name["column"].tensor

    assert state.optimizer.class_identity.qualname == "SGD"
    assert state.optimizer.groups[0].parameter_aliases == ("tied", "column")
    assert state.optimizer.groups[1].parameter_aliases == ("frozen",)
    assert state.optimizer.state[0].parameter_alias == "tied"
    assert isinstance(state.optimizer.state[0].state, MappingValue)
    assert state.rng.cuda.disposition == "omitted"
    assert state.rng.torch_cpu.content
    assert state.rng.numpy is not None
    assert state.step_ordinal == 7


def test_capture_is_deterministic_round_trips_and_does_not_mutate_live_state() -> None:
    model, optimizer, raw, post = _fixture()
    model_modes = tuple(module.training for module in model.modules())
    parameter_versions = tuple(parameter._version for parameter in model.parameters())
    parameter_bytes = tuple(parameter.detach().clone() for parameter in model.parameters())
    gradient_bytes = tuple(
        None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()
    )
    optimizer_groups = tuple(tuple(group["params"]) for group in optimizer.param_groups)
    python_rng = random.getstate()
    numpy_rng = np.random.get_state()
    torch_rng = torch.random.get_rng_state().clone()

    first = _capture(model, optimizer, raw, post)
    second = _capture(model, optimizer, raw, post)
    first_archive = serialize_training_state(first)

    assert first == second
    assert first_archive == serialize_training_state(second)
    assert deserialize_training_state(first_archive) == first
    assert first.semantic_digest() == second.semantic_digest()
    assert tuple(module.training for module in model.modules()) == model_modes
    assert tuple(parameter._version for parameter in model.parameters()) == parameter_versions
    assert all(torch.equal(before, after) for before, after in zip(parameter_bytes, model.parameters()))
    for before, parameter in zip(gradient_bytes, model.parameters()):
        assert (before is None and parameter.grad is None) or (
            before is not None and parameter.grad is not None and torch.equal(before, parameter.grad)
        )
    assert tuple(tuple(group["params"]) for group in optimizer.param_groups) == optimizer_groups
    assert random.getstate() == python_rng
    current_numpy_rng = np.random.get_state()
    assert current_numpy_rng[0] == numpy_rng[0]
    assert np.array_equal(current_numpy_rng[1], numpy_rng[1])
    assert current_numpy_rng[2:] == numpy_rng[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_rng)


def test_deserialization_revalidates_scalars_and_requires_canonical_archive_bytes() -> None:
    model, optimizer, raw, post = _fixture()
    archive = serialize_training_state(_capture(model, optimizer, raw, post))
    magic = b"GCWQSTATE\x00\x01"
    manifest_length = struct.unpack(">Q", archive[len(magic) : len(magic) + 8])[0]
    manifest_start = len(magic) + 8
    manifest_stop = manifest_start + manifest_length
    manifest = json.loads(archive[manifest_start:manifest_stop])

    malformed = dict(manifest)
    malformed["step_ordinal"] = True
    malformed_bytes = json.dumps(malformed, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    malformed_archive = magic + struct.pack(">Q", len(malformed_bytes)) + malformed_bytes + archive[manifest_stop:]
    with pytest.raises(QualificationStateError, match="step_ordinal"):
        deserialize_training_state(malformed_archive)

    noncanonical_bytes = json.dumps(manifest, sort_keys=True, indent=1, ensure_ascii=False).encode()
    noncanonical_archive = (
        magic + struct.pack(">Q", len(noncanonical_bytes)) + noncanonical_bytes + archive[manifest_stop:]
    )
    with pytest.raises(QualificationStateError, match="canonical serialized form"):
        deserialize_training_state(noncanonical_archive)


def _assert_path(expected: Any, actual: Any, path: str, kind: str | None = None) -> None:
    result = compare_semantic_state(expected, actual)
    assert not result.equal
    matches = [mismatch for mismatch in result.mismatches if mismatch.path == path]
    assert matches, result.mismatches
    if kind is not None:
        assert matches[0].kind == kind


def test_comparison_reports_precise_paths_for_important_field_classes() -> None:
    model, optimizer, raw, post = _fixture()
    state = _capture(model, optimizer, raw, post)

    modules = list(state.modules)
    modules[1] = replace(modules[1], training=not modules[1].training)
    _assert_path(state, replace(state, modules=tuple(modules)), "state.modules[1].training", "metadata")

    parameters = list(state.parameters)
    parameter = parameters[-1]
    assert parameter.tensor is not None
    changed_content = parameter.tensor.content + b"x"
    parameters[-1] = replace(
        parameter,
        tensor=replace(
            parameter.tensor,
            content=changed_content,
            content_sha256=__import__("hashlib").sha256(changed_content).hexdigest(),
        ),
    )
    _assert_path(
        state,
        replace(state, parameters=tuple(parameters)),
        f"state.parameters[{len(parameters) - 1}].tensor.content",
        "bytes",
    )

    parameters = list(state.parameters)
    parameters[-1] = replace(parameters[-1], storage_alias="different")
    _assert_path(
        state,
        replace(state, parameters=tuple(parameters)),
        f"state.parameters[{len(parameters) - 1}].storage_alias",
        "alias",
    )

    buffers = list(state.buffers)
    buffers[0] = replace(buffers[0], persistent=not buffers[0].persistent)
    _assert_path(state, replace(state, buffers=tuple(buffers)), "state.buffers[0].persistent", "metadata")

    raw_records = list(state.raw_gradients.gradients)
    raw_records[0] = replace(raw_records[0], tensor=None)
    changed_raw = replace(state.raw_gradients, gradients=tuple(raw_records))
    _assert_path(
        state, replace(state, raw_gradients=changed_raw), "state.raw_gradients.gradients[0].tensor", "structural"
    )

    post_records = list(state.post_clip_gradients.gradients)
    post_records[0] = replace(post_records[0], parameter_alias="other")
    changed_post = replace(state.post_clip_gradients, gradients=tuple(post_records))
    _assert_path(
        state,
        replace(state, post_clip_gradients=changed_post),
        "state.post_clip_gradients.gradients[0].parameter_alias",
        "alias",
    )

    groups = list(state.optimizer.groups)
    groups[0] = replace(groups[0], parameter_aliases=tuple(reversed(groups[0].parameter_aliases)))
    changed_optimizer = replace(state.optimizer, groups=tuple(groups))
    _assert_path(
        state, replace(state, optimizer=changed_optimizer), "state.optimizer.groups[0].parameter_aliases[0]", "alias"
    )

    _assert_path(state, replace(state, step_ordinal=8), "state.step_ordinal", "value")
    _assert_path(state, replace(state, cursor_after=StringValue("different")), "state.cursor_after", "structural")
    _assert_path(state, replace(state, cursor_before=StringValue("different")), "state.cursor_before", "structural")
    _assert_path(
        state,
        replace(state, lr_schedule_values=StringValue("different")),
        "state.lr_schedule_values",
        "structural",
    )
    _assert_path(
        state,
        replace(state, weight_decay_schedule_values=StringValue("different")),
        "state.weight_decay_schedule_values",
        "structural",
    )
    _assert_path(
        state,
        replace(state, reservation_manifest=StringValue("different")),
        "state.reservation_manifest",
        "structural",
    )

    parameter_index = next(index for index, slot in enumerate(state.parameters) if slot.tensor is not None)
    parameter_with_tensor = state.parameters[parameter_index]
    assert parameter_with_tensor.tensor is not None
    changed_metadata = replace(parameter_with_tensor.tensor.metadata, dtype="torch.float64")
    changed_parameter = replace(
        parameter_with_tensor, tensor=replace(parameter_with_tensor.tensor, metadata=changed_metadata)
    )
    metadata_parameters = list(state.parameters)
    metadata_parameters[parameter_index] = changed_parameter
    _assert_path(
        state,
        replace(state, parameters=tuple(metadata_parameters)),
        f"state.parameters[{parameter_index}].tensor.metadata.dtype",
        "metadata",
    )

    changed_class = replace(state.optimizer.class_identity, qualname="OtherOptimizer")
    _assert_path(
        state,
        replace(state, optimizer=replace(state.optimizer, class_identity=changed_class)),
        "state.optimizer.class_identity.qualname",
        "metadata",
    )
    default_entries = list(state.optimizer.defaults.entries)
    default_entries[0] = replace(default_entries[0], value=StringValue("different"))
    _assert_path(
        state,
        replace(state, optimizer=replace(state.optimizer, defaults=MappingValue(tuple(default_entries)))),
        "state.optimizer.defaults.entries[0].value",
        "structural",
    )
    option_entries = list(state.optimizer.groups[0].options.entries)
    option_entries[0] = replace(option_entries[0], value=StringValue("different"))
    option_groups = list(state.optimizer.groups)
    option_groups[0] = replace(option_groups[0], options=MappingValue(tuple(option_entries)))
    _assert_path(
        state,
        replace(state, optimizer=replace(state.optimizer, groups=tuple(option_groups))),
        "state.optimizer.groups[0].options.entries[0].value",
        "structural",
    )
    optimizer_entries = list(state.optimizer.state)
    optimizer_entries[0] = replace(optimizer_entries[0], state=StringValue("different"))
    _assert_path(
        state,
        replace(state, optimizer=replace(state.optimizer, state=tuple(optimizer_entries))),
        "state.optimizer.state[0].state",
        "structural",
    )

    cpu_rng_content = bytearray(state.rng.torch_cpu.content)
    cpu_rng_content[0] ^= 1
    changed_cpu_rng = replace(state.rng.torch_cpu, content=bytes(cpu_rng_content))
    _assert_path(
        state,
        replace(state, rng=replace(state.rng, torch_cpu=changed_cpu_rng)),
        "state.rng.torch_cpu.content",
        "bytes",
    )
    _assert_path(
        state,
        replace(state, rng=replace(state.rng, python=StringValue("different"))),
        "state.rng.python",
        "structural",
    )
    _assert_path(
        state,
        replace(state, rng=replace(state.rng, cuda=replace(state.rng.cuda, disposition="unavailable"))),
        "state.rng.cuda.disposition",
        "value",
    )
    _assert_path(
        state,
        replace(state, runtime=replace(state.runtime, semantic=StringValue("different"))),
        "state.runtime.semantic",
        "structural",
    )

    authority_changed = replace(
        state,
        runtime=replace(state.runtime, authority_local=MappingValue((MappingEntry("pid", StringValue("new")),))),
    )
    assert compare_semantic_state(state, authority_changed).equal
    authority_result = compare_authority_local(state, authority_changed)
    assert not authority_result.equal
    assert authority_result.mismatches[0].path.startswith("authority_local")
    assert state.semantic_digest() == authority_changed.semantic_digest()


def test_fail_closed_for_values_cycles_foreign_params_duplicates_and_bad_snapshots() -> None:
    with pytest.raises(QualificationStateError, match="non-string mapping key"):
        encode_value({1: "ambiguous"})
    with pytest.raises(QualificationStateError, match="unsupported non-deterministic"):
        encode_value({"set"})
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(QualificationStateError, match="cycle"):
        encode_value(cyclic)

    model, optimizer, raw, post = _fixture()
    foreign = nn.Parameter(torch.tensor([1.0]))
    foreign_optimizer = torch.optim.SGD([model.weight, foreign], lr=0.1)
    with pytest.raises(QualificationStateError, match="not owned by the model"):
        _capture(model, foreign_optimizer, raw, post)

    unsupported_optimizer = torch.optim.SGD([model.weight], lr=0.1)
    unsupported_optimizer.param_groups[0]["unsupported"] = {"unordered"}
    with pytest.raises(QualificationStateError, match="unsupported non-deterministic"):
        _capture(model, unsupported_optimizer, raw, post)

    optimizer.param_groups[1]["params"].append(model.weight)
    with pytest.raises(QualificationStateError, match="ambiguous duplicate"):
        _capture(model, optimizer, raw, post)

    with pytest.raises(QualificationStateError, match="does not match canonical"):
        _capture(model, torch.optim.SGD([model.weight], lr=0.1), replace(raw, gradients=raw.gradients[:-1]), post)


def test_production_dataclasses_namedtuples_and_enums_preserve_exact_types() -> None:
    batch = TrainingBatch(
        x=torch.arange(4).reshape(2, 2),
        y=torch.arange(4).reshape(2, 2),
        attention_mask=torch.ones(2, 2, dtype=torch.bool),
    )
    reservation = ReservedAdjacentPairs(
        pairs=tuple(batch for _ in range(32)),
        padded_lengths=tuple(2 for _ in range(32)),
    )
    band = PhysicalBatchBandConfig(max_padded_tokens=4096, physical_rows=16)
    groups = tuple(PairGroupPlan(start, start + 8, 16, 2, band) for start in range(0, 32, 8))
    plan = AdaptivePairStepPlan(reservation, groups)
    for value in (
        RandomCursor((1, 2, 3)),
        FineWebCursor(epoch=2, shard_index=3, step_in_shard=4),
        SFTCursor(epoch=5, row=6),
        reservation,
        plan,
    ):
        encoded = encode_value(value)
        assert isinstance(encoded, DataclassValue)
        assert encoded.class_identity.module.startswith("gcw.")

    named = encode_value(CursorTuple(epoch=7, row=8))
    assert isinstance(named, NamedTupleValue)
    assert [field.key for field in named.fields] == ["epoch", "row"]
    integer_enum = encode_value(RuntimeKind.REFERENCE)
    string_enum = encode_value(RuntimeLabel.EAGER)
    assert isinstance(integer_enum, EnumValue)
    assert isinstance(string_enum, EnumValue)
    assert integer_enum.class_identity.qualname == "RuntimeKind"
    assert string_enum.class_identity.qualname == "RuntimeLabel"


def test_nonfinite_scalar_bits_and_numpy_scalar_dtype_are_preserved() -> None:
    negative_nan_bits = "fff8000000000001"
    negative_nan = struct.unpack(">d", bytes.fromhex(negative_nan_bits))[0]
    encoded_nan = encode_value(negative_nan)
    assert isinstance(encoded_nan, FloatValue)
    assert encoded_nan.hexadecimal == negative_nan_bits
    positive_zero = encode_value(0.0)
    negative_zero = encode_value(-0.0)
    assert isinstance(positive_zero, FloatValue)
    assert isinstance(negative_zero, FloatValue)
    assert positive_zero.hexadecimal != negative_zero.hexadecimal

    numpy_nan = np.array([0xFFC00001], dtype=np.uint32).view(np.float32)[0]
    encoded_numpy = encode_value(numpy_nan)
    assert isinstance(encoded_numpy, NumpyScalarValue)
    assert encoded_numpy.dtype == np.asarray(numpy_nan).dtype.str
    assert encoded_numpy.content == np.asarray(numpy_nan).tobytes()


def test_extra_state_frozen_gradients_and_deserialized_scalars_fail_closed() -> None:
    model, optimizer, raw, post = _fixture()
    state = _capture(model, optimizer, raw, post)
    frozen = next(record for record in state.raw_gradients.gradients if record.parameter_alias == "frozen")
    assert frozen.tensor is not None
    assert frozen.tensor.content == torch.tensor([11.0]).view(torch.uint8).numpy().tobytes()

    model.extra_generation += 1
    changed = _capture(model, optimizer, raw, post)
    _assert_path(
        state,
        changed,
        "state.module_extra_states[0].state.entries[1].value.value",
        "value",
    )

    with pytest.raises(QualificationStateError, match="step_ordinal"):
        validate_training_state(replace(state, step_ordinal=True))
    with pytest.raises(QualificationStateError, match="CUDA RNG disposition"):
        validate_training_state(
            replace(state, rng=replace(state.rng, cuda=replace(state.rng.cuda, disposition="forged")))  # type: ignore[arg-type]
        )
    with pytest.raises(QualificationStateError, match="module extra state"):
        validate_training_state(replace(state, module_extra_states=()))


def test_recursive_values_preserve_object_and_storage_alias_graphs() -> None:
    shared: list[object] = []
    shared_graph = encode_value({"left": shared, "right": shared})
    distinct_graph = encode_value({"left": [], "right": []})
    assert isinstance(shared_graph, MappingValue)
    assert isinstance(distinct_graph, MappingValue)
    shared_items = {entry.key: entry.value for entry in shared_graph.entries}
    distinct_items = {entry.key: entry.value for entry in distinct_graph.entries}
    assert isinstance(shared_items["left"], SequenceValue)
    assert isinstance(shared_items["right"], SequenceValue)
    assert shared_items["left"].object_alias == shared_items["right"].object_alias
    assert isinstance(distinct_items["left"], SequenceValue)
    assert isinstance(distinct_items["right"], SequenceValue)
    assert distinct_items["left"].object_alias != distinct_items["right"].object_alias
    assert shared_graph != distinct_graph

    tensor = torch.arange(8.0)
    tensor_view = tensor[2:6]
    tensor_graph = encode_value({"base": tensor, "same": tensor, "view": tensor_view})
    assert isinstance(tensor_graph, MappingValue)
    tensor_items = {entry.key: entry.value for entry in tensor_graph.entries}
    base_tensor = tensor_items["base"]
    same_tensor = tensor_items["same"]
    view_tensor = tensor_items["view"]
    assert isinstance(base_tensor, TensorValue)
    assert isinstance(same_tensor, TensorValue)
    assert isinstance(view_tensor, TensorValue)
    assert base_tensor.object_alias == same_tensor.object_alias
    assert base_tensor.object_alias != view_tensor.object_alias
    assert base_tensor.storage_alias == same_tensor.storage_alias == view_tensor.storage_alias
    assert view_tensor.tensor.metadata.storage_offset == 2
    assert view_tensor.tensor.metadata.storage_nbytes == tensor.untyped_storage().nbytes()

    numpy_base = np.arange(8, dtype=np.int32)
    numpy_view = numpy_base[::-2]
    numpy_graph = encode_value({"base": numpy_base, "same": numpy_base, "view": numpy_view})
    assert isinstance(numpy_graph, MappingValue)
    numpy_items = {entry.key: entry.value for entry in numpy_graph.entries}
    base_array = numpy_items["base"]
    same_array = numpy_items["same"]
    view_array = numpy_items["view"]
    assert isinstance(base_array, NumpyArrayValue)
    assert isinstance(same_array, NumpyArrayValue)
    assert isinstance(view_array, NumpyArrayValue)
    assert base_array.object_alias == same_array.object_alias
    assert base_array.object_alias != view_array.object_alias
    assert base_array.storage_alias == same_array.storage_alias == view_array.storage_alias
    assert view_array.strides == (-8,)
    assert view_array.storage_offset == 28
    assert view_array.storage_nbytes == numpy_base.nbytes

    owner = bytearray(range(32))
    first_buffer_view = np.frombuffer(owner, dtype=np.uint8)[2:14]
    second_buffer_view = np.frombuffer(owner, dtype=np.uint8)[4:20]
    buffer_graph = encode_value({"first": first_buffer_view, "second": second_buffer_view})
    assert isinstance(buffer_graph, MappingValue)
    buffer_items = {entry.key: entry.value for entry in buffer_graph.entries}
    first_array = buffer_items["first"]
    second_array = buffer_items["second"]
    assert isinstance(first_array, NumpyArrayValue)
    assert isinstance(second_array, NumpyArrayValue)
    assert first_array.storage_alias == second_array.storage_alias
    assert (first_array.storage_offset, second_array.storage_offset) == (2, 4)
    assert first_array.storage_nbytes == second_array.storage_nbytes == len(owner)


def test_aliases_span_complete_training_values_and_fail_closed_when_forged() -> None:
    model, optimizer, raw, post = _fixture()
    shared = [1, 2]
    state = _capture(
        model,
        optimizer,
        raw,
        post,
        cursor_before=shared,
        cursor_after=shared,
        runtime_semantic={"parameter": model.weight},
    )
    assert isinstance(state.cursor_before, SequenceValue)
    assert isinstance(state.cursor_after, SequenceValue)
    assert state.cursor_before.object_alias == state.cursor_after.object_alias
    assert isinstance(state.runtime.semantic, MappingValue)
    runtime_tensor = state.runtime.semantic.entries[0].value
    assert isinstance(runtime_tensor, TensorValue)
    parameter = next(slot for slot in state.parameters if slot.name == "weight")
    assert runtime_tensor.object_alias == parameter.object_alias
    assert runtime_tensor.storage_alias == parameter.storage_alias

    forged_after = replace(state.cursor_after, items=(IntValue(99),))
    with pytest.raises(QualificationStateError, match="inconsistent repeated object alias"):
        validate_training_state(replace(state, cursor_after=forged_after))

    forged_before = replace(state.cursor_before, object_alias="value-object:99999999")
    with pytest.raises(QualificationStateError, match="object alias"):
        validate_training_state(replace(state, cursor_before=forged_before))


def test_canonical_inventory_order_prefixes_and_structured_fields_are_enforced() -> None:
    root = nn.Module()
    weight = nn.Parameter(torch.tensor([1.0]))
    root.register_parameter("weight", weight)
    child_a = nn.Module()
    child_a.add_module("z", nn.Module())
    root.add_module("a", child_a)
    root.add_module("a!", nn.Module())
    optimizer = torch.optim.SGD([weight], lr=0.1)
    raw = capture_gradient_snapshot(root)
    state = capture_training_state(
        root,
        optimizer,
        raw_gradients=raw,
        post_clip_gradients=raw,
        cursor_before={},
        cursor_after={},
        step_ordinal=0,
        lr_schedule_values=(0.1,),
        weight_decay_schedule_values=(0.0,),
        reservation_manifest={},
        runtime_semantic={},
        runtime_authority_local={},
        capture_cuda_rng=False,
    )
    assert [record.name for record in state.modules] == ["", "a", "a.z", "a!"]
    assert [record.name for record in state.modules] != sorted(record.name for record in state.modules)

    reordered_modules = (state.modules[0], state.modules[1], state.modules[3], state.modules[2])
    with pytest.raises(QualificationStateError, match="depth-first"):
        validate_training_state(replace(state, modules=reordered_modules))

    ghost_modules = list(state.modules)
    ghost_modules[2] = replace(ghost_modules[2], name="ghost.child", object_alias="ghost.child")
    with pytest.raises(QualificationStateError, match="prefix-closed"):
        validate_training_state(replace(state, modules=tuple(ghost_modules)))

    model, toy_optimizer, toy_raw, toy_post = _fixture()
    toy_state = _capture(
        model,
        toy_optimizer,
        toy_raw,
        toy_post,
        cursor_before=SFTCursor(epoch=1, row=2),
    )
    with pytest.raises(QualificationStateError, match="parameter slots"):
        validate_training_state(replace(toy_state, parameters=tuple(reversed(toy_state.parameters))))
    with pytest.raises(QualificationStateError, match="buffer slots"):
        validate_training_state(replace(toy_state, buffers=tuple(reversed(toy_state.buffers))))
    assert isinstance(toy_state.cursor_before, DataclassValue)
    noncanonical_cursor = replace(toy_state.cursor_before, fields=tuple(reversed(toy_state.cursor_before.fields)))
    with pytest.raises(QualificationStateError, match="noncanonical structured fields"):
        validate_training_state(replace(toy_state, cursor_before=noncanonical_cursor))


def test_deserialization_rejects_semantically_equivalent_blob_permutations() -> None:
    model, optimizer, raw, post = _fixture()
    archive = serialize_training_state(_capture(model, optimizer, raw, post))
    magic = b"GCWQSTATE\x00\x01"
    position = len(magic)
    manifest_length = struct.unpack(">Q", archive[position : position + 8])[0]
    position += 8
    manifest = json.loads(archive[position : position + manifest_length])
    position += manifest_length
    blob_count = struct.unpack(">Q", archive[position : position + 8])[0]
    position += 8
    blobs: list[bytes] = []
    for _ in range(blob_count):
        length = struct.unpack(">Q", archive[position : position + 8])[0]
        position += 8
        blobs.append(archive[position : position + length])
        position += length
    assert len(blobs) >= 2 and position == len(archive)

    def swap_blob_references(value: object) -> None:
        if isinstance(value, dict):
            index = value.get("$blob")
            if index == 0:
                value["$blob"] = 1
            elif index == 1:
                value["$blob"] = 0
            for child in value.values():
                swap_blob_references(child)
        elif isinstance(value, list):
            for child in value:
                swap_blob_references(child)

    swap_blob_references(manifest)
    blobs[0], blobs[1] = blobs[1], blobs[0]
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    permuted = bytearray(magic)
    permuted.extend(struct.pack(">Q", len(manifest_bytes)))
    permuted.extend(manifest_bytes)
    permuted.extend(struct.pack(">Q", len(blobs)))
    for blob in blobs:
        permuted.extend(struct.pack(">Q", len(blob)))
        permuted.extend(blob)

    with pytest.raises(QualificationStateError, match="canonical serialized form"):
        deserialize_training_state(bytes(permuted))


def test_sparse_quantized_and_meta_tensors_fail_closed() -> None:
    with pytest.raises(QualificationStateError, match="non-strided"):
        encode_value(torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.0]), (1,)))
    with pytest.raises(QualificationStateError, match="quantized"):
        encode_value(torch.quantize_per_tensor(torch.tensor([1.0]), scale=0.1, zero_point=0, dtype=torch.qint8))
    with pytest.raises(QualificationStateError, match="meta"):
        encode_value(torch.empty(1, device="meta"))


@pytest.mark.gpu
def test_cuda_rng_capture_is_opt_in_to_gpu_ci() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model, optimizer, raw, post = _fixture()
    state = _capture(model, optimizer, raw, post, capture_cuda_rng=True)
    assert state.rng.cuda.disposition == "captured"
    assert len(state.rng.cuda.device_states) == torch.cuda.device_count()
