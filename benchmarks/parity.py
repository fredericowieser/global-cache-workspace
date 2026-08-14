from __future__ import annotations

import types
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.models.cache_processors.state import CacheState
from gcw.models.cache_processors.standard import _merge_kv, _split_kv


@dataclass
class ProcessorParityRun:
    cache: CacheState
    selected_indices: torch.Tensor
    input_keys: list[torch.Tensor]
    input_values: list[torch.Tensor]


def small_cpu_config(variant: str) -> ModelConfig:
    if variant not in {"bt", "bgt"}:
        raise ValueError("eager parity fixtures support BT and BGT")
    return ModelConfig(
        variant=variant,
        vocab_size=64,
        hidden_size=24,
        intermediate_size=48,
        num_layers=2,
        num_heads=3,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        tie_embeddings=True,
        cache_processor=CacheProcessorConfig(
            process_every=3,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_layers=1,
            num_heads=1,
            num_kv_heads=1,
            attention_dropout=0.0,
        ),
    )


def deterministic_cache(config: ModelConfig, *, requires_grad: bool) -> CacheState:
    generator = torch.Generator().manual_seed(8128)
    num_heads, head_dim = config.stored_cache_shape
    shape = (2, num_heads, 7, head_dim)
    keys = [torch.randn(shape, generator=generator).requires_grad_(requires_grad) for _ in range(config.cache_num_layers)]
    values = [
        torch.randn(shape, generator=generator).requires_grad_(requires_grad) for _ in range(config.cache_num_layers)
    ]
    scores = torch.tensor(
        [[0.1, 1.8, 0.2, 1.2, 0.4, 0.5, 0.6], [1.7, 0.2, 1.4, 0.3, 0.7, 0.6, 0.5]],
        dtype=torch.float32,
    )
    return CacheState(keys, values, scores)


def clone_cache(cache: CacheState, *, requires_grad: bool) -> CacheState:
    keys = [tensor.detach().clone().requires_grad_(requires_grad) for tensor in cache.keys]
    values = [tensor.detach().clone().requires_grad_(requires_grad) for tensor in cache.values]
    return CacheState(keys, values, cache.scores.detach().clone())


def selected_indices(processor: torch.nn.Module, cache: CacheState, *, select_all: bool = False) -> torch.Tensor:
    if select_all:
        return cache.all_indices()
    select = getattr(processor, "_select_indices", None)
    if callable(select):
        return select(cache, False)
    config = processor.config
    return cache.select_indices(config.process_every, config.topk)


def eager_processor_forward(
    processor: torch.nn.Module,
    cache: CacheState,
    *,
    select_all: bool = False,
) -> CacheState:
    indices = selected_indices(processor, cache, select_all=select_all)
    blocks = getattr(processor, "blocks", None)
    if blocks is not None:
        for layer_idx, block in enumerate(blocks):
            keys, values = cache.gather_layer(layer_idx, indices)
            if processor.config.update_stream == "v":
                updated = block(_merge_values(values), indices)
                cache.values[layer_idx] = _scatter_values(
                    cache.values[layer_idx], indices, _split_values(updated, values.shape[1], values.shape[3])
                )
            else:
                updated = block(_merge_kv(keys, values), indices)
                new_keys, new_values = _split_kv(updated, keys.shape[1], keys.shape[3])
                cache.scatter_layer(layer_idx, indices, new_keys, new_values)
        return cache

    parts: list[torch.Tensor] = []
    shapes: list[tuple[int, int]] = []
    for layer_idx in range(processor.num_layers):
        keys, values = cache.gather_layer(layer_idx, indices)
        shapes.append((keys.shape[1], keys.shape[3]))
        parts.append(_merge_values(values) if processor.config.update_stream == "v" else _merge_kv(keys, values))
    updated = processor.block(torch.cat(parts, dim=-1), indices)
    widths = [part.shape[-1] for part in parts]
    for layer_idx, part in enumerate(torch.split(updated, widths, dim=-1)):
        if processor.config.update_stream == "v":
            cache.values[layer_idx] = _scatter_values(
                cache.values[layer_idx], indices, _split_values(part, *shapes[layer_idx])
            )
        else:
            new_keys, new_values = _split_kv(part, *shapes[layer_idx])
            cache.scatter_layer(layer_idx, indices, new_keys, new_values)
    return cache


def run_processor_parity_path(
    processor: torch.nn.Module,
    source: CacheState,
    *,
    eager: bool,
) -> ProcessorParityRun:
    cache = clone_cache(source, requires_grad=True)
    input_keys = list(cache.keys)
    input_values = list(cache.values)
    indices = selected_indices(processor, cache).detach().clone()
    output = eager_processor_forward(processor, cache) if eager else processor(cache)
    return ProcessorParityRun(output, indices, input_keys, input_values)


def patch_eager_reference(model: torch.nn.Module) -> None:
    processor = model.cache_processor

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        return eager_processor_forward(self, cache, select_all=select_all)

    def prepare_forward_execution(self):
        return nullcontext(self)

    processor.forward = types.MethodType(forward, processor)
    processor.prepare_forward_execution = types.MethodType(prepare_forward_execution, processor)
    model.backbone.forward_windowed_loss = types.MethodType(eager_windowed_loss, model.backbone)


def eager_windowed_loss(
    self,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    processor: torch.nn.Module,
    process_every: int,
    loss_reduction: str = "mean",
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("eager parity reference only supports non-padding fixtures")
    cache = CacheState([], [], input_ids.new_empty(input_ids.shape[0], 0, dtype=torch.float32))
    loss_sum = input_ids.new_zeros((), dtype=torch.float32)
    token_count = input_ids.new_zeros((), dtype=torch.float32)
    for start in range(0, input_ids.shape[1], process_every):
        if start:
            cache = processor(cache)
        end = start + process_every
        logits, cache = self.forward_chunk(input_ids[:, start:end], start, cache, logits=True)
        chunk_targets = targets[:, start:end].reshape(-1).clone()
        valid = chunk_targets >= 0
        chunk_targets[~valid] = -100
        loss_sum = loss_sum + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), chunk_targets, ignore_index=-100, reduction="sum"
        )
        token_count = token_count + valid.sum().float()
    if loss_reduction == "sum":
        return loss_sum
    if loss_reduction == "none":
        raise ValueError("eager windowed loss does not support reduction='none'")
    return loss_sum / token_count.clamp_min(1.0)


def cache_objective(cache: CacheState) -> torch.Tensor:
    terms = [tensor.float().square().mean() for tensor in (*cache.keys, *cache.values)]
    return torch.stack(terms).sum()


def assert_cache_close(actual: CacheState, expected: CacheState, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert len(actual.keys) == len(expected.keys)
    for actual_tensor, expected_tensor in zip((*actual.keys, *actual.values), (*expected.keys, *expected.values)):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.scores, expected.scores, atol=0.0, rtol=0.0)


def assert_tensor_gradients_close(
    actual: list[torch.Tensor],
    expected: list[torch.Tensor],
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> None:
    assert len(actual) == len(expected)
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert actual_tensor.grad is not None
        assert expected_tensor.grad is not None
        torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, atol=atol, rtol=rtol)


def assert_parameter_gradients_close(
    actual: torch.nn.Module,
    expected: torch.nn.Module,
    *,
    atol: float = 2e-5,
    rtol: float = 2e-4,
) -> None:
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        actual_gradient = actual_parameters[name].grad
        expected_gradient = expected_parameters[name].grad
        assert actual_gradient is not None, name
        assert expected_gradient is not None, name
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=atol, rtol=rtol, msg=name)


def _merge_values(values: torch.Tensor) -> torch.Tensor:
    batch, heads, seq, dim = values.shape
    return values.permute(0, 2, 1, 3).reshape(batch, seq, heads * dim)


def _split_values(x: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    batch, seq, _ = x.shape
    return x.view(batch, seq, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def _scatter_values(old: torch.Tensor, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    index = indices[:, None, :, None].expand(-1, old.shape[1], -1, old.shape[3])
    return old.scatter(2, index, values.to(old.dtype))
