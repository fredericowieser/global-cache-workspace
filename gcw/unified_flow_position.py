"""Unified flow-workspace and continuous-position processing for frozen Qwen3.

This module deliberately composes the existing Longshot workspace processor and
continuous-thought decoder.  It does not own the backbone: checkpoints therefore
contain only the trainable cache processor, workspace adapter, and emitter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn

from gcw.config import CacheProcessorConfig, ModelConfig, WorkspaceFlowRefinementConfig
from gcw.continuous_thought import (
    ContinuousThoughtEmitter,
    ContinuousThoughtEmitterConfig,
    ContinuousThoughtTrace,
    append_continuous_thoughts,
)
from gcw.models.longshot import GlobalCacheProcessor, LongshotWorkspaceSource
from gcw.models.transformer import TF
from gcw.models.windowed_cache import WindowedCache


@dataclass(frozen=True)
class UnifiedFlowPositionConfig:
    """Architecture controls for the selected hybrid.

    The recommended setting is two tied workspace cycles, midpoint-2 flow with
    velocity 0.1 in the Longshot config, and two cycles of two latent positions.
    ``source_delta_bound`` bounds every adapter coordinate before it reaches the
    existing continuous-thought emitter.
    """

    model_width: int
    workspace_cycles: int = 2
    position_cycles: int = 2
    latent_tokens_per_cycle: int = 2
    emitter_width: int = 512
    source_delta_bound: float = 0.1
    select_all: bool = True

    def __post_init__(self) -> None:
        for name in ("model_width", "emitter_width", "latent_tokens_per_cycle"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("workspace_cycles", "position_cycles"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.source_delta_bound, int | float) or isinstance(self.source_delta_bound, bool):
            raise ValueError("source_delta_bound must be positive and finite")
        bound = float(self.source_delta_bound)
        if not torch.isfinite(torch.tensor(bound)) or bound <= 0:
            raise ValueError("source_delta_bound must be positive and finite")
        if not isinstance(self.select_all, bool):
            raise ValueError("select_all must be boolean")


@dataclass(frozen=True)
class UnifiedSource:
    """Detached source for exact matched/shuffled source transplantation."""

    final_hidden: torch.Tensor
    workspace: LongshotWorkspaceSource


@dataclass(frozen=True)
class UnifiedFlowPositionTrace:
    active: bool
    source_mode: str
    prompt_cache_length: int
    edited_positions: torch.Tensor | None
    workspace_cycles: int
    workspace_flow_max_displacement: float
    source_delta_max_abs: float
    continuous: ContinuousThoughtTrace

    @property
    def added_positions(self) -> int:
        return self.continuous.added_positions


@dataclass(frozen=True)
class UnifiedFlowPositionLoss:
    loss: torch.Tensor
    trace: UnifiedFlowPositionTrace
    answer_start_position: int


@dataclass(frozen=True)
class UnifiedFlowPositionGeneration:
    token_ids: torch.Tensor
    trace: UnifiedFlowPositionTrace
    answer_start_position: int
    decode_positions: tuple[int, ...]


class UnifiedFlowPositionProcessor(nn.Module):
    """Trainable hybrid around a separately supplied frozen Qwen3 backbone."""

    def __init__(
        self,
        workspace_processor: GlobalCacheProcessor,
        config: UnifiedFlowPositionConfig,
    ) -> None:
        super().__init__()
        if workspace_processor.config.update_stream != "v":
            raise ValueError("unified processor requires a value-only workspace processor")
        if workspace_processor.workspace_flow_refinement is None:
            raise ValueError("unified processor requires workspace flow refinement")
        self.config = config
        self.workspace_processor = workspace_processor
        self.workspace_norm = nn.LayerNorm(workspace_processor.block.output_dim)
        self.workspace_projection = nn.Linear(
            workspace_processor.block.output_dim,
            config.model_width,
            bias=False,
        )
        self.emitter = ContinuousThoughtEmitter(
            ContinuousThoughtEmitterConfig(
                model_width=config.model_width,
                processor_width=config.emitter_width,
                latent_tokens_per_cycle=config.latent_tokens_per_cycle,
            )
        )
        nn.init.normal_(self.workspace_projection.weight, std=0.02)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _validate_backbone(self, backbone: TF) -> None:
        if backbone.config.family != "qwen3":
            raise ValueError("unified flow-position processing currently targets Qwen3")
        if backbone.config.hidden_size != self.config.model_width:
            raise ValueError("unified processor width differs from backbone hidden size")
        if any(parameter.requires_grad for parameter in backbone.parameters()):
            raise ValueError("unified flow-position processing requires a frozen backbone")

    @torch.no_grad()
    def prepare_source(
        self,
        backbone: TF,
        prompt_ids: torch.Tensor,
    ) -> UnifiedSource:
        """Prefill and detach a donor source for shuffled-source controls."""
        self._validate_backbone(backbone)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] <= 0 or prompt_ids.shape[1] <= 0:
            raise ValueError("prompt_ids must be a non-empty rank-2 tensor")
        hidden, cache = backbone.forward_chunk(
            prompt_ids,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
        if not isinstance(cache, WindowedCache):
            raise TypeError("prompt prefill did not return WindowedCache")
        source = self.workspace_processor.encode_source(cache, select_all=self.config.select_all)
        return _detach_source(UnifiedSource(hidden[:, -1], source))

    def process_cache(
        self,
        backbone: TF,
        prompt_final_hidden: torch.Tensor,
        cache: WindowedCache,
        *,
        active: bool,
        source_override: UnifiedSource | None = None,
    ) -> tuple[torch.Tensor, WindowedCache, UnifiedFlowPositionTrace]:
        """Jointly edit recipient V-cache and append continuous causal positions.

        ``active=False`` and ``position_cycles==0`` are exact object-identity
        skips.  A shuffled donor supplies content and head selection, while the
        recipient source always supplies write indices; this prevents donor
        prompt length from changing which recipient position is edited.
        """
        self._validate_backbone(backbone)
        expected_hidden: tuple[int, int] = (int(cache.batch_size), int(self.config.model_width))
        if prompt_final_hidden.shape != expected_hidden:
            raise ValueError(f"prompt_final_hidden must have shape {expected_hidden}")
        if not active or self.config.position_cycles == 0:
            trace = UnifiedFlowPositionTrace(
                active=False,
                source_mode="off",
                prompt_cache_length=cache.seq_len,
                edited_positions=None,
                workspace_cycles=0,
                workspace_flow_max_displacement=0.0,
                source_delta_max_abs=0.0,
                continuous=ContinuousThoughtTrace(0, self.config.latent_tokens_per_cycle, (), ()),
            )
            return prompt_final_hidden, cache, trace

        processor_dtype = next(self.workspace_processor.parameters()).dtype
        recipient = _workspace_source_to(
            self.workspace_processor.encode_source(cache, select_all=self.config.select_all),
            device=cache.keys.device,
            dtype=processor_dtype,
        )
        final_source = prompt_final_hidden
        source_mode = "matched"
        source = recipient
        if source_override is not None:
            _validate_override(source_override, recipient, expected_hidden)
            final_source = source_override.final_hidden.to(
                device=prompt_final_hidden.device,
                dtype=prompt_final_hidden.dtype,
            )
            donor = source_override.workspace
            source = LongshotWorkspaceSource(
                merged=donor.merged.to(device=cache.keys.device, dtype=processor_dtype),
                indices=donor.indices.to(device=cache.keys.device),
                selected_heads=donor.selected_heads.to(device=cache.keys.device),
                latest_indices=recipient.latest_indices,
            )
            source_mode = "shuffled"

        workspace = self.workspace_processor.initialise_workspace(source)
        for _ in range(self.config.workspace_cycles):
            workspace = self.workspace_processor.deliberate(workspace, source)
        refined = self.workspace_processor.refine_workspace(workspace, source)
        edited_cache = self.workspace_processor.write_for_readout(cache, source, workspace)

        pooled = self.workspace_norm(refined).mean(dim=1)
        delta = self.config.source_delta_bound * torch.tanh(self.workspace_projection(pooled))
        fused_source = final_source.to(delta.dtype) + delta
        fused_source = fused_source.to(prompt_final_hidden.dtype)
        readout, positioned_cache, continuous = append_continuous_thoughts(
            backbone,
            self.emitter,
            fused_source,
            edited_cache,
            cycles=self.config.position_cycles,
        )
        flow_displacement = float((refined.detach().float() - workspace.detach().float()).abs().max())
        trace = UnifiedFlowPositionTrace(
            active=True,
            source_mode=source_mode,
            prompt_cache_length=cache.seq_len,
            edited_positions=recipient.latest_indices.detach(),
            workspace_cycles=self.config.workspace_cycles,
            workspace_flow_max_displacement=flow_displacement,
            source_delta_max_abs=float(delta.detach().float().abs().max()),
            continuous=continuous,
        )
        return readout, positioned_cache, trace


def build_unified_flow_position_processor(
    backbone_config: ModelConfig,
    *,
    cache_processor: CacheProcessorConfig,
    hybrid: UnifiedFlowPositionConfig,
    flow_refinement: WorkspaceFlowRefinementConfig | None = None,
) -> UnifiedFlowPositionProcessor:
    """Construct the hybrid without constructing a second backbone."""
    if backbone_config.family != "qwen3":
        raise ValueError("unified flow-position processing currently targets Qwen3")
    refinement = flow_refinement or WorkspaceFlowRefinementConfig(
        solver="midpoint",
        solver_steps=2,
        max_velocity=0.1,
    )
    longshot_config = replace(
        backbone_config,
        variant="longshot",
        cache_processor=cache_processor,
        workspace_flow_refinement=refinement,
    )
    return UnifiedFlowPositionProcessor(GlobalCacheProcessor(longshot_config), hybrid)


def unified_answer_loss(
    backbone: TF,
    processor: UnifiedFlowPositionProcessor,
    *,
    prompt_ids: torch.Tensor,
    response_prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    active: bool,
    source_override: UnifiedSource | None = None,
) -> UnifiedFlowPositionLoss:
    """Teacher-forced final-answer CE for off, matched, or shuffled conditions."""
    processor._validate_backbone(backbone)
    tensors = (prompt_ids, response_prefix_ids, target_ids)
    if any(tensor.ndim != 2 or tensor.shape[1] == 0 for tensor in tensors):
        raise ValueError("prompt, response prefix, and target IDs must be non-empty rank-2 tensors")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("prompt, response prefix, and target batch sizes must match")
    with torch.no_grad():
        prompt_hidden, prompt_cache = backbone.forward_chunk(
            prompt_ids,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("prompt prefill did not return WindowedCache")
    _, processed_cache, trace = processor.process_cache(
        backbone,
        prompt_hidden[:, -1].detach(),
        prompt_cache,
        active=active,
        source_override=source_override,
    )
    answer_inputs = torch.cat([response_prefix_ids, target_ids[:, :-1]], dim=1)
    masked_prefix = target_ids.new_full((target_ids.shape[0], response_prefix_ids.shape[1] - 1), -1)
    answer_targets = torch.cat([masked_prefix, target_ids], dim=1)
    answer_start = prompt_ids.shape[1] + trace.added_positions
    if answer_start != processed_cache.seq_len:
        raise RuntimeError("unified position accounting differs from cache length")
    answer_hidden, _ = backbone.forward_chunk(
        answer_inputs,
        answer_start,
        processed_cache,
        logits=False,
        cache_format="windowed",
    )
    loss = backbone.loss_from_hidden(answer_hidden, answer_targets, "mean")
    return UnifiedFlowPositionLoss(loss=loss, trace=trace, answer_start_position=answer_start)


@torch.no_grad()
def generate_unified_answer(
    backbone: TF,
    processor: UnifiedFlowPositionProcessor,
    *,
    prompt_ids: torch.Tensor,
    response_prefix_ids: torch.Tensor,
    active: bool,
    max_new_tokens: int,
    eos_token_id: int | None,
    valid_vocab_size: int | None = None,
    source_override: UnifiedSource | None = None,
) -> UnifiedFlowPositionGeneration:
    """Greedy generated-answer path used by the full 1,319-row evaluation."""
    processor._validate_backbone(backbone)
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("generation requires one non-empty prompt")
    if response_prefix_ids.ndim != 1:
        raise ValueError("generation response prefix must be rank-1")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_hidden, prompt_cache = backbone.forward_chunk(
        prompt_ids.unsqueeze(0), 0, None, logits=False, cache_format="windowed"
    )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("prompt prefill did not return WindowedCache")
    readout, processed_cache, trace = processor.process_cache(
        backbone,
        prompt_hidden[:, -1],
        prompt_cache,
        active=active,
        source_override=source_override,
    )
    answer_start = prompt_ids.numel() + trace.added_positions
    if response_prefix_ids.numel():
        hidden, decode_cache = backbone.forward_chunk(
            response_prefix_ids.unsqueeze(0),
            answer_start,
            processed_cache,
            logits=False,
            cache_format="windowed",
        )
        if not isinstance(decode_cache, WindowedCache):
            raise TypeError("answer-prefix decode did not return WindowedCache")
        next_logits = backbone.logits_from_hidden(hidden[:, -1:])[:, -1]
    else:
        # Chat templates already end in the assistant generation marker.  The
        # off condition reads logits from that marker; active conditions read
        # them from the final continuous position, with no synthetic token.
        decode_cache = processed_cache
        next_logits = backbone.logits_from_hidden(readout[:, None])[:, -1]
    vocab_size = next_logits.shape[-1] if valid_vocab_size is None else valid_vocab_size
    if vocab_size <= 0 or vocab_size > next_logits.shape[-1]:
        raise ValueError("valid_vocab_size must be within the model vocabulary")
    generated: list[torch.Tensor] = []
    decode_positions: list[int] = []
    for token_index in range(max_new_tokens):
        next_token = torch.argmax(next_logits[:, :vocab_size], dim=-1)
        generated.append(next_token)
        if eos_token_id is not None and bool(torch.all(next_token == eos_token_id)):
            break
        if token_index + 1 >= max_new_tokens:
            break
        decode_positions.append(decode_cache.seq_len)
        hidden, next_cache = backbone.forward_chunk(
            next_token[:, None],
            decode_cache.seq_len,
            decode_cache,
            logits=False,
            cache_format="windowed",
        )
        if not isinstance(next_cache, WindowedCache):
            raise TypeError("greedy decode did not return WindowedCache")
        decode_cache = next_cache
        next_logits = backbone.logits_from_hidden(hidden[:, -1:])[:, -1]
    return UnifiedFlowPositionGeneration(
        token_ids=torch.cat(generated, dim=0),
        trace=trace,
        answer_start_position=answer_start,
        decode_positions=tuple(decode_positions),
    )


def save_unified_checkpoint(
    path: str | Path,
    processor: UnifiedFlowPositionProcessor,
    *,
    construction: dict[str, Any],
    train_state: dict[str, Any] | None = None,
) -> None:
    """Save processor-only weights and enough construction metadata to audit them."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in processor.state_dict().items()}
    save_file(state, destination / "processor.safetensors")
    payload = {"hybrid": asdict(processor.config), "construction": construction}
    (destination / "config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (destination / "train_state.json").write_text(
        json.dumps(train_state or {}, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_unified_weights(
    path: str | Path,
    processor: UnifiedFlowPositionProcessor,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if payload.get("hybrid") != asdict(processor.config):
        raise ValueError("unified checkpoint hybrid config differs from constructed processor")
    processor.load_state_dict(load_file(source / "processor.safetensors", device=str(device)), strict=True)
    train_state = json.loads((source / "train_state.json").read_text(encoding="utf-8"))
    if not isinstance(train_state, dict):
        raise ValueError("invalid unified checkpoint train state")
    return train_state


def _workspace_source_to(
    source: LongshotWorkspaceSource,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> LongshotWorkspaceSource:
    return LongshotWorkspaceSource(
        merged=source.merged.to(device=device, dtype=dtype),
        indices=source.indices.to(device=device),
        selected_heads=source.selected_heads.to(device=device),
        latest_indices=source.latest_indices.to(device=device),
    )


def _detach_source(source: UnifiedSource) -> UnifiedSource:
    workspace = source.workspace
    return UnifiedSource(
        source.final_hidden.detach(),
        LongshotWorkspaceSource(
            merged=workspace.merged.detach(),
            indices=workspace.indices.detach(),
            selected_heads=workspace.selected_heads.detach(),
            latest_indices=workspace.latest_indices.detach(),
        ),
    )


def _validate_override(
    source: UnifiedSource,
    recipient: LongshotWorkspaceSource,
    expected_hidden: Sequence[int],
) -> None:
    if source.final_hidden.shape != expected_hidden:
        raise ValueError(f"source override final hidden must have shape {expected_hidden}")
    donor = source.workspace
    if donor.merged.ndim != 4 or donor.merged.shape[0] != expected_hidden[0]:
        raise ValueError("source override workspace batch differs from recipient")
    if donor.merged.shape[1] != recipient.merged.shape[1] or donor.merged.shape[-1] != recipient.merged.shape[-1]:
        raise ValueError("source override workspace cache shape differs from recipient")
    if donor.selected_heads.shape != recipient.selected_heads.shape:
        raise ValueError("source override selected-head shape differs from recipient")
