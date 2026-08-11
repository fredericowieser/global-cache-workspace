"""Exploratory continuous-thought training on a frozen GCW Qwen backbone.

The emitter creates model-width embeddings. Each embedding is decoded through
normal frozen transformer blocks and therefore becomes a genuine causal K/V
position. Only the emitter is intended to be trainable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
import torch
import torch.nn as nn

from gcw.models.transformer import TF
from gcw.models.windowed_cache import WindowedCache


@dataclass(frozen=True)
class ContinuousThoughtEmitterConfig:
    model_width: int
    processor_width: int = 512
    latent_tokens_per_cycle: int = 1

    def __post_init__(self) -> None:
        if self.model_width <= 0:
            raise ValueError("model_width must be positive")
        if self.processor_width <= 0:
            raise ValueError("processor_width must be positive")
        if self.latent_tokens_per_cycle <= 0:
            raise ValueError("latent_tokens_per_cycle must be positive")


@dataclass(frozen=True)
class ContinuousThoughtTrace:
    cycles: int
    latent_tokens_per_cycle: int
    start_positions: tuple[int, ...]
    cache_lengths: tuple[int, ...]

    @property
    def added_positions(self) -> int:
        return self.cycles * self.latent_tokens_per_cycle


@dataclass(frozen=True)
class ContinuousThoughtLoss:
    loss: torch.Tensor
    trace: ContinuousThoughtTrace
    prompt_cache_length: int
    answer_start_position: int


@dataclass(frozen=True)
class ContinuousThoughtGeneration:
    """Greedy discrete continuation produced after continuous cache positions."""

    token_ids: torch.Tensor
    trace: ContinuousThoughtTrace
    prompt_cache_length: int
    answer_start_position: int
    decode_positions: tuple[int, ...]


class ContinuousThoughtEmitter(nn.Module):
    """Small recurrent processor that emits one latent-token chunk per cycle."""

    def __init__(self, config: ContinuousThoughtEmitterConfig) -> None:
        super().__init__()
        self.config = config
        width = config.processor_width
        self.source_projection = nn.Linear(config.model_width, width, bias=False)
        self.feedback_projection = nn.Linear(config.model_width, width, bias=False)
        self.recurrent_projection = nn.Linear(width, width, bias=False)
        self.slot_embeddings = nn.Parameter(torch.empty(config.latent_tokens_per_cycle, width))
        self.output_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, config.model_width, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.slot_embeddings, std=0.02)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def _processor_dtype(self) -> torch.dtype:
        return self.output_projection.weight.dtype

    def initial_state(self, source: torch.Tensor) -> torch.Tensor:
        self._validate_model_width(source, "source")
        return torch.tanh(self.source_projection(source.to(self._processor_dtype())))

    def forward(
        self,
        source: torch.Tensor,
        feedback: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(model_width_latents, recurrent_state)`` for one cycle."""
        self._validate_model_width(source, "source")
        self._validate_model_width(feedback, "feedback")
        expected_state = (source.shape[0], self.config.processor_width)
        if state.shape != expected_state:
            raise ValueError(f"state must have shape {expected_state}, got {tuple(state.shape)}")
        dtype = self._processor_dtype()
        source = source.to(dtype)
        feedback = feedback.to(dtype)
        state = state.to(dtype)
        next_state = torch.tanh(
            self.source_projection(source) + self.feedback_projection(feedback) + self.recurrent_projection(state)
        )
        slots = next_state[:, None, :] + self.slot_embeddings[None, :, :]
        delta = self.output_projection(torch.tanh(self.output_norm(slots)))
        # A residual source gives the randomly initialized processor a stable,
        # in-distribution decoder input while gradients train its emitted delta.
        latents = source[:, None, :] + delta
        return latents, next_state

    def _validate_model_width(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.ndim != 2 or tensor.shape[1] != self.config.model_width:
            raise ValueError(f"{name} must have shape (batch, {self.config.model_width}), got {tuple(tensor.shape)}")


def frozen_qwen_backbone(model: nn.Module) -> TF:
    """Freeze a TF or Longshot-like wrapper and return its native TF backbone."""
    model.requires_grad_(False)
    model.eval()
    candidate: nn.Module = getattr(model, "backbone", model)
    if not isinstance(candidate, TF):
        raise TypeError("continuous thought currently requires a native GCW TF backbone")
    if candidate.config.family != "qwen3":
        raise ValueError("continuous thought production path currently targets Qwen3")
    return candidate


def append_continuous_thoughts(
    backbone: TF,
    emitter: ContinuousThoughtEmitter,
    source: torch.Tensor,
    cache: WindowedCache,
    *,
    cycles: int,
) -> tuple[torch.Tensor, WindowedCache, ContinuousThoughtTrace]:
    """Append ``cycles * latent_tokens_per_cycle`` normal causal positions.

    ``cycles == 0`` is an exact skip: the emitter and embedding decoder are not
    called, the original cache object is returned, and ``source`` is returned as
    the readout hidden state.
    """
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if source.ndim != 2 or source.shape != (cache.batch_size, backbone.config.hidden_size):
        raise ValueError("source must match cache batch size and backbone hidden size")
    if emitter.config.model_width != backbone.config.hidden_size:
        raise ValueError("emitter model width must match backbone hidden size")
    if cycles == 0:
        return source, cache, ContinuousThoughtTrace(0, emitter.config.latent_tokens_per_cycle, (), ())

    state = emitter.initial_state(source)
    feedback = source
    starts: list[int] = []
    lengths: list[int] = []
    for _ in range(cycles):
        latents, state = emitter(source, feedback, state)
        latents = latents.to(dtype=backbone.embed_tokens.weight.dtype)
        start = cache.seq_len
        hidden, next_cache = backbone.forward_embedding_chunk(
            latents,
            start,
            cache,
            logits=False,
            cache_format="windowed",
        )
        if not isinstance(next_cache, WindowedCache):
            raise TypeError("embedding decode did not return WindowedCache")
        cache = next_cache
        feedback = hidden[:, -1]
        starts.append(start)
        lengths.append(cache.seq_len)
    trace = ContinuousThoughtTrace(
        cycles=cycles,
        latent_tokens_per_cycle=emitter.config.latent_tokens_per_cycle,
        start_positions=tuple(starts),
        cache_lengths=tuple(lengths),
    )
    return feedback, cache, trace


def continuous_thought_answer_loss(
    backbone: TF,
    emitter: ContinuousThoughtEmitter,
    *,
    prompt_ids: torch.Tensor,
    response_prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    cycles: int,
    source_override: torch.Tensor | None = None,
) -> ContinuousThoughtLoss:
    """Compute teacher-forced answer CE after optional continuous thoughts."""
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("continuous-thought answer loss requires a frozen backbone")
    tensors = (prompt_ids, response_prefix_ids, target_ids)
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("prompt, response prefix, and target IDs must be rank-2")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("prompt, response prefix, and target batch sizes must match")
    if any(tensor.shape[1] == 0 for tensor in tensors):
        raise ValueError("prompt, response prefix, and target sequences must be non-empty")

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
    source = prompt_hidden[:, -1].detach()
    if source_override is not None:
        if source_override.shape != source.shape:
            raise ValueError(
                f"source_override must have shape {tuple(source.shape)}, got {tuple(source_override.shape)}"
            )
        source = source_override.detach().to(device=source.device, dtype=source.dtype)
    _, thought_cache, trace = append_continuous_thoughts(
        backbone,
        emitter,
        source,
        prompt_cache,
        cycles=cycles,
    )

    answer_inputs = torch.cat([response_prefix_ids, target_ids[:, :-1]], dim=1)
    masked_prefix = target_ids.new_full((target_ids.shape[0], response_prefix_ids.shape[1] - 1), -1)
    answer_targets = torch.cat([masked_prefix, target_ids], dim=1)
    answer_start = prompt_ids.shape[1] + trace.added_positions
    if answer_start != thought_cache.seq_len:
        raise RuntimeError("continuous-thought position accounting differs from cache length")
    answer_hidden, _ = backbone.forward_chunk(
        answer_inputs,
        answer_start,
        thought_cache,
        logits=False,
        cache_format="windowed",
    )
    loss = backbone.loss_from_hidden(answer_hidden, answer_targets, "mean")
    return ContinuousThoughtLoss(
        loss=loss,
        trace=trace,
        prompt_cache_length=prompt_cache.seq_len,
        answer_start_position=answer_start,
    )


@torch.no_grad()
def generate_continuous_thought_answer(
    backbone: TF,
    emitter: ContinuousThoughtEmitter,
    *,
    prompt_ids: torch.Tensor,
    response_prefix_ids: torch.Tensor,
    cycles: int,
    max_new_tokens: int,
    eos_token_id: int | None,
    valid_vocab_size: int | None = None,
    source_override: torch.Tensor | None = None,
) -> ContinuousThoughtGeneration:
    """Greedily generate ordinary token positions after optional thoughts.

    The recipient prompt always supplies the K/V cache. ``source_override``
    replaces only the prompt-final hidden state supplied to the emitter, which
    makes source-transplant controls precise. ``cycles == 0`` remains an exact
    skip and never invokes the emitter.
    """
    if any(parameter.requires_grad for parameter in backbone.parameters()):
        raise ValueError("continuous-thought generation requires a frozen backbone")
    if prompt_ids.ndim != 1 or prompt_ids.numel() == 0:
        raise ValueError("generation requires one non-empty prompt")
    if response_prefix_ids.ndim != 1 or response_prefix_ids.numel() == 0:
        raise ValueError("generation requires one non-empty response prefix")
    if cycles < 0:
        raise ValueError("cycles must be non-negative")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    prompt_hidden, prompt_cache = backbone.forward_chunk(
        prompt_ids.unsqueeze(0),
        0,
        None,
        logits=False,
        cache_format="windowed",
    )
    if not isinstance(prompt_cache, WindowedCache):
        raise TypeError("prompt prefill did not return WindowedCache")
    source = prompt_hidden[:, -1]
    if source_override is not None:
        if source_override.shape != source.shape:
            raise ValueError(
                f"source_override must have shape {tuple(source.shape)}, got {tuple(source_override.shape)}"
            )
        source = source_override.to(device=source.device, dtype=source.dtype)
    _, thought_cache, trace = append_continuous_thoughts(
        backbone,
        emitter,
        source,
        prompt_cache,
        cycles=cycles,
    )
    answer_start = prompt_ids.numel() + trace.added_positions
    if answer_start != thought_cache.seq_len:
        raise RuntimeError("continuous-thought position accounting differs from cache length")
    hidden, decode_cache = backbone.forward_chunk(
        response_prefix_ids.unsqueeze(0),
        answer_start,
        thought_cache,
        logits=False,
        cache_format="windowed",
    )
    if not isinstance(decode_cache, WindowedCache):
        raise TypeError("answer-prefix decode did not return WindowedCache")
    next_logits = backbone.logits_from_hidden(hidden[:, -1:])[:, -1]
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
    return ContinuousThoughtGeneration(
        token_ids=torch.cat(generated, dim=0),
        trace=trace,
        prompt_cache_length=prompt_cache.seq_len,
        answer_start_position=answer_start,
        decode_positions=tuple(decode_positions),
    )


def save_emitter_checkpoint(
    path: str | Path,
    emitter: ContinuousThoughtEmitter,
    *,
    train_state: dict[str, Any] | None = None,
) -> None:
    """Save an emitter-only checkpoint; frozen backbone tensors are excluded."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in emitter.state_dict().items()}
    save_file(state, destination / "emitter.safetensors")
    (destination / "config.json").write_text(
        json.dumps({"emitter": asdict(emitter.config)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "train_state.json").write_text(
        json.dumps(train_state or {}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_emitter_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[ContinuousThoughtEmitter, dict[str, Any]]:
    source = Path(path)
    payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"emitter"}:
        raise ValueError("invalid continuous-thought emitter checkpoint config")
    emitter = ContinuousThoughtEmitter(ContinuousThoughtEmitterConfig(**payload["emitter"]))
    emitter.load_state_dict(load_file(source / "emitter.safetensors", device=str(device)), strict=True)
    emitter.to(device)
    train_state = json.loads((source / "train_state.json").read_text(encoding="utf-8"))
    if not isinstance(train_state, dict):
        raise ValueError("invalid continuous-thought emitter train state")
    return emitter, train_state
