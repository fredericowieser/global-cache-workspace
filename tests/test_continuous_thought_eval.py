from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from gcw.config import ModelConfig
from gcw.continuous_thought import (
    ContinuousThoughtEmitter,
    ContinuousThoughtEmitterConfig,
    frozen_qwen_backbone,
    generate_continuous_thought_answer,
)
from gcw.models import build_model


def _config() -> ModelConfig:
    return ModelConfig(
        variant="tf",
        family="qwen3",
        sequence_len=64,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        rope_theta=10_000.0,
        norm_eps=1e-6,
        tie_embeddings=False,
        qk_norm=True,
        rotary_style="llama",
    )


def _frozen() -> tuple[torch.nn.Module, ContinuousThoughtEmitter]:
    torch.manual_seed(91)
    model = build_model(_config())
    model.init_weights()
    backbone = frozen_qwen_backbone(model)
    torch.manual_seed(92)
    emitter = ContinuousThoughtEmitter(
        ContinuousThoughtEmitterConfig(
            model_width=32,
            processor_width=24,
            latent_tokens_per_cycle=2,
        )
    )
    emitter.eval()
    return backbone, emitter


def _evaluator() -> ModuleType:
    script = Path(__file__).parents[1] / "scripts" / "evaluate_continuous_thought_records.py"
    spec = importlib.util.spec_from_file_location("evaluate_continuous_thought_records", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_greedy_generation_uses_continuous_and_discrete_cache_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backbone, emitter = _frozen()
    prompt = torch.tensor([3, 5, 8, 13])
    prefix = torch.tensor([21, 22])
    token_calls: list[tuple[int, list[int]]] = []
    original_forward_chunk = backbone.forward_chunk

    def observed_forward_chunk(input_ids, start_pos, cache=None, **kwargs):
        token_calls.append((start_pos, input_ids.flatten().tolist()))
        return original_forward_chunk(input_ids, start_pos, cache, **kwargs)

    forced = iter((11, 12, 2))

    def forced_logits(hidden):
        logits = torch.full((*hidden.shape[:-1], 64), -100.0, device=hidden.device)
        logits[..., next(forced)] = 100.0
        return logits

    monkeypatch.setattr(backbone, "forward_chunk", observed_forward_chunk)
    monkeypatch.setattr(backbone, "logits_from_hidden", forced_logits)
    result = generate_continuous_thought_answer(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        cycles=2,
        max_new_tokens=8,
        eos_token_id=2,
        valid_vocab_size=64,
    )

    assert result.token_ids.tolist() == [11, 12, 2]
    assert result.trace.start_positions == (4, 6)
    assert result.trace.cache_lengths == (6, 8)
    assert result.answer_start_position == 8
    assert result.decode_positions == (10, 11)
    assert token_calls == [(0, [3, 5, 8, 13]), (8, [21, 22]), (10, [11]), (11, [12])]


def test_k0_generation_has_ordinary_greedy_token_parity_and_skips_emitter() -> None:
    backbone, emitter = _frozen()
    prompt = torch.tensor([2, 4, 6, 8])
    prefix = torch.tensor([19, 20])
    emitter_calls = 0

    def count_call(_module, _inputs, _output) -> None:
        nonlocal emitter_calls
        emitter_calls += 1

    handle = emitter.register_forward_hook(count_call)
    actual = generate_continuous_thought_answer(
        backbone,
        emitter,
        prompt_ids=prompt,
        response_prefix_ids=prefix,
        cycles=0,
        max_new_tokens=5,
        eos_token_id=None,
        valid_vocab_size=64,
    )
    handle.remove()
    ordinary_input = torch.cat([prompt, prefix]).unsqueeze(0)
    ordinary = backbone.generate(ordinary_input, max_new_tokens=5, eos_token_id=None)

    assert emitter_calls == 0
    assert actual.trace.added_positions == 0
    assert actual.answer_start_position == prompt.numel()
    assert torch.equal(actual.token_ids, ordinary[0, ordinary_input.shape[1] :])


def test_matched_and_shuffled_generation_transplant_only_emitter_source() -> None:
    backbone, emitter = _frozen()
    recipient = torch.tensor([3, 5, 8, 13])
    donor = torch.tensor([13, 8, 5, 3])
    prefix = torch.tensor([21, 22])
    with torch.no_grad():
        recipient_hidden, _ = backbone.forward_chunk(
            recipient.unsqueeze(0), 0, None, logits=False, cache_format="windowed"
        )
        donor_hidden, _ = backbone.forward_chunk(
            donor.unsqueeze(0), 0, None, logits=False, cache_format="windowed"
        )
    observed: list[torch.Tensor] = []

    def capture_source(_module, inputs) -> None:
        observed.append(inputs[0].detach().clone())

    handle = emitter.register_forward_pre_hook(capture_source)
    matched = generate_continuous_thought_answer(
        backbone,
        emitter,
        prompt_ids=recipient,
        response_prefix_ids=prefix,
        cycles=2,
        max_new_tokens=1,
        eos_token_id=None,
    )
    matched_sources = observed[:]
    observed.clear()
    shuffled = generate_continuous_thought_answer(
        backbone,
        emitter,
        prompt_ids=recipient,
        response_prefix_ids=prefix,
        cycles=2,
        max_new_tokens=1,
        eos_token_id=None,
        source_override=donor_hidden[:, -1],
    )
    handle.remove()

    assert matched.trace == shuffled.trace
    assert matched.answer_start_position == shuffled.answer_start_position
    assert len(matched_sources) == len(observed) == 2
    assert all(torch.equal(source, recipient_hidden[:, -1]) for source in matched_sources)
    assert all(torch.equal(source, donor_hidden[:, -1]) for source in observed)
    assert not torch.equal(recipient_hidden[:, -1], donor_hidden[:, -1])


def test_answer_extraction_and_canonical_jsonl_are_exact_and_deterministic(tmp_path: Path) -> None:
    evaluator = _evaluator()
    assert evaluator.extract_canonical_answer(r"work 99; final \boxed{\frac{3}{2}}") == "3/2"
    assert evaluator.extract_canonical_answer("work 1; therefore -2,500.00") == "-2500"
    assert evaluator.extract_canonical_answer("no numeric answer") is None
    assert evaluator.exact_answers_match(r"\boxed{1.50}", r"\boxed{3/2}")
    assert not evaluator.exact_answers_match(r"\boxed{4}", r"\boxed{5}")

    rows = [{"z": "line\u2028separator", "a": 2}, {"a": 1, "z": "é"}]
    first = evaluator.canonical_jsonl_bytes(rows)
    second = evaluator.canonical_jsonl_bytes(rows)
    assert first == second
    assert first == (
        '{"a":2,"z":"line\u2028separator"}\n{"a":1,"z":"é"}\n'.encode("utf-8")
    )
    path = tmp_path / "rows.jsonl"
    digest = evaluator.write_canonical_jsonl(path, rows)
    assert path.read_bytes() == first
    assert digest == evaluator._sha256_bytes(first)
    with pytest.raises(FileExistsError):
        evaluator.write_canonical_jsonl(path, rows)


def test_evaluator_pins_authenticated_heldout_interval_and_checkpoint_default() -> None:
    evaluator = _evaluator()
    args = evaluator.build_parser().parse_args(["--output", "out"])
    assert (evaluator.HELDOUT_START, evaluator.HELDOUT_STOP, evaluator.HELDOUT_ROWS) == (6000, 6256, 256)
    assert str(args.emitter_checkpoint) == (
        "artifacts/experiments/explore/continuous-thought/epoch-k2-l2-lr3e4/emitter-checkpoint"
    )
    assert str(args.tokenizer) == "artifacts/models/Qwen3-0.6B-base-v1"
