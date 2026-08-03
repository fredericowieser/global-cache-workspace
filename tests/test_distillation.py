from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gcw.config import CacheProcessorConfig, DataConfig, DistillationConfig, ModelConfig, TrainConfig
from gcw.distill.cache import truncate_last_cache_token
from gcw.distill.data import iter_openmath_prompts
from gcw.distill.objective import clipped_forward_kl
from gcw.distill.prompt import ChatPrompt, format_chat_prompt
from gcw.distill.session import BGTDecodeSession, sample_next_token
from gcw.distill.teacher import split_teacher_reasoning, teacher_answer_logits
from gcw.distill.schedule import processor_event_positions
from gcw.models import build_model
from gcw.models.cache_state import CacheState
from gcw.data.tokenizers import TransformersTokenizer


class FakeChatTokenizer:
    eot_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize
        assert messages == [{"role": "user", "content": "2+2?"}]
        base = [10, 11, 12]
        if not add_generation_prompt:
            return base
        return base + ([20, 21] if enable_thinking else [30, 31])


def _distillation_train_config(**overrides) -> TrainConfig:
    kwargs = {
        "model": ModelConfig(
            variant="bgt",
            family="qwen3",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(process_every=64, topk=4, hidden_size=16, intermediate_size=32, num_heads=2),
        ),
        "data": DataConfig(kind="openmath_prompts", dataset="prompts.jsonl", tokenizer="hf:Qwen/Qwen3-0.6B"),
        "mode": "cot_distillation",
        "freeze_backbone": True,
        "backbone_checkpoint": "checkpoint",
        "batch_size": 1,
        "distillation": DistillationConfig(),
    }
    kwargs.update(overrides)
    return TrainConfig(**kwargs)


def test_distillation_config_accepts_explicit_v0_contract():
    config = _distillation_train_config()

    assert config.distillation is not None
    assert config.distillation.response_process_every == 64
    assert config.distillation.teacher_thinking
    assert not config.distillation.student_thinking


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"freeze_backbone": False}, "frozen backbone"),
        ({"model": ModelConfig()}, "BGT"),
        ({"batch_size": 2}, "batch_size=1"),
    ],
)
def test_distillation_config_rejects_unsupported_v0(overrides, message):
    with pytest.raises(ValueError, match=message):
        _distillation_train_config(**overrides)


def test_processor_event_positions_are_response_relative():
    assert processor_event_positions(max_response_tokens=0, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=63, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=64, process_every=64) == (0, 64)
    assert processor_event_positions(max_response_tokens=130, process_every=64) == (0, 64, 128)


def test_format_chat_prompt_splits_assistant_trigger_without_retokenising():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=False)

    assert prompt.base_ids == (10, 11, 12)
    assert prompt.trigger_ids == (30, 31)
    assert prompt.full_ids == (10, 11, 12, 30, 31)


def test_format_chat_prompt_keeps_thinking_mode_in_trigger():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=True)

    assert prompt.trigger_ids == (20, 21)


def test_clipped_forward_kl_matches_released_opsd_order_and_masks_tokens():
    student = torch.tensor([[[4.0, -2.0], [0.5, -0.5], [-3.0, 3.0]]], requires_grad=True)
    teacher = torch.tensor([[[-2.0, 4.0], [1.0, -1.0], [3.0, -3.0]]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    temperature = 1.1
    clip = 0.05

    loss = clipped_forward_kl(student, teacher, mask, temperature=temperature, token_clip=clip)

    student_log_probs = F.log_softmax(student / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher.detach() / temperature, dim=-1)
    per_vocab = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
    expected = per_vocab.clamp(max=clip)[mask].sum() / mask.sum()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_clipped_forward_kl_rejects_empty_answer_mask():
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="at least one answer token"):
        clipped_forward_kl(logits, logits, torch.zeros(1, 2, dtype=torch.bool))


def test_truncate_last_cache_token_preserves_earlier_values_and_gradients():
    keys = torch.arange(1 * 1 * 4 * 2, dtype=torch.float32).reshape(1, 1, 4, 2).requires_grad_()
    values = (keys.detach() + 100).requires_grad_()
    scores = torch.arange(4, dtype=torch.float32).reshape(1, 4).requires_grad_()
    cache = CacheState([keys], [values], scores)

    truncated = truncate_last_cache_token(cache)

    assert truncated.seq_len == 3
    torch.testing.assert_close(truncated.keys[0], keys[:, :, :-1])
    torch.testing.assert_close(truncated.values[0], values[:, :, :-1])
    torch.testing.assert_close(truncated.scores, scores[:, :-1])
    truncated.keys[0].sum().backward()
    assert keys.grad is not None
    assert torch.count_nonzero(keys.grad[:, :, -1]) == 0


def test_truncate_last_cache_token_rejects_empty_cache():
    cache = CacheState([], [], torch.empty(1, 0))
    with pytest.raises(ValueError, match="empty cache"):
        truncate_last_cache_token(cache)



def _tiny_bgt():
    model = build_model(
        ModelConfig(
            variant="bgt",
            family="qwen3",
            sequence_len=256,
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(
                process_every=64,
                topk=4,
                hidden_size=16,
                intermediate_size=32,
                num_heads=1,
                num_kv_heads=1,
            ),
        )
    )
    model.init_weights()
    return model


class RecordingZeroProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls: list[tuple[int, bool]] = []

    def forward(self, cache: CacheState, *, select_all: bool = False) -> CacheState:
        self.calls.append((cache.seq_len, select_all))
        return CacheState(
            [torch.zeros_like(key) for key in cache.keys],
            [torch.zeros_like(value) for value in cache.values],
            cache.scores,
        )


def test_replay_processes_boundary_before_token_one_and_every_64_response_tokens():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor
    prompt = ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4, 5))
    response_ids = tuple([6] * 65)

    with torch.no_grad():
        _, unprocessed_cache = model.backbone.forward_chunk(torch.tensor([prompt.base_ids]), 0, logits=False)
        unprocessed_logits, _ = model.backbone.forward_chunk(
            torch.tensor([prompt.trigger_ids]),
            len(prompt.base_ids),
            unprocessed_cache,
        )
        replay = BGTDecodeSession(model, process_every=64).replay(prompt, response_ids)

    assert processor.calls == [(3, False), (69, False)]
    assert replay.event_cache_lengths == (3, 69)
    assert replay.logits.shape == (1, 65, 64)
    assert not torch.allclose(replay.logits[:, 0], unprocessed_logits[:, -1])


def test_replay_select_all_is_explicit_and_applies_to_every_event():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor

    BGTDecodeSession(model, process_every=2, select_all=True).replay(
        ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)),
        (4, 5, 6),
    )

    assert processor.calls == [(2, True), (5, True)]


def test_replay_backpropagates_only_into_bgt_when_backbone_is_frozen():
    model = _tiny_bgt()
    model.backbone.requires_grad_(False)
    session = BGTDecodeSession(model, process_every=2)

    replay = session.replay(ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,)), (5, 6, 7))
    replay.logits.square().mean().backward()

    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    processor_grads = [parameter.grad for parameter in model.cache_processor.parameters()]
    assert processor_grads
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in processor_grads)



def test_sample_next_token_top_k_one_is_argmax():
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    token = sample_next_token(logits, temperature=1.1, top_p=1.0, top_k=1)
    assert token.item() == 1


def test_rollout_and_replay_share_response_relative_event_schedule():
    model = _tiny_bgt()
    processor = RecordingZeroProcessor()
    model.cache_processor = processor
    session = BGTDecodeSession(model, process_every=2)
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    generator = torch.Generator().manual_seed(7)

    rollout = session.rollout(
        prompt,
        max_response_tokens=5,
        eos_token_id=None,
        temperature=1.1,
        top_p=1.0,
        top_k=0,
        generator=generator,
    )
    rollout_calls = tuple(processor.calls)
    processor.calls.clear()
    replay = session.replay(prompt, rollout.response_ids)

    assert len(rollout.response_ids) == 5
    assert rollout.event_cache_lengths == (2, 5, 7)
    assert replay.event_cache_lengths == rollout.event_cache_lengths
    assert tuple(processor.calls) == rollout_calls



class PositionLogitBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.inputs: list[torch.Tensor] = []

    def forward_full(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.inputs.append(input_ids.detach().clone())
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        return positions[None, :, None].expand(input_ids.shape[0], -1, self.vocab_size)


def test_teacher_reasoning_is_context_but_not_a_kl_position():
    backbone = PositionLogitBackbone()
    teacher_context = (10, 11, 12, 90, 91, 92)
    response = (20, 21)

    logits = teacher_answer_logits(backbone, teacher_context, response)

    assert backbone.inputs[0].tolist() == [list(teacher_context + response)]
    assert logits.shape == (1, 2, backbone.vocab_size)
    torch.testing.assert_close(logits[:, 0], torch.full((1, backbone.vocab_size), 5.0))
    torch.testing.assert_close(logits[:, 1], torch.full((1, backbone.vocab_size), 6.0))
    assert not logits.requires_grad


def test_split_teacher_reasoning_stops_after_close_separator():
    generated = (40, 41, 50, 51, 52, 60, 61)
    assert split_teacher_reasoning(generated, close_ids=(50, 51, 52)) == (40, 41, 50, 51, 52)


def test_split_teacher_reasoning_rejects_missing_close_separator():
    with pytest.raises(ValueError, match="close separator"):
        split_teacher_reasoning((1, 2, 3), close_ids=(8, 9))


def test_openmath_prompt_reader_ignores_answer_content(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"prompt": "p0", "answer": "secret-a"}\n'
        '{"problem": "p1", "solution": "secret-b"}\n'
        '{"question": "p2", "output": "secret-c"}\n',
        encoding="utf-8",
    )

    records = list(iter_openmath_prompts(path))

    assert [record.prompt for record in records] == ["p0", "p1", "p2"]
    assert [record.source_index for record in records] == [0, 1, 2]
    assert all("secret" not in record.prompt for record in records)



class FakeHFTokenizerBackend:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, 2, 3]


def test_transformers_tokenizer_exposes_thinking_chat_template():
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = FakeHFTokenizerBackend()

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert ids == [1, 2, 3]
    assert tokenizer.enc.calls[0][1] == {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
