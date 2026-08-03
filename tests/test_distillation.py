from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gcw.checkpoint import load_checkpoint
from gcw.config import CacheProcessorConfig, DataConfig, DistillationConfig, ModelConfig, TrainConfig, load_train_config
from gcw.distill import trainer as distill_trainer
from gcw.distill.cache import truncate_last_cache_token
from gcw.distill.data import iter_openmath_prompts
from gcw.distill.objective import clipped_forward_kl
from gcw.distill.pipeline import DistillationExample, distillation_example_loss
from gcw.distill.prompt import ChatPrompt, format_chat_prompt, format_teacher_prompt
from gcw.distill.session import BGTDecodeSession, sample_next_token
from gcw.distill.teacher import sample_backbone_completion, split_teacher_reasoning, teacher_answer_logits
from gcw.distill.traces import TeacherTraceStore
from safetensors.torch import save_file
from gcw.distill.schedule import processor_event_positions
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.models.cache_state import CacheState, StaticCacheState
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

    def encode(self, text: str):
        assert text == "<think>\n"
        return [20, 21]


class ThinkingAgnosticTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )


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
        "max_steps": 1,
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


def test_distillation_repeated_epochs_require_explicit_overfit_diagnostic():
    with pytest.raises(ValueError, match="overfit diagnostic"):
        DistillationConfig(epochs=2)

    with pytest.raises(ValueError, match="expected_locked_examples"):
        DistillationConfig(epochs=2, overfit_diagnostic=True)

    diagnostic = DistillationConfig(
        epochs=2,
        overfit_diagnostic=True,
        expected_locked_examples=1,
    )

    assert diagnostic.epochs == 2
    assert diagnostic.overfit_diagnostic
    assert diagnostic.expected_locked_examples == 1


@pytest.mark.parametrize(
    ("path", "take_rows", "epochs", "expected_locked"),
    [
        ("configs/train/opsd/qwen3_06b_opsd_overfit16_smoke.yaml", 1, 2, 1),
        ("configs/train/opsd/qwen3_06b_opsd_overfit16x20.yaml", 21, 20, 16),
    ],
)
def test_opsd_overfit_configs_are_explicit_diagnostics(path, take_rows, epochs, expected_locked):
    config = load_train_config(path)

    assert config.data.sft_take_rows == take_rows
    assert config.distillation is not None
    assert config.distillation.epochs == epochs
    assert config.distillation.overfit_diagnostic
    assert config.distillation.expected_locked_examples == expected_locked


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"freeze_backbone": False}, "frozen backbone"),
        ({"model": ModelConfig()}, "BGT"),
        ({"batch_size": 2}, "batch_size=1"),
        ({"max_steps": 2}, "max_steps=1"),
    ],
)
def test_distillation_config_rejects_unsupported_v0(overrides, message):
    with pytest.raises(ValueError, match=message):
        _distillation_train_config(**overrides)


def test_processor_event_positions_are_response_relative():
    assert processor_event_positions(max_response_tokens=0, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=63, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=64, process_every=64) == (0,)
    assert processor_event_positions(max_response_tokens=65, process_every=64) == (0, 64)
    assert processor_event_positions(max_response_tokens=130, process_every=64) == (0, 64, 128)


def test_format_chat_prompt_splits_assistant_trigger_without_retokenising():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=False)

    assert prompt.base_ids == (10, 11, 12)
    assert prompt.trigger_ids == (30, 31)
    assert prompt.full_ids == (10, 11, 12, 30, 31)


def test_format_chat_prompt_keeps_thinking_mode_in_trigger():
    prompt = format_chat_prompt(FakeChatTokenizer(), "2+2?", enable_thinking=True)

    assert prompt.trigger_ids == (20, 21)


def test_teacher_prompt_does_not_duplicate_template_thinking_open():
    prompt = format_teacher_prompt(FakeChatTokenizer(), "2+2?", reasoning_open_text="<think>\n")

    assert prompt.trigger_ids == (20, 21)


def test_teacher_prompt_appends_thinking_open_when_template_ignores_flag():
    prompt = format_teacher_prompt(ThinkingAgnosticTokenizer(), "2+2?", reasoning_open_text="<think>\n")

    assert prompt.trigger_ids == (30, 31, 20, 21)


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

    def forward(
        self,
        cache: CacheState | StaticCacheState,
        *,
        select_all: bool = False,
    ) -> CacheState | StaticCacheState:
        self.calls.append((cache.seq_len, select_all))
        if isinstance(cache, StaticCacheState):
            for key, value in zip(cache.keys, cache.values, strict=True):
                key.zero_()
                value.zero_()
            return cache
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


def test_teacher_reasoning_closes_terminal_eot_for_direct_answer_checkpoint():
    reasoning = split_teacher_reasoning(
        (1, 2, 3, 0),
        close_ids=(8, 9),
        eos_token_id=0,
        close_at_eos=True,
    )

    assert reasoning == (1, 2, 3, 8, 9)


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
        return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}


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



def test_distillation_example_loss_uses_teacher_cot_and_trains_only_bgt():
    model = _tiny_bgt()
    model.backbone.requires_grad_(False)
    example = DistillationExample(
        student_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,)),
        teacher_prompt=ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(8, 9)),
        teacher_reasoning_ids=(10, 11, 12),
        response_ids=(5, 6, 7),
    )

    result = distillation_example_loss(
        model,
        example,
        BGTDecodeSession(model, process_every=2),
        temperature=1.1,
        token_clip=0.05,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.answer_tokens == 3
    assert result.event_cache_lengths == (3, 6)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(parameter.grad is not None for parameter in model.cache_processor.parameters())



def test_teacher_sampling_honours_reasoning_stop_sequence():
    backbone = _tiny_bgt().backbone
    prompt_ids = (1, 2, 3)
    full = sample_backbone_completion(
        backbone,
        prompt_ids,
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=None,
        temperature=1.1,
        top_p=1.0,
        top_k=1,
    )
    stopped = sample_backbone_completion(
        backbone,
        prompt_ids,
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=(full[0],),
        temperature=1.1,
        top_p=1.0,
        top_k=1,
    )

    assert len(full) == 3
    assert stopped == (full[0],)



def test_teacher_trace_store_round_trips_and_rejects_prompt_drift(tmp_path):
    path = tmp_path / "teacher-traces.jsonl"
    store = TeacherTraceStore(path, fingerprint="teacher-v1")
    store.put(source_index=7, prompt="original", reasoning_ids=(10, 11, 12))

    reopened = TeacherTraceStore(path, fingerprint="teacher-v1")
    assert reopened.get(source_index=7, prompt="original") == (10, 11, 12)
    with pytest.raises(ValueError, match="prompt hash"):
        reopened.get(source_index=7, prompt="changed")


def test_teacher_trace_store_rejects_teacher_fingerprint_drift(tmp_path):
    path = tmp_path / "teacher-traces.jsonl"
    TeacherTraceStore(path, fingerprint="teacher-v1").put(
        source_index=0,
        prompt="p",
        reasoning_ids=(1, 2),
    )

    with pytest.raises(ValueError, match="fingerprint"):
        TeacherTraceStore(path, fingerprint="teacher-v2")



class TrainingHFBackend:
    def apply_chat_template(self, messages, **kwargs):
        base = [1, 2]
        if not kwargs["add_generation_prompt"]:
            return base
        return base + ([8] if kwargs["enable_thinking"] else [3])

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [8, 9]


def test_distillation_trainer_reuses_fixed_trace_across_overfit_epochs(tmp_path, monkeypatch):
    source_model = _tiny_bgt()
    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    save_file(
        {name: tensor.detach().clone() for name, tensor in source_model.backbone.state_dict().items()},
        checkpoint / "model.safetensors",
    )
    data_path = tmp_path / "prompts.jsonl"
    data_path.write_text('{"prompt":"solve this"}\n', encoding="utf-8")
    output_dir = tmp_path / "run"

    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = TrainingHFBackend()
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    monkeypatch.setattr(distill_trainer, "get_tokenizer", lambda _: tokenizer)
    teacher_calls = []

    def fake_teacher(*args, **kwargs):
        teacher_calls.append((args, kwargs))
        return (8, 9)

    monkeypatch.setattr(distill_trainer, "sample_backbone_completion", fake_teacher)
    verification_saw_staging = []
    real_verify = distill_trainer._verify_overfit_checkpoint

    def verify_staging(checkpoint_path, *args, **kwargs):
        verification_saw_staging.append(checkpoint_path.name.startswith(".final-overfit-staging"))
        assert not (output_dir / "final").exists()
        return real_verify(checkpoint_path, *args, **kwargs)

    monkeypatch.setattr(distill_trainer, "_verify_overfit_checkpoint", verify_staging)
    config = TrainConfig(
        model=source_model.config,
        data=DataConfig(
            kind="openmath_prompts",
            data_dir=str(data_path),
            tokenizer="hf:fake",
            sft_take_rows=1,
        ),
        mode="cot_distillation",
        freeze_backbone=True,
        backbone_checkpoint=str(checkpoint),
        output_dir=str(output_dir),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        distillation=DistillationConfig(
            epochs=2,
            overfit_diagnostic=True,
            expected_locked_examples=1,
            max_teacher_reasoning_tokens=2,
            max_response_tokens=3,
            top_k=1,
            response_process_every=2,
            reasoning_close_text="close",
        ),
    )

    distill_trainer.train_distillation(config)

    summary = __import__("json").loads((output_dir / "summary.json").read_text())
    assert summary["epoch"] == 2
    assert summary["seen_prompts"] == 2
    assert summary["valid_prompts"] == 2
    assert summary["skipped_prompts"] == 0
    assert summary["optimizer_steps"] == 2
    assert (output_dir / "final" / "model.safetensors").is_file()
    assert verification_saw_staging == [True]
    metric_rows = [
        __import__("json").loads(line)
        for line in (output_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in metric_rows] == [1, 2]
    assert [row["teacher_trace_cached"] for row in metric_rows] == [True, True]
    assert all(isinstance(row["processor_gate_before_example"], float) for row in metric_rows)
    assert all(isinstance(row["processor_gate_after_example"], float) for row in metric_rows)
    assert all(
        len(row["cache_relative_l2_displacements"]) == len(row["cache_events"])
        for row in metric_rows
    )
    assert len(teacher_calls) == 1
    diagnostic = summary["overfit_diagnostic"]
    assert diagnostic["locked_examples"] == 1
    assert diagnostic["lock_attempts"] == 1
    assert diagnostic["lock_skips"] == []
    assert diagnostic["locked_source_indices"] == [0]
    assert diagnostic["initial"]["examples"] == 1
    assert diagnostic["pre_save"]["examples"] == 1
    assert diagnostic["post_load"]["examples"] == 1
    assert diagnostic["pre_save"] == diagnostic["post_load"]
    assert diagnostic["processor_changed_from_initialization"] is True
    assert diagnostic["initial_processor_sha256"] != diagnostic["pre_save_processor_sha256"]
    assert diagnostic["logit_probe_response_tokens"] == 3
    assert len(diagnostic["logit_probe_event_cache_lengths"]) == 2
    assert diagnostic["pre_save_logit_probe_sha256"] == diagnostic["post_load_logit_probe_sha256"]
    assert diagnostic["post_load_logits_verified"] is True
    loaded, loaded_config, loaded_state = load_checkpoint(output_dir / "final")
    assert isinstance(loaded, BGT)
    assert loaded_config == source_model.config
    assert loaded_state["valid_prompts"] == 2



def test_clipped_forward_kl_computes_bfloat16_logits_in_float32():
    student = torch.tensor([[[0.0, 1.0, -1.0]]], dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[[1.0, 0.0, -1.0]]], dtype=torch.bfloat16)
    loss = clipped_forward_kl(student, teacher, torch.ones((1, 1), dtype=torch.bool))

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)



def test_blockwise_replay_matches_tokenwise_reference_between_events():
    model = _tiny_bgt()
    session = BGTDecodeSession(model, process_every=64)
    prompt = ChatPrompt(base_ids=(1, 2, 3), trigger_ids=(4,))
    response = (5, 6, 7, 8, 9)

    next_logits, cache, events, displacements = session._start(prompt)
    assert displacements == []
    tokenwise = []
    for response_index, token_id in enumerate(response):
        tokenwise.append(next_logits)
        consumed = response_index + 1
        if consumed < len(response):
            next_logits, cache = session._consume(
                prompt,
                token_id,
                response_index,
                consumed,
                cache,
                events,
                will_continue=True,
            )

    blockwise = session.replay(prompt, response)
    torch.testing.assert_close(blockwise.logits, torch.stack(tokenwise, dim=1), atol=1e-6, rtol=1e-5)



def test_train_dispatches_distillation_before_distributed_initialisation(monkeypatch):
    from gcw.train import train

    calls = []
    monkeypatch.setattr(distill_trainer, "train_distillation", lambda config: calls.append(config))
    config = _distillation_train_config()

    train(config)

    assert calls == [config]



def test_sample_next_token_masks_padded_model_vocabulary():
    logits = torch.tensor([[0.0, 1.0, 100.0]])
    token = sample_next_token(
        logits,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        valid_vocab_size=2,
    )
    assert token.item() == 1



def test_native_teacher_incremental_answer_logits_match_full_forward(monkeypatch):
    backbone = _tiny_bgt().backbone
    context = (1, 2, 3, 4)
    response = (5, 6, 7)
    expected_full = backbone.forward_full(torch.tensor([context + response]))
    expected = expected_full[:, len(context) - 1 : len(context) - 1 + len(response)]
    monkeypatch.setattr(
        backbone,
        "forward_full",
        lambda _: (_ for _ in ()).throw(AssertionError("native teacher scoring must not project context logits")),
    )

    actual = teacher_answer_logits(backbone, context, response)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)



def test_chunked_kl_matches_unchunked_value_and_student_gradient():
    torch.manual_seed(3)
    base_student = torch.randn(2, 5, 11)
    teacher = torch.randn(2, 5, 11)
    mask = torch.tensor([[True, True, False, True, True], [True, False, True, True, False]])
    student_full = base_student.clone().requires_grad_(True)
    student_chunked = base_student.clone().requires_grad_(True)

    full = clipped_forward_kl(
        student_full,
        teacher,
        mask,
        temperature=1.1,
        token_clip=0.05,
        sequence_chunk_size=5,
    )
    chunked = clipped_forward_kl(
        student_chunked,
        teacher,
        mask,
        temperature=1.1,
        token_clip=0.05,
        sequence_chunk_size=2,
    )
    full.backward()
    chunked.backward()

    torch.testing.assert_close(chunked, full, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(student_chunked.grad, student_full.grad, atol=1e-7, rtol=1e-6)



@pytest.mark.parametrize(
    ("processor_overrides", "message"),
    [
        ({"attention_dropout": 0.1}, "zero processor attention dropout"),
        ({"processor_mode": "workspace_broadcast"}, "standard cache processor mode"),
    ],
)
def test_distillation_config_rejects_nondeterministic_or_nonstandard_processor(processor_overrides, message):
    processor = CacheProcessorConfig(
        process_every=64,
        topk=4,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        **processor_overrides,
    )
    model = ModelConfig(
        variant="bgt",
        family="qwen3",
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=processor,
    )
    with pytest.raises(ValueError, match=message):
        _distillation_train_config(model=model)


def test_openmath_take_zero_yields_no_records(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text('{"problem": "unused"}\n')
    assert list(iter_openmath_prompts(source, take_rows=0)) == []



def test_teacher_fingerprint_invalidates_seed_limit_and_chat_template(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"frozen-weights")
    tokenizer = object.__new__(TransformersTokenizer)
    tokenizer.enc = FakeHFTokenizerBackend()
    tokenizer.enc.chat_template = "template-v1"
    tokenizer.enc.name_or_path = "fake"
    tokenizer.enc.eos_token_id = 62
    tokenizer.enc.pad_token_id = None
    tokenizer.enc.bos_token_id = 1
    tokenizer.enc.special_tokens_map = {"eos_token": "<eos>"}
    tokenizer.enc.get_vocab = lambda: {"a": 0, "<eos>": 62}
    tokenizer.source = "fake"
    tokenizer.eot_id = 62
    tokenizer.vocab_size = 63
    base = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=8),
    )
    baseline = distill_trainer._teacher_fingerprint(base, (6, 7), (8, 9), tokenizer)
    changed_seed = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=8,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=8),
    )
    changed_limit = _distillation_train_config(
        backbone_checkpoint=str(checkpoint),
        seed=7,
        distillation=DistillationConfig(max_teacher_reasoning_tokens=9),
    )
    assert distill_trainer._teacher_fingerprint(changed_seed, (6, 7), (8, 9), tokenizer) != baseline
    assert distill_trainer._teacher_fingerprint(changed_limit, (6, 7), (8, 9), tokenizer) != baseline
    tokenizer.enc.chat_template = "template-v2"
    assert distill_trainer._teacher_fingerprint(base, (6, 7), (8, 9), tokenizer) != baseline



def test_bf16_backbone_fp32_processor_rollout_and_replay_under_autocast():
    model = _tiny_bgt()
    model.backbone.to(dtype=torch.bfloat16)
    assert next(model.cache_processor.parameters()).dtype == torch.float32
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(model, process_every=2, valid_vocab_size=model.config.vocab_size)
    generator = torch.Generator().manual_seed(4)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        rollout = session.rollout(
            prompt,
            max_response_tokens=3,
            eos_token_id=None,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            generator=generator,
        )
        replay = session.replay(prompt, rollout.response_ids)

    assert replay.logits.shape[:2] == (1, 3)
    replay.logits.float().sum().backward()
    assert any(parameter.grad is not None for parameter in model.cache_processor.parameters())



def test_mixed_final_checkpoint_verifies_exact_processor_round_trip(tmp_path):
    model = _tiny_bgt()
    model.backbone.to(dtype=torch.bfloat16)
    model.cache_processor.to(dtype=torch.float32)
    model.cache_processor.block.gate.data.fill_(-4.9896)
    destination = tmp_path / "final"
    expected_processor = {
        name: tensor.detach().clone()
        for name, tensor in model.cache_processor.state_dict().items()
    }

    published_state = distill_trainer._publish_checkpoint(
        destination,
        model,
        model.config,
        {
            "dtype": "mixed_bfloat16_backbone_float32_processor",
            "valid_prompts": 1,
        },
    )
    loaded, _, state = load_checkpoint(destination)

    assert published_state["checkpoint_fidelity_verified"] is True
    assert state == published_state
    assert state["pre_save_processor_sha256"] == state["post_load_processor_sha256"]
    assert state["pre_save_processor_gate"] == pytest.approx(-4.9896)
    assert state["post_load_processor_gate"] == pytest.approx(-4.9896)
    assert all(parameter.dtype == torch.bfloat16 for parameter in loaded.backbone.parameters())
    assert all(parameter.dtype == torch.float32 for parameter in loaded.cache_processor.parameters())
    for name, expected in expected_processor.items():
        torch.testing.assert_close(loaded.cache_processor.state_dict()[name], expected, rtol=0, atol=0)


def test_replay_records_relative_cache_displacement_when_enabled():
    model = _tiny_bgt()
    session = BGTDecodeSession(
        model,
        process_every=2,
        valid_vocab_size=model.config.vocab_size,
        record_displacements=True,
    )

    replay = session.replay(ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)), (4, 5, 6))

    assert len(replay.event_relative_l2_displacements) == len(replay.event_cache_lengths)
    assert all(torch.isfinite(torch.tensor(value)) and value >= 0 for value in replay.event_relative_l2_displacements)


def test_student_rollout_uses_static_cache(monkeypatch):
    model = _tiny_bgt()
    session = BGTDecodeSession(model, process_every=2, valid_vocab_size=model.config.vocab_size)
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    monkeypatch.setattr(
        model.backbone,
        "forward_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("rollout must use static cache")),
    )

    rollout = session.rollout(
        prompt,
        max_response_tokens=3,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(5),
    )

    assert len(rollout.response_ids) == 3
    assert len(rollout.event_cache_lengths) == 2



def test_native_teacher_sampling_uses_static_cache(monkeypatch):
    backbone = _tiny_bgt().backbone
    monkeypatch.setattr(
        backbone,
        "forward_chunk",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("teacher sampling must use static cache")),
    )
    generated = sample_backbone_completion(
        backbone,
        (1, 2, 3),
        max_new_tokens=3,
        eos_token_id=None,
        stop_ids=(),
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        valid_vocab_size=backbone.config.vocab_size,
        generator=torch.Generator().manual_seed(6),
    )
    assert len(generated) == 3



def test_decode_session_can_disable_processor_for_matched_eval_control():
    model = _tiny_bgt()
    recorder = RecordingZeroProcessor()
    model.cache_processor = recorder
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )

    rollout = session.rollout(
        ChatPrompt(base_ids=(1, 2), trigger_ids=(3,)),
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
    )

    assert len(rollout.response_ids) == 4
    assert rollout.event_cache_lengths == ()
    assert recorder.calls == []



def test_processor_disabled_session_matches_raw_backbone_greedy_generation():
    model = _tiny_bgt()
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )

    rollout = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        greedy=True,
    )
    raw = model.backbone.generate_fast(
        torch.tensor([prompt.full_ids], dtype=torch.long),
        max_new_tokens=4,
        eos_token_id=None,
    )

    assert rollout.response_ids == tuple(raw[0, len(prompt.full_ids) :].tolist())



def test_greedy_decode_session_stops_online_on_token_sequence():
    model = _tiny_bgt()
    prompt = ChatPrompt(base_ids=(1, 2), trigger_ids=(3,))
    session = BGTDecodeSession(
        model,
        process_every=2,
        select_all=False,
        valid_vocab_size=model.config.vocab_size,
        processor_enabled=False,
    )
    baseline = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        greedy=True,
    )
    stopped = session.rollout(
        prompt,
        max_response_tokens=4,
        eos_token_id=None,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        stop_sequences=[[baseline.response_ids[0]]],
        greedy=True,
    )

    assert stopped.response_ids == baseline.response_ids[:1]
    assert stopped.event_cache_lengths == ()
