from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    DistillationConfig,
    ModelConfig,
    TrainConfig,
    to_dict,
)
from gcw.distill.data import PromptRecord
from gcw.distill import precompute
from gcw.distill.traces import TeacherTraceStore
from gcw.distill.trainer import _teacher_fingerprint


class FingerprintTokenizer:
    eot_id = 62

    @staticmethod
    def identity_sha256() -> str:
        return "1" * 64


class DynamicTokenizer:
    eot_id = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert tokenize and enable_thinking
        prompt = messages[0]["content"]
        prefix = [10] if prompt == "short" else [10, 11, 12]
        return prefix + ([8, 9] if add_generation_prompt else [])

    @staticmethod
    def encode(text: str):
        if text == "<think>\n":
            return [8, 9]
        if text == "</think>\n\n":
            return [6, 7]
        raise AssertionError(text)

    @staticmethod
    def get_vocab_size() -> int:
        return 64


def _config(checkpoint: Path, distill: DistillationConfig) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            variant="bgt",
            family="qwen3",
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            cache_processor=CacheProcessorConfig(
                hidden_size=16, intermediate_size=32, num_heads=2
            ),
        ),
        data=DataConfig(
            kind="openmath_prompts", tokenizer="hf:fake", data_dir="prompts.jsonl"
        ),
        mode="cot_distillation",
        freeze_backbone=True,
        backbone_checkpoint=str(checkpoint),
        batch_size=1,
        max_steps=1,
        distillation=distill,
    )


def _legacy_static_fingerprint(
    config: TrainConfig,
    open_ids: tuple[int, ...],
    close_ids: tuple[int, ...],
    tokenizer: FingerprintTokenizer,
) -> str:
    checkpoint = Path(config.backbone_checkpoint or "") / "model.safetensors"
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    model_payload = to_dict(config.model)
    model_payload.pop("cache_processor", None)
    distill = config.distillation
    assert distill is not None
    payload = {
        "schema": "gcw.teacher-trace.v6",
        "checkpoint_sha256": checkpoint_sha256,
        "model": model_payload,
        "tokenizer": config.data.tokenizer,
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "teacher_generation": {
            "seed": config.seed,
            "teacher_thinking": distill.teacher_thinking,
            "temperature": distill.temperature,
            "top_p": distill.top_p,
            "top_k": distill.top_k,
            "max_teacher_reasoning_tokens": distill.max_teacher_reasoning_tokens,
            "max_teacher_completion_tokens": distill.max_teacher_completion_tokens,
            "teacher_trajectory_mode": distill.teacher_trajectory_mode,
            "eos_token_id": tokenizer.eot_id,
            "reasoning_open_text": distill.reasoning_open_text,
            "open_ids": list(open_ids),
            "reasoning_close_text": distill.reasoning_close_text,
            "close_ids": list(close_ids),
            "close_reasoning_at_eos": distill.close_reasoning_at_eos,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_default_static_fingerprint_remains_v6_byte_compatible(tmp_path: Path) -> None:
    checkpoint = tmp_path / "gcw"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"gcw-weights")
    config = _config(checkpoint, DistillationConfig())
    tokenizer = FingerprintTokenizer()

    actual = _teacher_fingerprint(config, (8, 9), (6, 7), tokenizer)

    assert config.distillation is not None
    assert config.distillation.teacher_trace_backend == "gcw_static"
    assert actual == "96bf3e820db2d00a7ba2eaf3d84516a357d9f7def3856c1989f73bf1fd9506be"
    assert actual == _legacy_static_fingerprint(config, (8, 9), (6, 7), tokenizer)


def test_hf_dynamic_fingerprint_uses_new_schema_and_checkpoint_identity(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "gcw"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"gcw-weights")
    hf_checkpoint = tmp_path / "hf"
    hf_checkpoint.mkdir()
    (hf_checkpoint / "config.json").write_text('{"model_type":"qwen3"}\n')
    weights = hf_checkpoint / "model.safetensors"
    weights.write_bytes(b"hf-weights-v1")
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        close_reasoning_at_eos=False,
        require_precomputed_teacher_traces=True,
        teacher_trace_backend="hf_dynamic",
        teacher_trace_checkpoint=str(hf_checkpoint),
        teacher_trace_batch_size=4,
    )
    config = _config(checkpoint, distill)
    tokenizer = FingerprintTokenizer()
    first = _teacher_fingerprint(config, (8, 9), (6, 7), tokenizer)
    different_batch_config = _config(
        checkpoint,
        DistillationConfig(
            teacher_trajectory_mode="native_full",
            visible_trajectory_mode="both",
            close_reasoning_at_eos=False,
            require_precomputed_teacher_traces=True,
            teacher_trace_backend="hf_dynamic",
            teacher_trace_checkpoint=distill.teacher_trace_checkpoint,
            teacher_trace_batch_size=8,
        ),
    )
    different_batch = _teacher_fingerprint(
        different_batch_config, (8, 9), (6, 7), tokenizer
    )
    weights.write_bytes(b"hf-weights-v2")
    second = _teacher_fingerprint(config, (8, 9), (6, 7), tokenizer)

    assert first != second
    assert first != different_batch
    assert first != _legacy_static_fingerprint(config, (8, 9), (6, 7), tokenizer)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"teacher_trace_backend": "unknown"}, "teacher_trace_backend"),
        ({"teacher_trace_batch_size": 0}, "positive integer"),
        ({"teacher_trace_backend": "hf_dynamic"}, "teacher_trace_checkpoint"),
        ({"teacher_trace_checkpoint": "/tmp/hf"}, "only valid"),
    ],
)
def test_teacher_trace_backend_config_validation(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DistillationConfig(**kwargs)


def test_variable_length_fixed_chunks_and_left_padding_are_stable() -> None:
    prompts = ((1,), (2, 3, 4), (5, 6), (7, 8, 9, 10), (11,))

    first = precompute.fixed_batch_chunks(prompts, 2)
    second = precompute.fixed_batch_chunks(prompts, 2)
    input_ids, attention_mask, position_ids, lengths = precompute.left_pad_hf_prompts(
        first[0], pad_token_id=0, device=torch.device("cpu")
    )

    assert first == second == (((1,), (2, 3, 4)), ((5, 6), (7, 8, 9, 10)), ((11,),))
    assert input_ids.tolist() == [[0, 0, 1], [2, 3, 4]]
    assert attention_mask.tolist() == [[0, 0, 1], [1, 1, 1]]
    assert position_ids.tolist() == [[0, 0, 0], [0, 1, 2]]
    assert lengths.tolist() == [1, 3]


class FakeHFModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(max_position_embeddings=32)
        self.calls: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(kwargs)
        batch, sequence = kwargs["input_ids"].shape
        logits = torch.full((batch, sequence, 8), -100.0)
        token = (4, 5, 7)[len(self.calls) - 1]
        logits[:, -1, token] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=(len(self.calls),))


def test_hf_dynamic_sampling_is_repeat_exact_for_a_fixed_variable_length_plan() -> None:
    common = {
        "max_new_tokens": 3,
        "eos_token_id": 7,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 1,
        "valid_vocab_size": 8,
        "device": torch.device("cpu"),
    }
    prompts = ((1,), (2, 3, 4))
    first_model = FakeHFModel()
    first = precompute.sample_hf_dynamic_completions(
        first_model,
        prompts,
        generators=(torch.Generator().manual_seed(10), torch.Generator().manual_seed(12)),
        **common,
    )
    second = precompute.sample_hf_dynamic_completions(
        FakeHFModel(),
        prompts,
        generators=(torch.Generator().manual_seed(10), torch.Generator().manual_seed(12)),
        **common,
    )

    assert first == second == ((4, 5, 7), (4, 5, 7))
    assert first_model.calls[0]["input_ids"].tolist() == [[7, 7, 1], [2, 3, 4]]
    assert first_model.calls[1]["position_ids"].tolist() == [[1], [3]]
    assert first_model.calls[1]["cache_position"].tolist() == [3]


def test_hf_dynamic_precompute_uses_mixed_length_chunks_and_native_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = DynamicTokenizer()
    records = (
        PromptRecord(source_index=2, prompt="short"),
        PromptRecord(source_index=5, prompt="a longer prompt"),
    )
    store = TeacherTraceStore(tmp_path / "traces.jsonl", fingerprint="hf-v1")
    observed: list[tuple[tuple[int, ...], ...]] = []

    def fake_sample(model, prompts, **kwargs):
        del model
        observed.append(tuple(prompts))
        assert [generator.initial_seed() for generator in kwargs["generators"]] == [28, 34]
        return ((40, 6, 7, 50, 0), (41, 6, 7, 51, 0))

    monkeypatch.setattr(precompute, "sample_hf_dynamic_completions", fake_sample)
    distill = DistillationConfig(
        teacher_trajectory_mode="native_full",
        visible_trajectory_mode="both",
        max_teacher_reasoning_tokens=3,
        max_teacher_completion_tokens=5,
        close_reasoning_at_eos=False,
        require_precomputed_teacher_traces=True,
        teacher_trace_backend="hf_dynamic",
        teacher_trace_checkpoint="/unused/in-unit-test",
        teacher_trace_batch_size=2,
    )

    report = precompute.precompute_hf_dynamic_native_teacher_records(
        object(),
        tokenizer,  # type: ignore[arg-type]
        records,
        distill,
        store,
        batch_size=2,
        seed=24,
        device=torch.device("cpu"),
    )

    assert [list(map(len, chunk)) for chunk in observed] == [[3, 5]]
    assert report["valid"] == report["generated"] == 2
    assert store.get_native_trajectory(source_index=2, prompt="short") is not None
    assert store.get_native_trajectory(source_index=5, prompt="a longer prompt") is not None
