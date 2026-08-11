from __future__ import annotations

import hashlib
import json

import pytest
import torch

from gcw.checkpoint import save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    ProcessBranchConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.models.longshot import Longshot
from gcw.process_branch import InvalidProcessBranchExample
from gcw.process_data import ProcessBranchRecord, ProcessDataError
from gcw.state_hash import tensor_state_sha256
from gcw.train import train
from gcw.trajectory_distillation import (
    _masked_trajectory_targets,
    authenticate_trajectory_artifact,
    tokenize_trajectory_record,
    trajectory_replay_example_loss,
)
from scripts.prepare_gsm8k_trajectory_replay import prepare_trajectory_records


class _ChatTokenizer:
    eot_id = 0
    vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 119 + 8 for character in text]

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize
        content = messages[0]["content"]
        base = [1, *self.encode(content), 2]
        if not add_generation_prompt:
            return base
        return base + ([3, 4] if enable_thinking else [5, 6])


def _record() -> ProcessBranchRecord:
    return ProcessBranchRecord(
        source_id="gsm8k-train:0",
        prompt="Q?",
        reasoning_steps=("First add.", "Then divide."),
        final_answer="\\boxed{7}",
    )


def _tiny_longshot() -> Longshot:
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=96,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=0,
            topk_heads=4,
            score="kv_norm",
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_heads=2,
            num_kv_heads=2,
            gate_init=0.0,
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
        ),
    )
    model = build_model(config)
    assert isinstance(model, Longshot)
    model.init_weights()
    model.backbone.requires_grad_(False)
    model.backbone.eval()
    model.cache_processor.train()
    return model


def test_trajectory_tokenization_aligns_exact_thinking_modes() -> None:
    tokenizer = _ChatTokenizer()
    example = tokenize_trajectory_record(
        _record(),
        tokenizer,
        prompt_suffix="Show reliable arithmetic.",
        rationale_joiner="\n",
        reasoning_close_text="\n</think>\n\n",
        max_prompt_tokens=64,
        max_trajectory_tokens=64,
        max_answer_tokens=32,
    )

    expected_prompt = "Q?\n\nShow reliable arithmetic."
    assert example.prompt_ids.tolist() == [1, *tokenizer.encode(expected_prompt), 2]
    assert example.step_response_prefix_ids.tolist() == [3, 4]
    assert example.answer_response_prefix_ids.tolist() == [5, 6]
    assert len(example.reasoning_step_ids) == 1
    assert example.reasoning_step_ids[0].tolist() == tokenizer.encode(
        "First add.\nThen divide."
    ) + tokenizer.encode("\n</think>\n\n")
    assert example.final_answer_ids.tolist() == tokenizer.encode("\\boxed{7}") + [0]


def test_trajectory_tokenization_rejects_overlong_full_rationale() -> None:
    with pytest.raises(InvalidProcessBranchExample, match="trajectory_too_long"):
        tokenize_trajectory_record(
            _record(),
            _ChatTokenizer(),
            prompt_suffix="",
            rationale_joiner="\n",
            reasoning_close_text="X",
            max_prompt_tokens=64,
            max_trajectory_tokens=2,
            max_answer_tokens=32,
        )


def test_masked_targets_split_rationale_and_answer_without_overlap() -> None:
    rationale, answer = _masked_trajectory_targets(
        prefix_length=3,
        rationale_length=4,
        answer_length=2,
        device="cpu",
    )
    assert rationale.tolist() == [[-1, -1, 0, 0, 0, 0, -1, -1]]
    assert answer.tolist() == [[-1, -1, -1, -1, -1, -1, 0, 0]]
    assert not bool(torch.any((rationale >= 0) & (answer >= 0)))


def test_replay_loss_updates_only_processor_and_balances_both_answer_modes() -> None:
    torch.manual_seed(24)
    model = _tiny_longshot()
    example = tokenize_trajectory_record(
        _record(),
        _ChatTokenizer(),
        prompt_suffix="",
        rationale_joiner=" ",
        reasoning_close_text=" X ",
        max_prompt_tokens=32,
        max_trajectory_tokens=48,
        max_answer_tokens=16,
    )

    losses = trajectory_replay_example_loss(
        model,
        example,
        cycles=2,
        rationale_weight=0.5,
        answer_weight=1.25,
    )
    assert losses.cycles == 2
    assert bool(torch.isfinite(losses.total_loss))
    assert torch.allclose(
        losses.answer_loss,
        0.5 * (losses.direct_answer_loss + losses.thinking_answer_loss),
    )
    assert torch.allclose(
        losses.total_loss,
        0.5 * losses.rationale_loss + 1.25 * losses.answer_loss,
    )

    losses.total_loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    processor_grads = [
        parameter.grad
        for parameter in model.cache_processor.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert processor_grads
    assert any(bool(torch.count_nonzero(gradient)) for gradient in processor_grads)


def test_trajectory_authentication_binds_bytes_and_ordered_ids() -> None:
    payloads = [
        {
            "source_id": "gsm8k-train:0",
            "prompt": "Q",
            "reasoning_steps": ["R"],
            "final_answer": "\\boxed{1}",
        },
        {
            "source_id": "gsm8k-train:1",
            "prompt": "Q2",
            "reasoning_steps": ["R2"],
            "final_answer": "\\boxed{2}",
        },
    ]
    source = "".join(json.dumps(row, sort_keys=True) + "\n" for row in payloads).encode()
    ids = json.dumps(
        [row["source_id"] for row in payloads],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    records = authenticate_trajectory_artifact(
        source,
        expected_artifact_sha256=hashlib.sha256(source).hexdigest(),
        expected_source_ids_sha256=hashlib.sha256(ids).hexdigest(),
    )
    assert [record.source_id for record in records] == ["gsm8k-train:0", "gsm8k-train:1"]
    with pytest.raises(ProcessDataError, match="artifact sha256"):
        authenticate_trajectory_artifact(
            source + b"\n",
            expected_artifact_sha256=hashlib.sha256(source).hexdigest(),
            expected_source_ids_sha256=hashlib.sha256(ids).hexdigest(),
        )


def test_prepare_trajectory_records_keeps_all_reasoning_but_removes_calculator_tags(
    tmp_path,
) -> None:
    output = tmp_path / "trajectory.jsonl"
    audit = prepare_trajectory_records(
        [
            {
                "question": "What is 2+3?",
                "answer": "Compute 2+3 = <<2+3=5>>5.\nSo the result is 5.\n#### 5",
            }
        ],
        output,
    )
    payload = json.loads(output.read_text())

    assert audit["source_rows"] == audit["accepted_rows"] == 1
    assert audit["rejected_rows"] == 0
    assert payload == {
        "source_id": "gsm8k-train:0",
        "prompt": "What is 2+3?",
        "reasoning_steps": ["Compute 2+3 = 5.", "So the result is 5."],
        "final_answer": "\\boxed{5}",
    }
    assert "<<" not in output.read_text()


def test_trajectory_replay_config_validates_objective_specific_text() -> None:
    process = ProcessBranchConfig(
        objective="trajectory_replay",
        process_weight=0.5,
        answer_weight=1.0,
        rationale_joiner="\n",
        reasoning_close_text="\n</think>\n\n",
    )
    assert process.objective == "trajectory_replay"
    with pytest.raises(ValueError, match="reasoning_close_text"):
        ProcessBranchConfig(objective="trajectory_replay", reasoning_close_text="")


def test_one_step_trainer_route_updates_only_cache_processor(monkeypatch, tmp_path) -> None:
    model = _tiny_longshot()
    backbone_hash = tensor_state_sha256(model.backbone.state_dict())
    backbone = tmp_path / "backbone"
    save_checkpoint(backbone, model, model.config)
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}?",
                    "reasoning_steps": [f"Compute R{index}."],
                    "final_answer": f"\\\\boxed{{{index}}}",
                },
                sort_keys=True,
            )
            + "\n"
            for index in range(3)
        )
    )
    output = tmp_path / "run"
    config = TrainConfig(
        model=model.config,
        data=DataConfig(
            kind="process_branch_jsonl",
            data_dir=str(records_path),
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            sft_take_rows=1,
            prompt_suffix="",
            response_prefix="unused-but-required",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint=str(backbone),
        output_dir=str(output),
        batch_size=1,
        max_steps=1,
        learning_rate=1e-3,
        warmup_steps=0,
        weight_decay=0.0,
        process_branch=ProcessBranchConfig(
            objective="trajectory_replay",
            cycle_choices=(1,),
            process_weight=0.5,
            answer_weight=1.0,
            rationale_joiner=" ",
            reasoning_close_text=" X ",
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_prompt_tokens=32,
            max_step_tokens=32,
            max_answer_tokens=16,
            heldout_skip_rows=1,
            heldout_take_rows=2,
        ),
    )
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer", lambda _name: _ChatTokenizer()
    )

    train(config)

    summary = json.loads((output / "summary.json").read_text())
    metric = json.loads((output / "metrics.jsonl").read_text())
    assert summary["objective"] == "trajectory_replay"
    assert summary["optimizer_steps"] == 1
    assert summary["initial_processor_sha256"] != summary["final_processor_sha256"]
    assert summary["initial_backbone_sha256"] == backbone_hash
    assert summary["final_backbone_sha256"] == backbone_hash
    assert metric["train/process_loss"] > 0.0
    assert metric["train/answer_loss"] > 0.0
    assert metric["train/answer_loss"] == pytest.approx(
        0.5
        * (
            metric["train/direct_answer_loss"]
            + metric["train/thinking_answer_loss"]
        )
    )
