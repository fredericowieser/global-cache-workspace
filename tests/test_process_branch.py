from __future__ import annotations

import pytest
import torch

from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    ProcessBranchConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.models.longshot import Longshot
from gcw.process_branch import (
    InvalidProcessBranchExample,
    ProcessBranchTokenExample,
    branch_answer_loss,
    branch_next_token_loss,
    build_process_branch_state,
    generate_process_branch,
    process_branch_example_loss,
    tokenize_process_record,
)
from gcw.process_data import ProcessBranchRecord


def _tiny_longshot() -> Longshot:
    config = ModelConfig(
        variant="longshot",
        family="smollm",
        sequence_len=32,
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
            encoder_method="query_attention",
            encoder_query_source="final_token",
            update_stream="v",
            gate_init=0.0,
        ),
    )
    model = build_model(config)
    assert isinstance(model, Longshot)
    model.init_weights()
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    return model


def test_branch_loss_backpropagates_through_frozen_backbone_to_workspace() -> None:
    torch.manual_seed(11)
    model = _tiny_longshot()
    prompt = torch.tensor([[4, 5, 6, 7]])
    prefix = torch.tensor([[8, 9]])
    target = torch.tensor([[10, 11, 12]])

    with torch.no_grad():
        _, prompt_cache = model.backbone.forward_chunk(
            prompt,
            0,
            None,
            logits=False,
            cache_format="windowed",
        )
    source = model.cache_processor.encode_source(prompt_cache, select_all=True)
    workspace = model.cache_processor.deliberate(
        model.cache_processor.initialise_workspace(source),
        source,
    )
    branch_cache = model.cache_processor.write_for_readout(prompt_cache, source, workspace)

    loss = branch_next_token_loss(
        model.backbone,
        branch_cache,
        start_pos=prompt.shape[1],
        response_prefix_ids=prefix,
        target_ids=target,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.cache_processor.parameters()
    )


def test_one_cycle_objective_reports_process_and_answer_losses() -> None:
    torch.manual_seed(13)
    model = _tiny_longshot()
    example = ProcessBranchTokenExample(
        prompt_ids=torch.tensor([4, 5, 6, 7]),
        step_response_prefix_ids=torch.tensor([8, 9]),
        answer_response_prefix_ids=torch.tensor([8]),
        reasoning_step_ids=(torch.tensor([10, 11]),),
        final_answer_ids=torch.tensor([12, 13]),
    )

    losses = process_branch_example_loss(
        model,
        example,
        process_weight=0.75,
        answer_weight=1.25,
    )

    assert losses.cycles == 1
    assert losses.process_loss.ndim == 0
    assert losses.answer_loss.ndim == 0
    assert torch.allclose(
        losses.total_loss,
        0.75 * losses.process_loss + 1.25 * losses.answer_loss,
    )
    losses.total_loss.backward()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.cache_processor.parameters()
    )


def test_process_branch_objective_rejects_unfrozen_backbone() -> None:
    model = _tiny_longshot()
    model.backbone.requires_grad_(True)
    example = ProcessBranchTokenExample(
        prompt_ids=torch.tensor([4, 5, 6, 7]),
        step_response_prefix_ids=torch.tensor([8]),
        answer_response_prefix_ids=torch.tensor([9]),
        reasoning_step_ids=(torch.tensor([10]),),
        final_answer_ids=torch.tensor([11]),
    )

    with pytest.raises(ValueError, match="frozen backbone"):
        process_branch_example_loss(
            model,
            example,
            process_weight=1.0,
            answer_weight=1.0,
        )


def test_answer_control_can_swap_workspace_without_changing_prompt_source() -> None:
    torch.manual_seed(17)
    model = _tiny_longshot()
    first = ProcessBranchTokenExample(
        prompt_ids=torch.tensor([4, 5, 6, 7]),
        step_response_prefix_ids=torch.tensor([8]),
        answer_response_prefix_ids=torch.tensor([8]),
        reasoning_step_ids=(torch.tensor([9]),),
        final_answer_ids=torch.tensor([10, 0]),
    )
    second = ProcessBranchTokenExample(
        prompt_ids=torch.tensor([11, 12, 13, 14]),
        step_response_prefix_ids=torch.tensor([8]),
        answer_response_prefix_ids=torch.tensor([8]),
        reasoning_step_ids=(torch.tensor([15]),),
        final_answer_ids=torch.tensor([16, 0]),
    )

    first_state = build_process_branch_state(model, first, cycles=1)
    second_state = build_process_branch_state(model, second, cycles=1)
    source_values = first_state.prompt_cache.values.clone()
    matched = branch_answer_loss(model, first, first_state)
    shuffled = branch_answer_loss(
        model,
        first,
        first_state,
        workspace_override=second_state.workspace,
    )

    assert torch.isfinite(matched)
    assert torch.isfinite(shuffled)
    assert not torch.equal(matched, shuffled)
    assert torch.equal(first_state.prompt_cache.values, source_values)


def test_process_branch_generation_decodes_from_answer_prefix() -> None:
    torch.manual_seed(19)
    model = _tiny_longshot()

    generated = generate_process_branch(
        model,
        prompt_ids=torch.tensor([4, 5, 6, 7]),
        answer_prefix_ids=torch.tensor([8, 9]),
        cycles=1,
        max_new_tokens=3,
        eos_token_id=None,
        valid_vocab_size=64,
    )

    assert generated.shape == (3,)
    assert generated.dtype == torch.long
    assert generated.min().item() >= 0
    assert generated.max().item() < 64


def _process_train_config(model_config: ModelConfig) -> TrainConfig:
    return TrainConfig(
        model=model_config,
        data=DataConfig(
            kind="process_branch_jsonl",
            data_dir="records.jsonl",
            tokenizer="hf:tokenizer",
            response_prefix="\n\n",
        ),
        mode="process_branch",
        distributed="none",
        freeze_backbone=True,
        backbone_checkpoint="backbone",
        batch_size=1,
        max_steps=1,
        process_branch=ProcessBranchConfig(
            cycle_choices=(1,),
            gradient_accumulation_steps=2,
            max_optimizer_steps=3,
            heldout_skip_rows=4,
            heldout_take_rows=2,
        ),
    )


def test_process_branch_config_accepts_frozen_value_only_longshot() -> None:
    config = _process_train_config(_tiny_longshot().config)

    assert config.mode == "process_branch"
    assert config.process_branch is not None
    assert config.process_branch.cycle_choices == (1,)


def test_process_branch_config_rejects_key_value_writes() -> None:
    model_config = _tiny_longshot().config
    assert model_config.cache_processor is not None
    model_config.cache_processor.update_stream = "kv"

    with pytest.raises(ValueError, match="value-only"):
        _process_train_config(model_config)


class _TinyTextTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 127 + 1 for character in text]


def test_process_record_tokenization_is_bounded_and_eot_terminated() -> None:
    record = ProcessBranchRecord(
        source_id="gsm8k-train:9",
        prompt="Q",
        final_answer="A",
        reasoning_steps=("first", "second"),
    )

    example = tokenize_process_record(
        record,
        _TinyTextTokenizer(),
        step_response_prefix="T",
        answer_response_prefix="P",
        prompt_suffix="S",
        cycle_count=1,
        max_prompt_tokens=8,
        max_step_tokens=8,
        max_answer_tokens=4,
    )

    assert example.prompt_ids.tolist() == _TinyTextTokenizer().encode("Q\n\nS")
    assert example.step_response_prefix_ids.tolist() == _TinyTextTokenizer().encode("T")
    assert example.answer_response_prefix_ids.tolist() == _TinyTextTokenizer().encode("P")
    assert len(example.reasoning_step_ids) == 1
    assert example.reasoning_step_ids[0][-1].item() == 0
    assert example.final_answer_ids[-1].item() == 0

    with pytest.raises(InvalidProcessBranchExample, match="prompt_too_long"):
        tokenize_process_record(
            record,
            _TinyTextTokenizer(),
            step_response_prefix="T",
            answer_response_prefix="P",
            prompt_suffix="S",
            cycle_count=1,
            max_prompt_tokens=2,
            max_step_tokens=8,
            max_answer_tokens=4,
        )
