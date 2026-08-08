from __future__ import annotations

import json

from gcw.checkpoint import save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    ModelConfig,
    ProcessBranchConfig,
    TrainConfig,
)
from gcw.models import build_model
from gcw.train import train


class _TinyTokenizer:
    eot_id = 0
    vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [ord(character) % 127 + 1 for character in text]

    def get_vocab_size(self) -> int:
        return self.vocab_size


def _model_config() -> ModelConfig:
    return ModelConfig(
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


def test_process_branch_trainer_updates_processor_and_writes_reloadable_checkpoint(
    monkeypatch,
    tmp_path,
) -> None:
    model_config = _model_config()
    backbone = tmp_path / "backbone"
    initial_model = build_model(model_config)
    initial_model.init_weights()
    save_checkpoint(backbone, initial_model, model_config)

    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": f"gsm8k-train:{index}",
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                    "reasoning_steps": [f"R{index}"],
                }
            )
            + "\n"
            for index in range(3)
        )
    )
    output = tmp_path / "run"
    config = TrainConfig(
        model=model_config,
        data=DataConfig(
            kind="process_branch_jsonl",
            data_dir=str(records),
            tokenizer="hf",
            tokenizer_source="fake-tokenizer",
            sft_take_rows=1,
            response_prefix="P",
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
            cycle_choices=(1,),
            gradient_accumulation_steps=1,
            max_optimizer_steps=1,
            max_prompt_tokens=8,
            max_step_tokens=8,
            max_answer_tokens=8,
            heldout_skip_rows=1,
            heldout_take_rows=2,
        ),
    )
    tokenizer_names = []
    monkeypatch.setattr(
        "gcw.process_branch_trainer.get_tokenizer",
        lambda name: tokenizer_names.append(name) or _TinyTokenizer(),
    )

    train(config)

    assert tokenizer_names == ["hf:fake-tokenizer"]

    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimizer_steps"] == 1
    assert summary["valid_records"] == 1
    assert summary["initial_processor_sha256"] != summary["final_processor_sha256"]
    assert summary["checkpoint_reload_verified"] is True
    assert summary["heldout_controls"]["examples"] == 2
    assert set(summary["heldout_controls"]) >= {
        "matched_answer_loss",
        "processor_off_answer_loss",
        "shuffled_answer_loss",
    }
    metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 1
    assert metrics[0]["train/cycles"] == 1
    assert (output / "final" / "model.safetensors").is_file()
