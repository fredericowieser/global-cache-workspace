from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from gcw.config import ModelConfig, load_train_config
from gcw.params import estimate_non_embedding_params, processor_params
from scripts import generate_math_experiment_configs as generator


def test_experiment_configs_share_data_steps_rows_and_wandb(tmp_path, monkeypatch):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("{}\n")
    data_manifest = tmp_path / "data.json"
    data_manifest.write_text(
        json.dumps(
            {
                "rows": 129,
                "dataset_path": str(dataset.resolve()),
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "validation_path": str(dataset.resolve()),
                "source_ids_sha256": "12" * 32,
                "tokenizer_identity_sha256": "23" * 32,
            }
        )
    )
    checkpoint = tmp_path / "backbone"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        generator,
        "config_from_hf",
        lambda *_args, **_kwargs: ModelConfig(
            family="qwen3",
            sequence_len=8192,
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            qk_norm=True,
            rotary_style="llama",
        ),
    )
    args = Namespace(
        out_dir=str(tmp_path / "configs"),
        model="Qwen/Qwen3-0.6B",
        backbone_checkpoint=str(checkpoint),
        data_manifest=str(data_manifest),
        run_root=str(tmp_path / "runs"),
        sequence_length=4096,
        global_batch_rows=64,
        processor_width=128,
        micro_batch=1,
        distributed="none",
        seed=24,
        experiment="experiment",
    )

    paths = generator.generate(args)
    configs = [load_train_config(path) for path in paths]

    assert len(configs) == 11
    assert {config.sequence_len for config in configs} == {4096}
    assert {config.global_batch_rows for config in configs} == {64}
    assert {config.max_steps for config in configs} == {3}
    assert {config.data.dataset for config in configs} == {str(dataset.resolve())}
    assert {config.wandb.entity for config in configs} == {"huawei-noahs-ark"}
    assert {config.wandb.project for config in configs} == {"global-cache-workspace"}
    for config in configs:
        if config.model.variant != "tf" and estimate_non_embedding_params(config.model) > 1_000_000:
            assert processor_params(config.model) <= 0.05 * estimate_non_embedding_params(config.model)
