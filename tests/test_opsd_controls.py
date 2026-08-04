from __future__ import annotations

import hashlib
import json

import pytest
import torch

from gcw.checkpoint import load_checkpoint, save_checkpoint
from gcw.config import CacheProcessorConfig, ModelConfig, to_dict
from gcw.models import build_model
from gcw.models.bgt import BGT
from gcw.opsd_eval import aggregate_opsd_three_way
from gcw import opsd_controls
from gcw.state_hash import tensor_state_sha256


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        variant="bgt",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        cache_processor=CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
        ),
    )


def _checkpoint_manifest(path):
    files = {}
    for name in ("model.safetensors", "config.json", "train_state.json"):
        payload = (path / name).read_bytes()
        files[name] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": files,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _eval_document(path, config, state, *, processor_enabled: bool, score: int):
    checkpoint_manifest = _checkpoint_manifest(path)
    return {
        "source_revision": "a" * 40,
        "checkpoint_sha256": checkpoint_manifest["files"]["model.safetensors"][
            "sha256"
        ],
        "checkpoint_manifest": checkpoint_manifest,
        "tokenizer_source": "/tokenizer",
        "tokenizer_identity_sha256": "b" * 64,
        "datasets_cache": "/cache",
        "tasks": ["gsm8k_openmath"],
        "task_config_sha256": {"gsm8k_openmath": "c" * 64},
        "dataset_files": {"gsm8k_openmath": {"sha256": "d" * 64}},
        "limit_per_task": 1,
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "fewshot": 0,
            "apply_chat_template_in_lm_eval": False,
        },
        "model_config": to_dict(config),
        "train_state": state,
        "condition": {
            "name": "cot-off__cp-on" if processor_enabled else "cot-off__cp-off",
            "enable_thinking": False,
            "processor_enabled": processor_enabled,
        },
        "sample_manifest": [
            {
                "task": "gsm8k_openmath",
                "doc_id": 0,
                "doc_hash": "1" * 64,
                "prompt_hash": "2" * 64,
                "target_hash": "3" * 64,
            }
        ],
        "generation_records": [
            {
                "prompt_sha256": "4" * 64,
                "prompt_text_sha256": "4" * 64,
                "normalized_prompt_text_sha256": "7" * 64,
                "base_token_ids_sha256": "5" * 64,
                "full_prompt_token_ids_sha256": "6" * 64,
                "base_tokens": 5,
                "trigger_tokens": 2,
                "response_tokens": 5,
                "max_response_tokens": 10,
                "raw_response_tokens": 5,
                "cache_event_positions": [0] if processor_enabled else [],
                "cache_event_lengths": [5] if processor_enabled else [],
            }
        ],
        "lm_eval": {
            "results": {
                "gsm8k_openmath": {"exact_match,flexible": float(score)}
            },
            "samples": {
                "gsm8k_openmath": [
                    {
                        "doc_id": 0,
                        "filter": "flexible",
                        "exact_match": score,
                        "resps": [[str(score)]],
                    }
                ]
            },
        },
    }


def test_derive_untrained_control_resets_only_fp32_processor_atomically(
    tmp_path, monkeypatch
):
    seed = 24
    config = _tiny_config()
    torch.manual_seed(seed)
    trained = build_model(config)
    assert isinstance(trained, BGT)
    initial_processor_sha256 = tensor_state_sha256(
        dict(trained.cache_processor.state_dict())
    )
    initial_gate = float(trained.cache_processor.block.gate.item())
    trained.backbone.to(dtype=torch.bfloat16)
    trained.cache_processor.to(dtype=torch.float32)
    trained.cache_processor.block.gate.data.add_(0.25)
    trained_processor_sha256 = tensor_state_sha256(
        dict(trained.cache_processor.state_dict())
    )
    run_config = {
        "seed": seed,
        "output_dir": str(tmp_path),
        "data": {"tokenizer": "/tokenizer"},
    }
    run_config_path = tmp_path / "run_config.json"
    run_config_path.write_text(json.dumps(run_config))
    run_config_sha256 = hashlib.sha256(run_config_path.read_bytes()).hexdigest()
    source = tmp_path / "trained"
    save_checkpoint(
        source,
        trained,
        config,
        {
            "dtype": opsd_controls.MIXED_DTYPE,
            "optimizer_steps": 3,
            "checkpoint_fidelity_verified": True,
            "source_revision": "a" * 40,
            "source_clean": True,
            "training_seed": seed,
            "processor_initialization_recipe": (
                "torch.manual_seed(training_seed); build_model(model_config); v1"
            ),
            "torch_version": torch.__version__,
            "tokenizer_source": "/tokenizer",
            "tokenizer_identity_sha256": "b" * 64,
            "run_config_sha256": run_config_sha256,
            "initial_processor_sha256": initial_processor_sha256,
            "initial_processor_gate": initial_gate,
            "post_load_processor_sha256": trained_processor_sha256,
        },
    )
    monkeypatch.setattr(opsd_controls, "_verify_revision", lambda *_: None)
    destination = tmp_path / "untrained"

    report = opsd_controls.derive_untrained_processor_control(
        source,
        destination,
        seed=seed,
        source_revision="a" * 40,
        tokenizer_source="/tokenizer",
        repository=tmp_path,
    )

    control, loaded_config, state = load_checkpoint(destination)
    source_loaded, _, source_loaded_state = load_checkpoint(source)
    torch.manual_seed(seed)
    fresh = build_model(config)
    assert isinstance(control, BGT)
    assert isinstance(source_loaded, BGT)
    assert isinstance(fresh, BGT)
    assert loaded_config == config
    assert state["kind"] == "derived_untrained_fp32_cache_processor_control"
    assert state["processor_storage_dtype"] == "float32"
    assert report["processor_values_different_from_trained"] > 0
    assert report["processor_values_total"] > 0
    assert report["initial_gate_float32"] == pytest.approx(-5.0)
    assert report["trained_checkpoint_gate_float32"] == pytest.approx(-4.75)
    assert {parameter.dtype for parameter in control.backbone.parameters()} == {torch.bfloat16}
    assert {parameter.dtype for parameter in control.cache_processor.parameters()} == {
        torch.float32
    }
    for name, tensor in fresh.cache_processor.state_dict().items():
        assert torch.equal(control.cache_processor.state_dict()[name], tensor)
    for name, tensor in source_loaded.backbone.state_dict().items():
        assert torch.equal(control.backbone.state_dict()[name], tensor)

    aggregate = aggregate_opsd_three_way(
        {
            "off": _eval_document(
                source,
                config,
                source_loaded_state,
                processor_enabled=False,
                score=0,
            ),
            "untrained": _eval_document(
                destination,
                config,
                state,
                processor_enabled=True,
                score=0,
            ),
            "trained": _eval_document(
                source,
                config,
                source_loaded_state,
                processor_enabled=True,
                score=1,
            ),
        },
        bootstrap_seed=24,
        bootstrap_samples=10,
        expected_tasks=("gsm8k_openmath",),
        expected_limit=1,
    )
    assert aggregate["paired_contrasts"]["trained_minus_untrained"][
        "three_task_macro_delta"
    ] == 1.0
    assert state["control_backbone_sha256"] == state["source_backbone_sha256"]
    assert set(state["checkpoint_dtype_counts"]) == {
        "torch.bfloat16",
        "torch.float32",
    }
    assert not list(tmp_path.glob(".untrained-*"))

    with pytest.raises(ValueError, match="source pilot provenance"):
        opsd_controls.derive_untrained_processor_control(
            source,
            tmp_path / "wrong-seed",
            seed=seed + 1,
            source_revision="a" * 40,
            tokenizer_source="/tokenizer",
            repository=tmp_path,
        )
    with pytest.raises(ValueError, match="source pilot provenance"):
        opsd_controls.derive_untrained_processor_control(
            source,
            tmp_path / "wrong-tokenizer",
            seed=seed,
            source_revision="a" * 40,
            tokenizer_source="/other-tokenizer",
            repository=tmp_path,
        )
    with pytest.raises(FileExistsError, match="refusing existing control"):
        opsd_controls.derive_untrained_processor_control(
            source,
            destination,
            seed=seed,
            source_revision="a" * 40,
            tokenizer_source="/tokenizer",
            repository=tmp_path,
        )
