from __future__ import annotations

import copy

import pytest

from gcw.cli import _atomic_write_new
from gcw.eval import _opsd_sample_manifest
from gcw.opsd_eval import OPSD_SCREEN_CONDITIONS, aggregate_opsd_screen


def _document(name: str, score: float) -> dict:
    processor_enabled = name.endswith("cp-on")
    return {
        "source_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "checkpoint_manifest": {
            "files": {
                "model.safetensors": {"size_bytes": 1, "sha256": "b" * 64},
                "config.json": {"size_bytes": 2, "sha256": "2" * 64},
                "train_state.json": {"size_bytes": 3, "sha256": "3" * 64},
            },
            "manifest_sha256": "4" * 64,
        },
        "tokenizer_source": "/models/qwen3-sft",
        "tokenizer_identity_sha256": "f" * 64,
        "datasets_cache": "/cache",
        "tasks": ["gsm8k_openmath"],
        "task_config_sha256": {"gsm8k_openmath": "c" * 64},
        "dataset_files": {
            "gsm8k_openmath": {
                "path": "/cache/gsm8k-test.arrow",
                "size_bytes": 123,
                "sha256": "1" * 64,
            }
        },
        "limit_per_task": 1,
        "condition": {
            "name": name,
            "enable_thinking": name.startswith("cot-on"),
            "processor_enabled": processor_enabled,
        },
        "sample_manifest": [
            {
                "task": "gsm8k_openmath",
                "doc_id": 0,
                "doc_hash": "5" * 64,
                "prompt_hash": "6" * 64,
                "target_hash": "7" * 64,
            }
        ],
        "generation_records": [
            {
                "prompt_sha256": "d" * 64,
                "base_token_ids_sha256": "8" * 64,
                "full_prompt_token_ids_sha256": (
                    "9" * 64 if name.startswith("cot-off") else "a" * 64
                ),
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
                "gsm8k_openmath": {
                    "exact_match,flexible": score,
                    "exact_match_stderr,flexible": 0.01,
                    "sample_len": 1,
                    "alias": "gsm8k_openmath",
                }
            }
        },
    }


def test_opsd_screen_aggregate_computes_causal_contrasts():
    documents = [
        _document("cot-off__cp-off", 0.1),
        _document("cot-off__cp-on", 0.2),
        _document("cot-on__cp-off", 0.3),
        _document("cot-on__cp-on", 0.5),
    ]

    result = aggregate_opsd_screen(documents)

    assert result["conditions"] == list(OPSD_SCREEN_CONDITIONS)
    assert result["prompt_count"] == 1
    assert result["metrics"]["cot-on__cp-on"]["gsm8k_openmath"] == {
        "exact_match,flexible": 0.5
    }
    assert result["contrasts"]["cp_effect_cot_off"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.1)
    assert result["contrasts"]["cp_effect_cot_on"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.2)
    assert result["contrasts"]["interaction"]["gsm8k_openmath"]["exact_match,flexible"] == pytest.approx(0.1)


def test_opsd_screen_aggregate_rejects_prompt_manifest_drift():
    documents = [_document(name, 0.1) for name in OPSD_SCREEN_CONDITIONS]
    documents = copy.deepcopy(documents)
    documents[-1]["generation_records"][0]["prompt_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="prompt manifest"):
        aggregate_opsd_screen(documents)



def test_opsd_screen_publication_is_atomic_and_refuses_overwrite(tmp_path):
    output = tmp_path / "condition.json"
    _atomic_write_new(output, '{"ok": true}')

    assert output.read_text() == '{"ok": true}'
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_new(output, '{"ok": false}')



def test_opsd_sample_manifest_deduplicates_lm_eval_filter_rows():
    identity = {
        "doc_id": 7,
        "doc_hash": "a",
        "prompt_hash": "b",
        "target_hash": "c",
    }
    evaluated = {"samples": {"gsm8k_openmath": [identity, dict(identity), dict(identity)]}}

    assert _opsd_sample_manifest(evaluated, ("gsm8k_openmath",)) == [
        {"task": "gsm8k_openmath", **identity}
    ]



def test_opsd_screen_aggregate_rejects_impossible_cache_lengths():
    documents = [
        _document(name, score)
        for name, score in zip(OPSD_SCREEN_CONDITIONS, (0.1, 0.2, 0.3, 0.5), strict=True)
    ]
    documents[1]["generation_records"][0]["cache_event_lengths"] = [999999]

    with pytest.raises(ValueError, match="cache-event evidence"):
        aggregate_opsd_screen(documents)


def test_opsd_screen_aggregate_rejects_condition_flag_mismatch():
    documents = [
        _document(name, score)
        for name, score in zip(OPSD_SCREEN_CONDITIONS, (0.1, 0.2, 0.3, 0.5), strict=True)
    ]
    documents[1]["condition"]["processor_enabled"] = False
    documents[1]["generation_records"][0]["cache_event_positions"] = []
    documents[1]["generation_records"][0]["cache_event_lengths"] = []

    with pytest.raises(ValueError, match="condition name disagrees"):
        aggregate_opsd_screen(documents)


def test_atomic_write_new_refuses_broken_symlink(tmp_path):
    destination = tmp_path / "broken.json"
    destination.symlink_to(tmp_path / "missing-target")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _atomic_write_new(destination, "new")

    assert destination.is_symlink()
    assert not destination.resolve().exists()
