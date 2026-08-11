from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.evaluate_process_branch_suite as suite


def test_dry_run_receipt_separates_heldout_and_blocked_final_rows(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "gcw.answer_only_eval.protocol.v1",
                "evaluator_version": "gcw.answer_only_eval.v1",
                "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
                "dataset": {
                    "path": "/data/answer.jsonl",
                    "sha256": "a" * 64,
                    "ordered_source_ids_sha256": "b" * 64,
                },
                "generation": {
                    "answer_prefix": "\n\n",
                    "prompt_suffix": "answer",
                    "max_new_tokens": 32,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": 0,
                },
                "splits": {
                    "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
                    "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
                },
                "arms": {
                    "p0": {
                        "run_id": "p0", "train_config": "configs/p0.yaml",
                        "checkpoint": "/runs/p0/final", "checkpoint_sha256": None,
                        "checkpoint_manifest_sha256": None, "config_sha256": "c" * 64,
                    },
                    "p1": {
                        "run_id": "p1", "train_config": "configs/p1.yaml",
                        "checkpoint": "/runs/p1/final", "checkpoint_sha256": None,
                        "checkpoint_manifest_sha256": None, "config_sha256": "c" * 64,
                    },
                },
            }
        )
    )

    receipt = suite.build_answer_only_dry_run_receipt(protocol)

    assert receipt["schema"] == "gcw.answer_only_eval.dry_run_receipt.v1"
    assert len(receipt["commands"]) == 4
    assert {(command["arm"], command["split"]) for command in receipt["commands"]} == {
        ("p0", "heldout"), ("p0", "final"), ("p1", "heldout"), ("p1", "final")
    }
    assert all(command["execute"] is False for command in receipt["commands"])
    final = [command for command in receipt["commands"] if command["split"] == "final"]
    assert all("protocol_forbids_execution" in command["blockers"] for command in final)
    assert all("checkpoint_sha256_not_frozen" in command["blockers"] for command in receipt["commands"])
    assert all("checkpoint_manifest_sha256_not_frozen" in command["blockers"] for command in receipt["commands"])
    assert all("checkpoint_not_visible" in command["blockers"] for command in receipt["commands"])
    assert all("train_config_not_visible" in command["blockers"] for command in receipt["commands"])
    assert {(command["skip_rows"], command["take_rows"]) for command in receipt["commands"]} == {
        (6000, 256), (6256, 256)
    }


def test_suite_cli_writes_no_clobber_dry_run_receipt(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "schema": "gcw.answer_only_eval.protocol.v1",
                "evaluator_version": "gcw.answer_only_eval.v1",
                "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
                "dataset": {
                    "path": "/data/answer.jsonl",
                    "sha256": "a" * 64,
                    "ordered_source_ids_sha256": "b" * 64,
                },
                "generation": {"answer_prefix": "\n\n", "prompt_suffix": "answer", "max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0, "top_k": 0},
                "splits": {
                    "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
                    "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
                },
                "arms": {
                    "p0": {
                        "run_id": "p0", "train_config": "configs/p0.yaml",
                        "checkpoint": "/runs/p0/final", "checkpoint_sha256": None,
                        "checkpoint_manifest_sha256": None, "config_sha256": "c" * 64,
                    },
                    "p1": {
                        "run_id": "p1", "train_config": "configs/p1.yaml",
                        "checkpoint": "/runs/p1/final", "checkpoint_sha256": None,
                        "checkpoint_manifest_sha256": None, "config_sha256": "c" * 64,
                    },
                },
            }
        )
    )
    receipt = tmp_path / "receipt.json"

    assert suite.main(["--answer-only-protocol", str(protocol), "--dry-run-receipt", str(receipt)]) == 0
    assert json.loads(receipt.read_text())["final_rows_evaluated"] is False


def test_finalize_answer_only_protocol_authenticates_and_freezes_checkpoint_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hashlib

    calls = []
    arms = {}
    for arm_name in ("p0", "p1"):
        config = tmp_path / f"{arm_name}.yaml"
        config.write_bytes(f"arm: {arm_name}\n".encode())
        checkpoint = tmp_path / arm_name / "final"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(f"model-{arm_name}".encode())
        (checkpoint / "manifest.json").write_bytes(f'{{"arm":"{arm_name}"}}\n'.encode())
        arms[arm_name] = {
            "run_id": arm_name,
            "train_config": str(config),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": None,
            "checkpoint_manifest_sha256": None,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        }
    payload = {
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": "gcw.answer_only_eval.v1",
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "dataset": {
            "path": "/data/answer.jsonl",
            "sha256": "a" * 64,
            "ordered_source_ids_sha256": "b" * 64,
        },
        "generation": {
            "answer_prefix": "\n\n",
            "prompt_suffix": "answer",
            "max_new_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
        },
        "splits": {
            "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
            "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
        },
        "arms": arms,
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))
    original = protocol.read_bytes()

    def authenticate(checkpoint, *, expected_manifest_sha256, expected_model_sha256):
        calls.append((checkpoint, expected_manifest_sha256, expected_model_sha256))
        return {}

    monkeypatch.setattr(suite, "authenticate_canonical_checkpoint", authenticate)

    finalized = suite.finalize_answer_only_protocol(protocol)

    assert protocol.read_bytes() == original
    assert finalized["splits"]["final"]["execution_allowed"] is False
    assert len(calls) == 2
    for arm_name, arm in finalized["arms"].items():
        checkpoint = Path(arm["checkpoint"])
        assert arm["checkpoint_sha256"] == hashlib.sha256(
            (checkpoint / "model.safetensors").read_bytes()
        ).hexdigest()
        assert arm["checkpoint_manifest_sha256"] == hashlib.sha256(
            (checkpoint / "manifest.json").read_bytes()
        ).hexdigest()


def test_repository_precursor_protocol_freezes_p0_p1_and_keeps_final_blocked() -> None:
    protocol_path = Path("configs/eval/process_branch_qwen3_06b/answer_only_precursor_off_init_k_v1.json")
    protocol = suite._load_answer_only_protocol(protocol_path)

    assert protocol["splits"] == {
        "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
        "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
    }
    assert set(protocol["arms"]) == {"p0", "p1"}
    assert all(arm["checkpoint_sha256"] is None for arm in protocol["arms"].values())
    assert protocol["conditions"] == ["OFF", "INIT", "K1", "K2", "K4", "K8"]


def test_dry_run_receipt_uses_strict_protocol_json_parser(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"schema":"gcw.answer_only_eval.protocol.v1",'
        '"schema":"gcw.answer_only_eval.protocol.v1"}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key: schema"):
        suite.build_answer_only_dry_run_receipt(protocol)


@pytest.mark.parametrize("mutation", ["extra", "uppercase_digest", "nonfinite_generation"])
def test_dry_run_receipt_rejects_protocol_outside_exact_execution_contract(
    tmp_path, mutation: str
) -> None:
    payload = {
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": "gcw.answer_only_eval.v1",
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "dataset": {
            "path": "/data/answer.jsonl", "sha256": "a" * 64,
            "ordered_source_ids_sha256": "b" * 64,
        },
        "generation": {
            "answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 32,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        },
        "splits": {
            "heldout": {"skip_rows": 6000, "take_rows": 2, "execution_allowed": True},
            "final": {"skip_rows": 6002, "take_rows": 2, "execution_allowed": False},
        },
        "arms": {
            name: {
                "run_id": name, "train_config": f"/{name}.yaml",
                "checkpoint": f"/{name}/final", "checkpoint_sha256": None,
                "checkpoint_manifest_sha256": None, "config_sha256": "c" * 64,
            }
            for name in ("p0", "p1")
        },
    }
    if mutation == "extra":
        payload["unreviewed"] = True
    elif mutation == "uppercase_digest":
        payload["dataset"]["sha256"] = "A" * 64
    else:
        payload["generation"]["temperature"] = float("nan")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))

    with pytest.raises(ValueError):
        suite.build_answer_only_dry_run_receipt(protocol)


def test_dry_run_receipt_authenticates_ordered_ids_from_dataset_snapshot(tmp_path) -> None:
    import hashlib

    data = tmp_path / "data.jsonl"
    data.write_text('{"source_id":"id:0","prompt":"Q","final_answer":"A"}\n')
    config = tmp_path / "train.yaml"
    config.write_text("config\n")
    payload = {
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": "gcw.answer_only_eval.v1",
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "dataset": {
            "path": str(data),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "ordered_source_ids_sha256": "0" * 64,
        },
        "generation": {
            "answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        },
        "splits": {
            "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
            "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
        },
        "arms": {
            name: {
                "run_id": name, "train_config": str(config),
                "checkpoint": str(tmp_path / name), "checkpoint_sha256": None,
                "checkpoint_manifest_sha256": None,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            for name in ("p0", "p1")
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))

    receipt = suite.build_answer_only_dry_run_receipt(protocol)

    assert all(
        "dataset_authentication_failed" in command["blockers"]
        for command in receipt["commands"]
    )


def test_dry_run_receipt_binds_the_same_protocol_snapshot_it_parses(
    monkeypatch, tmp_path
) -> None:
    import hashlib

    data = tmp_path / "data.jsonl"
    data.write_text('{"source_id":"id:0","prompt":"Q","final_answer":"A"}\n')
    config = tmp_path / "train.yaml"
    config.write_text("config\n")
    payload = {
        "schema": "gcw.answer_only_eval.protocol.v1",
        "evaluator_version": "gcw.answer_only_eval.v1",
        "conditions": ["OFF", "INIT", "K1", "K2", "K4", "K8"],
        "dataset": {
            "path": str(data),
            "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
            "ordered_source_ids_sha256": hashlib.sha256(
                b'["id:0"]'
            ).hexdigest(),
        },
        "generation": {
            "answer_prefix": "P", "prompt_suffix": "S", "max_new_tokens": 2,
            "temperature": 0.0, "top_p": 1.0, "top_k": 0,
        },
        "splits": {
            "heldout": {"skip_rows": 6000, "take_rows": 256, "execution_allowed": True},
            "final": {"skip_rows": 6256, "take_rows": 256, "execution_allowed": False},
        },
        "arms": {
            name: {
                "run_id": name, "train_config": str(config),
                "checkpoint": str(tmp_path / name), "checkpoint_sha256": None,
                "checkpoint_manifest_sha256": None,
                "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            }
            for name in ("p0", "p1")
        },
    }
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload))
    authenticated_bytes = protocol.read_bytes()
    original_read_text = Path.read_text

    def mutate_after_read(path, *args, **kwargs):
        text = original_read_text(path, *args, **kwargs)
        if path == protocol:
            protocol.write_text('{"tampered":true}')
        return text

    monkeypatch.setattr(Path, "read_text", mutate_after_read)
    receipt = suite.build_answer_only_dry_run_receipt(protocol)

    assert receipt["protocol_sha256"] == hashlib.sha256(authenticated_bytes).hexdigest()
    assert protocol.read_bytes() == authenticated_bytes
