from __future__ import annotations

import hashlib
import json
import sys
import types

import pytest

from gcw import process_branch_trainer, process_data
from gcw.config import load_train_config
from gcw.process_data import ProcessDataError, read_answer_only_records


@pytest.mark.parametrize(
    "rationale_lines",
    [
        [],
        ["Malformed <<6*7=41 rationale."],
        ["Malformed << annotation"] * 7,
        ["overlong " * 20_000],
    ],
    ids=["zero", "one", "many", "overlong"],
)
def test_answer_only_preparer_reads_upstream_rows_without_process_filtering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    rationale_lines: list[str],
) -> None:
    from scripts import prepare_answer_only_gsm8k

    source = tmp_path / "gsm8k-train.arrow"
    source.write_bytes(b"authenticated upstream arrow bytes")
    output = tmp_path / "answer-only.jsonl"
    manifest = tmp_path / "answer-only.manifest.json"
    rows = [
        {
            "question": "  What is six times seven?  ",
            "answer": "\n".join([*rationale_lines, "#### 42"]),
        },
        {"question": None, "answer": "#### 0"},
        {"question": "What is nine minus four?", "answer": "#### 5"},
    ]

    class FakeDataset:
        @staticmethod
        def from_file(path: str):
            snapshot = type(source)(path)
            assert snapshot != source
            assert snapshot.read_bytes() == source.read_bytes()
            return rows

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))

    def process_parser_must_not_run(**_kwargs):
        raise AssertionError("strict process parser was invoked")

    monkeypatch.setattr("gcw.process_data.parse_gsm8k_process_record", process_parser_must_not_run)

    assert (
        prepare_answer_only_gsm8k.main(
            [
                "--source-arrow",
                str(source),
                "--source-split",
                "train",
                "--output",
                str(output),
                "--manifest",
                str(manifest),
            ]
        )
        == 0
    )

    payloads = [json.loads(line) for line in output.read_text().splitlines()]
    assert payloads == [
        {
            "source_id": "gsm8k-train:0",
            "prompt": "What is six times seven?",
            "final_answer": "\\boxed{42}",
        },
        {
            "source_id": "gsm8k-train:2",
            "prompt": "What is nine minus four?",
            "final_answer": "\\boxed{5}",
        },
    ]
    assert json.loads(manifest.read_text()) == {
        "accepted_rows": 2,
        "kind": "gcw.answer_only.gsm8k.v1",
        "ordered_source_ids_sha256": hashlib.sha256(
            b'["gsm8k-train:0","gsm8k-train:2"]'
        ).hexdigest(),
        "output": str(output),
        "rejected_rows": 1,
        "rejection_reasons": {"invalid_schema": 1},
        "schema_version": 1,
        "source_arrow": str(source),
        "source_rows": 3,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_split": "train",
        "transformed_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def test_answer_only_reader_preserves_unicode_line_separator_inside_json_string(tmp_path) -> None:
    artifact = tmp_path / "answer-only.jsonl"
    prompt = "First sentence.\u2028Second sentence."
    artifact.write_bytes(
        (
            json.dumps(
                {
                    "source_id": "gsm8k-train:2381",
                    "prompt": prompt,
                    "final_answer": "\\boxed{36}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    )

    records = read_answer_only_records(artifact)

    assert len(records) == 1
    assert records[0].prompt == prompt


def test_answer_only_reader_rejects_duplicate_source_ids_outside_selected_slice(tmp_path) -> None:
    artifact = tmp_path / "answer-only.jsonl"
    artifact.write_text(
        "".join(
            json.dumps(
                {
                    "source_id": source_id,
                    "prompt": f"Q{index}",
                    "final_answer": f"A{index}",
                }
            )
            + "\n"
            for index, source_id in enumerate(
                ["gsm8k-train:0", "gsm8k-train:1", "gsm8k-train:0"]
            )
        )
    )

    with pytest.raises(ProcessDataError, match="duplicate_source_id:gsm8k-train:0"):
        read_answer_only_records(artifact, skip_rows=1, take_rows=1)


def test_answer_only_artifact_authentication_checks_bytes_and_ordered_source_ids(tmp_path) -> None:
    artifact = tmp_path / "answer-only.jsonl"
    payloads = [
        {"source_id": "gsm8k-train:0", "prompt": "Q0", "final_answer": "A0"},
        {"source_id": "gsm8k-train:2", "prompt": "Q2", "final_answer": "A2"},
    ]
    artifact.write_text("".join(json.dumps(payload) + "\n" for payload in payloads))
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    source_ids_sha256 = hashlib.sha256(
        json.dumps(
            [payload["source_id"] for payload in payloads],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()

    authenticated = process_data.authenticate_answer_only_artifact(
        artifact.read_bytes(),
        expected_artifact_sha256=artifact_sha256,
        expected_source_ids_sha256=source_ids_sha256,
    )
    assert [record.source_id for record in authenticated] == [
        "gsm8k-train:0", "gsm8k-train:2"
    ]

    with pytest.raises(ProcessDataError, match="ordered source-ID sha256 mismatch"):
        process_data.authenticate_answer_only_artifact(
            artifact.read_bytes(),
            expected_artifact_sha256=artifact_sha256,
            expected_source_ids_sha256="1" * 64,
        )


def test_canonical_training_authenticates_answer_only_artifact_before_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    artifact = tmp_path / "answer-only.jsonl"
    artifact.write_text(
        json.dumps(
            {"source_id": "gsm8k-train:0", "prompt": "Q", "final_answer": "A"}
        )
        + "\n"
    )
    output = tmp_path / "must-not-exist"
    config.data.data_dir = str(artifact)
    config.output_dir = str(output)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n")
    config.data.tokenizer = f"hf:{tokenizer}"

    class AuthenticationReached(RuntimeError):
        pass

    def stop_after_authentication(
        source,
        *,
        expected_artifact_sha256,
        expected_source_ids_sha256,
    ) -> None:
        assert source == artifact.read_bytes()
        assert expected_artifact_sha256 == config.data.expected_artifact_sha256
        assert expected_source_ids_sha256 == config.data.expected_source_ids_sha256
        raise AuthenticationReached

    monkeypatch.setattr(
        process_branch_trainer,
        "authenticate_answer_only_artifact",
        stop_after_authentication,
        raising=False,
    )

    with pytest.raises(AuthenticationReached):
        process_branch_trainer.train_process_branch(config)
    assert not output.exists()


def test_canonical_training_rejects_artifact_changed_during_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    artifact = tmp_path / "answer-only.jsonl"
    artifact.write_text(
        json.dumps(
            {"source_id": "gsm8k-train:0", "prompt": "Q", "final_answer": "A"}
        )
        + "\n"
    )
    output = tmp_path / "must-not-exist"
    config.data.data_dir = str(artifact)
    config.output_dir = str(output)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.json").write_text("{}\n")
    config.data.tokenizer = f"hf:{tokenizer}"

    class AuthenticationReached(RuntimeError):
        pass

    original = artifact.read_bytes()

    def mutate_during_authentication(source_bytes, **_kwargs):
        assert source_bytes == original
        artifact.write_bytes(b"tampered path bytes")
        raise AuthenticationReached

    monkeypatch.setattr(
        process_branch_trainer,
        "authenticate_answer_only_artifact",
        mutate_during_authentication,
    )

    with pytest.raises(AuthenticationReached):
        process_branch_trainer.train_process_branch(config)
    assert not output.exists()


def test_canonical_training_uses_one_immutable_snapshot_and_writes_nothing_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = load_train_config(
        "configs/train/process_branch_qwen3_06b/qwen3_06b_answer_only_k1_6000_seed24.yaml"
    )
    artifact = tmp_path / "answer-only.jsonl"
    artifact.write_bytes(b"not-json\n")
    output = tmp_path / "must-not-exist"
    config.data.data_dir = str(artifact)
    config.data.canonical_execution = False
    config.output_dir = str(output)
    original_read_bytes = type(artifact).read_bytes
    artifact_reads = 0

    def adversarial_read_bytes(path):
        nonlocal artifact_reads
        if path == artifact:
            artifact_reads += 1
            if artifact_reads > 1:
                return b'{"source_id":"tampered","prompt":"Q","final_answer":"A"}\n'
        return original_read_bytes(path)

    def mutable_path_reader_must_not_run(*_args, **_kwargs):
        raise AssertionError("training reopened mutable JSONL path")

    monkeypatch.setattr(type(artifact), "read_bytes", adversarial_read_bytes)
    monkeypatch.setattr(
        process_branch_trainer,
        "read_answer_only_records",
        mutable_path_reader_must_not_run,
        raising=False,
    )

    with pytest.raises(ProcessDataError, match="invalid_json_row:0"):
        process_branch_trainer.train_process_branch(config)
    assert artifact_reads == 1
    assert not output.exists()


def test_answer_only_preparer_reads_private_arrow_snapshot_when_source_is_swapped(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from scripts import prepare_answer_only_gsm8k

    source = tmp_path / "source.arrow"
    original = b"immutable-arrow-snapshot"
    source.write_bytes(original)
    output = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    opened_paths = []

    class FakeDataset:
        @staticmethod
        def from_file(path: str):
            snapshot = type(source)(path)
            opened_paths.append(snapshot)
            assert snapshot != source
            assert snapshot.read_bytes() == original
            source.write_bytes(b"tampered-after-snapshot")
            return [{"question": "Q", "answer": "#### 1"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))

    assert prepare_answer_only_gsm8k.main(
        [
            "--source-arrow", str(source),
            "--source-split", "train",
            "--output", str(output),
            "--manifest", str(manifest),
        ]
    ) == 0
    payload = json.loads(manifest.read_text())
    assert payload["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert len(opened_paths) == 1
    assert not opened_paths[0].exists()


def test_answer_only_preparer_rolls_back_output_when_manifest_commit_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from scripts import prepare_answer_only_gsm8k

    source = tmp_path / "source.arrow"
    source.write_bytes(b"arrow")
    output = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"

    class FakeDataset:
        @staticmethod
        def from_file(_path: str):
            return [{"question": "Q", "answer": "#### 1"}]

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(Dataset=FakeDataset))
    original_publish = prepare_answer_only_gsm8k._publish_no_clobber

    def fail_manifest(staged, destination):
        if destination == manifest:
            raise RuntimeError("forced manifest commit failure")
        original_publish(staged, destination)

    monkeypatch.setattr(prepare_answer_only_gsm8k, "_publish_no_clobber", fail_manifest)
    with pytest.raises(RuntimeError, match="forced manifest commit failure"):
        prepare_answer_only_gsm8k.main([
            "--source-arrow", str(source), "--source-split", "train",
            "--output", str(output), "--manifest", str(manifest),
        ])
    assert not output.exists()
    assert not manifest.exists()
    assert not list(tmp_path.glob(".*.tmp"))
