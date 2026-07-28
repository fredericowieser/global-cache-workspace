from __future__ import annotations

from pathlib import Path

import pytest

from scripts.flow_math_eval_runner import (
    build_eval_command,
    CANONICAL_ARRAY_TARGETS,
    build_runtime_evidence,
    publish_result_no_replace,
    validate_runtime_evidence,
)


def _environment(cache_dir: Path, **overrides: str) -> dict[str, str]:
    values = {
        "CUDA_VISIBLE_DEVICES": "2",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "RANK": "0",
        "SLURM_JOB_ID": "123",
        "SLURM_ARRAY_JOB_ID": "120",
        "SLURM_ARRAY_TASK_ID": "1",
        "SLURM_NTASKS": "1",
        "SLURM_CPUS_PER_TASK": "16",
        "GCW_STRICT_FLASH": "1",
        "WANDB_MODE": "offline",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GCW_EVAL_PARTIAL_DIR": str(cache_dir.parent / "partial"),
        "GCW_EVAL_SKIP_TASKS": "theoremqa",
        "GCW_EVAL_REFERENCE_GENERATE": "0",
        "PYTHONPATH": "",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "HF_HOME": str(cache_dir / "huggingface"),
        "HF_DATASETS_CACHE": str(cache_dir / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(cache_dir / "huggingface" / "hub"),
    }
    values.update(overrides)
    return {key: value for key, value in values.items() if value != ""}


def test_canonical_array_targets_match_seed_major_matrix_order():
    assert CANONICAL_ARRAY_TARGETS == (
        "tf-backbone",
        "bgt-seed24", "icarus-seed24", "longshot-seed24",
        "bgt-seed25", "icarus-seed25", "longshot-seed25",
        "bgt-seed26", "icarus-seed26", "longshot-seed26",
    )


def test_runtime_evidence_records_and_validates_exact_invocation(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    output = tmp_path / "result.json"
    temporary_output = tmp_path / ".result.json.test.tmp.json"
    partial_identity = tmp_path / "identity.json"
    partial_identity.write_text('{"identity": 1}\n')
    command = build_eval_command(
        python=Path("/venv/bin/python"),
        checkpoint=checkpoint,
        batch_size=16,
        limit=None,
        out=temporary_output,
    )
    evidence = build_runtime_evidence(
        target="bgt-seed24",
        command=command,
        checkpoint=checkpoint,
        output=output,
        temporary_output=temporary_output,
        partial_identity=partial_identity,
        eval_cache_dir=tmp_path / "cache",
        expected_array_task_id=1,
        batch_size=16,
        limit=None,
        cuda_device_count=1,
        environ=_environment(tmp_path / "cache", GCW_EVAL_PARTIAL_DIR=str(partial_identity.parent.resolve())),
    )

    validate_runtime_evidence(
        evidence,
        target="bgt-seed24",
        checkpoint=checkpoint,
        output=output,
        partial_identity=partial_identity,
        eval_cache_dir=tmp_path / "cache",
        expected_array_task_id=1,
        expected_python=Path("/venv/bin/python"),
        batch_size=16,
        limit=None,
        require_slurm=True,
    )
    assert evidence["command"] == command
    assert evidence["checkpoint"] == str(checkpoint.resolve())
    assert evidence["output"] == str(output.resolve())
    assert evidence["partial_identity"] == str(partial_identity.resolve())
    assert evidence["strict_flash"] == "1"
    assert evidence["wandb_mode"] == "offline"
    assert evidence["generation_policy"] == {
        "do_sample": False,
        "legacy_bottleneck_generate": False,
    }

    tampered = dict(evidence)
    tampered["command"] = ["python", "-c", "print('fabricated')"]
    with pytest.raises(ValueError, match="command"):
        validate_runtime_evidence(
            tampered,
            target="bgt-seed24",
            checkpoint=checkpoint,
            output=output,
            partial_identity=partial_identity,
            eval_cache_dir=tmp_path / "cache",
            expected_array_task_id=1,
            expected_python=Path("/venv/bin/python"),
            batch_size=16,
            limit=None,
            require_slurm=True,
        )


@pytest.mark.parametrize(
    ("cuda_device_count", "overrides", "message"),
    [
        (2, {}, "exactly one visible CUDA device"),
        (1, {"WORLD_SIZE": "2"}, "world size 1"),
        (1, {"LOCAL_RANK": "1"}, "rank zero"),
        (1, {"GCW_STRICT_FLASH": "0"}, "strict FlashAttention"),
        (1, {"HF_HUB_OFFLINE": "0"}, "offline evaluation"),
    ],
)
def test_build_runtime_evidence_rejects_noncanonical_topology(
    tmp_path, cuda_device_count, overrides, message
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match=message):
        build_runtime_evidence(
            target="tf-backbone",
            command=["python"],
            checkpoint=checkpoint,
            output=tmp_path / "result.json",
            temporary_output=tmp_path / ".result.tmp.json",
            partial_identity=None,
            eval_cache_dir=tmp_path / "cache",
            expected_array_task_id=None,
            batch_size=16,
            limit=16,
            cuda_device_count=cuda_device_count,
            environ=_environment(
                tmp_path / "cache", GCW_EVAL_PARTIAL_DIR="", SLURM_ARRAY_JOB_ID="",
                SLURM_ARRAY_TASK_ID="", **overrides
            ),
        )


def test_build_eval_command_is_single_process_and_preserves_limit(tmp_path):
    command = build_eval_command(
        python=Path("/venv/bin/python"),
        checkpoint=Path("/checkpoint"),
        batch_size=16,
        limit=16,
        out=tmp_path / "result.json",
    )

    assert command[:6] == ["/venv/bin/python", "-m", "gcw.cli", "eval", "math", "--checkpoint"]
    assert "torch.distributed" not in " ".join(command)
    assert command[-2:] == ["--limit", "16"]


def test_publish_result_is_atomic_no_replace_and_read_only(tmp_path):
    temporary = tmp_path / ".result.tmp.json"
    temporary.write_text('{"ok": true}\n')
    output = tmp_path / "result.json"
    publish_result_no_replace(temporary, output)
    assert output.read_text() == '{"ok": true}\n'
    assert output.stat().st_mode & 0o222 == 0
    assert not temporary.exists()

    second = tmp_path / ".second.tmp.json"
    second.write_text('{"ok": false}\n')
    with pytest.raises(FileExistsError):
        publish_result_no_replace(second, output)
    assert output.read_text() == '{"ok": true}\n'
