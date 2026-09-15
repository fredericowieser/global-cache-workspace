import os
import re
import subprocess
import time
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "cluster" / "submit_array.sh"


def _write_fixture(
    tmp_path: Path, uv_body: str = "exit 0"
) -> tuple[Path, dict[str, str]]:
    repo_root = tmp_path / "work"
    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (repo_root / "scripts" / "cluster").mkdir(parents=True)
    config = tmp_path / "config.yaml"
    (repo_root / "scripts" / "cluster" / "check_cuda_env.py").write_text(
        "# fake\n", encoding="utf-8"
    )
    config.write_text(
        f"output_dir: {tmp_path / 'output'}\nmax_steps: 1\n", encoding="utf-8"
    )
    uv = bin_dir / "uv"
    uv.write_text(f"#!/usr/bin/env bash\n{uv_body}\n", encoding="utf-8")
    uv.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(home),
            "GCW_REPO_ROOT": str(repo_root),
            "LOCAL_STATE_ROOT": str(tmp_path / "local-state"),
            "EVAL_AFTER_TRAIN": "0",
            "JOB_NAME": "test-array",
            "LOCAL_STOP_WHEN_IDLE": "0",
        }
    )
    return config, env


def _run(
    args: list[str], env: dict[str, str], *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=SCRIPT.parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=check,
    )


def _state_dir(output: str) -> Path:
    match = re.search(r"^State directory: (.+)$", output, re.MULTILINE)
    assert match, output
    return Path(match.group(1))


def _wait_for_status(
    state_dir: Path, env: dict[str, str], expected: str
) -> str:
    deadline = time.monotonic() + 5
    last = ""
    while time.monotonic() < deadline:
        result = _run(["status", str(state_dir)], env, check=False)
        last = result.stdout
        if expected in last:
            return last
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {expected!r}:\n{last}")


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


def _wait_for_text(path: Path, expected: str) -> str:
    deadline = time.monotonic() + 5
    last = ""
    while time.monotonic() < deadline:
        if path.exists():
            last = path.read_text(encoding="utf-8")
            if expected in last:
                return last
        time.sleep(0.05)
    raise AssertionError(
        f"Timed out waiting for {expected!r} in {path}:\n{last}"
    )


def test_local_dry_run_partitions_gpus_and_caps_workers(tmp_path: Path) -> None:
    config, env = _write_fixture(tmp_path)
    env.update({"GPUS": "2", "EVAL_GPUS": "1"})

    result = _run(
        [
            "--backend",
            "local",
            "--dry-run",
            "--workers",
            "3",
            "--local-gpus",
            "0,1,2,3",
            str(config),
        ],
        env,
    )

    assert "Only 2 worker(s) fit" in result.stdout
    assert "Worker 0: assigned GPUs 0,1" in result.stdout
    assert "Worker 1: assigned GPUs 2,3" in result.stdout
    state_dir = _state_dir(result.stdout)
    assert not list(state_dir.glob("worker-*.pid"))


def test_local_worker_records_success_and_status(tmp_path: Path) -> None:
    config, env = _write_fixture(tmp_path)
    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)], env
    )
    state_dir = _state_dir(result.stdout)

    status = _wait_for_status(state_dir, env, "SUCCEEDED")
    assert "SUCCEEDED" in status
    assert "1 succeeded" in status
    assert (state_dir / "completed_configs").read_text(
        encoding="utf-8"
    ).splitlines() == [str(config)]
    assert (state_dir / "failed_configs.tsv").read_text(encoding="utf-8") == ""
    assert (state_dir / "worker-0.exit").read_text(
        encoding="utf-8"
    ).strip() == "0"
    assert (state_dir / "backend").read_text(
        encoding="utf-8"
    ).strip() == "local"


def test_worker_does_not_validate_unrelated_openmath_override(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path)
    env["OPENMATH_DATA"] = str(tmp_path / "missing.jsonl")

    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)], env
    )

    _wait_for_status(_state_dir(result.stdout), env, "SUCCEEDED")


def test_slurm_uses_ephemeral_export_file_without_login_env_retrieval(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path)
    state_root = tmp_path / "slurm-state"
    capture = tmp_path / "captured-environment"
    bin_dir = Path(env["HOME"]) / ".local" / "bin"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == --export-file=* ]]; then\n"
        "    cp \"${arg#--export-file=}\" \"$SBATCH_EXPORT_CAPTURE\"\n"
        "  fi\n"
        "done\n"
        "printf '321\\n'\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "SLURM_STATE_ROOT": str(state_root),
            "SLURM_LOG_DIR": str(tmp_path / "slurm-logs"),
            "SBATCH_EXPORT_CAPTURE": str(capture),
        }
    )

    result = _run(["--backend", "slurm", str(config)], env)

    assert "Submitted array job 321" in result.stdout
    state_dir = next(state_root.iterdir())
    assert not (state_dir / "slurm-environment").exists()
    entries = dict(
        item.split(b"=", 1)
        for item in capture.read_bytes().split(b"\0")
        if item
    )
    assert entries[b"GCW_ARRAY_MODE"] == b"worker"
    assert entries[b"CONFIG_LIST"] == str(
        state_dir / "configs.txt"
    ).encode()
    assert entries[b"SLURM_EXPORT_ENV"] == b"ALL"


def test_dev_defaults_use_qwen35_paper7_eval_and_cluster_preflight(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "uv-args"
    config, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"',
    )
    env.update({"EVAL_AFTER_TRAIN": "1", "UV_CAPTURE": str(capture)})

    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)], env
    )
    state_dir = _state_dir(result.stdout)
    _wait_for_status(state_dir, env, "SUCCEEDED")

    invocations = capture.read_text(encoding="utf-8").splitlines()
    expected_prefix = "run --locked --extra gpu --extra eval --extra qwen35"
    assert (
        f"{expected_prefix} python scripts/cluster/check_cuda_env.py"
        in invocations
    )
    assert any(
        f"{expected_prefix} gcw train --config " in line for line in invocations
    )
    assert any(
        f"{expected_prefix} gcw eval paper7 --checkpoint " in line
        for line in invocations
    )


def test_eval_only_targets_one_step_with_generation_cadence(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "uv-args"
    config, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"',
    )
    env.update(
        {
            "EVAL_ONLY": "1",
            "EVAL_CHECKPOINT_STEP": "3000",
            "GENERATION_PROCESS_EVERY": "1",
            "EVAL_HEAD_SELECTION_METHOD": "random",
            "OUT_SUFFIX": "generation-process-every1",
            "UV_CAPTURE": str(capture),
        }
    )

    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)],
        env,
    )
    state_dir = _state_dir(result.stdout)
    _wait_for_status(state_dir, env, "SUCCEEDED")

    invocation = next(
        line
        for line in capture.read_text(encoding="utf-8").splitlines()
        if " gcw eval " in f" {line} "
    )
    assert f"--checkpoint {tmp_path / 'output'}/checkpoint-3000" in invocation
    assert "--generation-process-every 1" in invocation
    assert "--head-selection-method random" in invocation
    assert "--batch-size 16" in invocation
    assert "step3000_generation-process-every1_paper7.json" in invocation


def test_eval_only_passes_automatic_batching_options(tmp_path: Path) -> None:
    capture = tmp_path / "uv-args"
    config, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"',
    )
    env.update(
        {
            "EVAL_ONLY": "1",
            "EVAL_BATCH_SIZE": "auto",
            "EVAL_MAX_BATCH_SIZE": "32",
            "UV_CAPTURE": str(capture),
        }
    )

    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)],
        env,
    )
    state_dir = _state_dir(result.stdout)
    _wait_for_status(state_dir, env, "SUCCEEDED")

    invocation = next(
        line
        for line in capture.read_text(encoding="utf-8").splitlines()
        if " gcw eval " in f" {line} "
    )
    assert "--batch-size auto" in invocation
    assert "--max-batch-size 32" in invocation


def test_generation_cadence_rejects_non_generative_eval_suite(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path)
    env.update({"EVAL_SUITE": "fineweb", "GENERATION_PROCESS_EVERY": "1"})

    result = _run(
        [
            "--backend",
            "local",
            "--dry-run",
            "--local-gpus",
            "GPU-test",
            str(config),
        ],
        env,
        check=False,
    )

    assert result.returncode != 0
    assert "not supported for fineweb" in result.stdout


def test_local_worker_can_be_cancelled_as_a_process_group(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path, uv_body="sleep 60")
    env.update({"LOCAL_STOP_WHEN_IDLE": "1", "LOCAL_IDLE_CHECK_INTERVAL": "1"})
    result = _run(["--backend", "local", "--local-gpus", "0", str(config)], env)
    state_dir = _state_dir(result.stdout)

    try:
        running = _wait_for_status(state_dir, env, "RUNNING")
        assert "0" in running

        blocked = _run(
            [
                "--backend",
                "local",
                "--dry-run",
                "--local-gpus",
                "0",
                str(config),
            ],
            env,
            check=False,
        )
        assert blocked.returncode != 0
        assert "No unreserved local GPU group" in blocked.stdout

        cancelled = _run(["cancel", str(state_dir)], env)
        assert "Sent SIGTERM to worker 0 process group" in cancelled.stdout
        final_status = _wait_for_status(state_dir, env, "CANCELLED")
        assert "1 cancelled" in final_status
        assert (state_dir / "worker-0.cancelled").exists()
        inhibit_marker = Path(env["LOCAL_STATE_ROOT"]) / "vm-stop-inhibited"
        assert inhibit_marker.exists()
        monitor_log = state_dir / "idle-stop-monitor.out"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (
                monitor_log.exists()
                and "keeping the node running"
                in monitor_log.read_text(encoding="utf-8")
            ):
                break
            time.sleep(0.05)
        else:
            raise AssertionError(
                "Idle monitor did not honor the cancellation inhibit marker"
            )
    finally:
        _run(["cancel", "--force", str(state_dir)], env, check=False)


def test_idle_monitor_stops_nebius_only_after_gpu_is_idle(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path)
    bin_dir = Path(env["HOME"]) / ".local" / "bin"
    capture = tmp_path / "nebius-args"
    discord_capture = tmp_path / "discord-payloads"
    query_capture = tmp_path / "gpu-query"
    busy = tmp_path / "gpu-busy"
    instance_id = tmp_path / "instance-id"
    nvidia_smi = bin_dir / "nvidia-smi"
    nebius = bin_dir / "nebius"
    curl = bin_dir / "curl"

    busy.write_text("", encoding="utf-8")
    instance_id.write_text("vm-test-123\n", encoding="utf-8")
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        'printf queried > "$GPU_QUERY_CAPTURE"\n'
        'if [[ -e "$GPU_BUSY_FILE" ]]; then\n'
        '  printf "99999\\n"\n'
        "fi\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)
    nebius.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" > "$NEBIUS_CAPTURE"\n',
        encoding="utf-8",
    )
    nebius.chmod(0o755)
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "while [[ $# -gt 0 ]]; do\n"
        '  if [[ "$1" == "--data" ]]; then\n'
        "    shift\n"
        '    printf "%s\\n" "$1" >> "$DISCORD_CAPTURE"\n'
        "  fi\n"
        "  shift\n"
        "done\n",
        encoding="utf-8",
    )
    curl.chmod(0o755)

    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "LOCAL_STOP_WHEN_IDLE": "1",
            "LOCAL_IDLE_CHECK_INTERVAL": "1",
            "NEBIUS_INSTANCE_ID_FILE": str(instance_id),
            "NEBIUS_CAPTURE": str(capture),
            "DISCORD_WEBHOOK_URL": "https://discord.invalid/fake-webhook",
            "DISCORD_CAPTURE": str(discord_capture),
            "GPU_QUERY_CAPTURE": str(query_capture),
            "GPU_BUSY_FILE": str(busy),
        }
    )

    result = _run(["--backend", "local", "--local-gpus", "0", str(config)], env)
    state_dir = _state_dir(result.stdout)

    _wait_for_status(state_dir, env, "SUCCEEDED")
    _wait_for_file(query_capture)
    assert not capture.exists()
    busy.unlink()

    _wait_for_file(capture)
    assert capture.read_text(encoding="utf-8").strip() == (
        "compute instance stop --id vm-test-123 --profile gpu-self-stop --async"
    )
    alerts = _wait_for_text(
        discord_capture, "auto-stopping Nebius instance vm-test-123"
    )
    assert "GCW local job finished" in alerts
    assert "test-array" in alerts
    stop_marker = Path(env["LOCAL_STATE_ROOT"]) / "vm-stop-requested"
    _wait_for_file(stop_marker)
    monitor_log = (state_dir / "idle-stop-monitor.out").read_text(
        encoding="utf-8"
    )
    assert "No local workers or GPU compute processes remain" in monitor_log

    blocked = _run(
        ["--backend", "local", "--dry-run", "--local-gpus", "0", str(config)],
        env,
        check=False,
    )
    assert blocked.returncode != 0
    assert "stop has already been requested during this boot" in blocked.stdout


def test_shell_script_has_valid_syntax() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
