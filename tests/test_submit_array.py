import os
import re
import shlex
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


def test_slurm_uses_resource_defaults_except_gpus(tmp_path: Path) -> None:
    config, env = _write_fixture(tmp_path)
    for name in ("PARTITION", "ARRAY_TIME", "CPUS_PER_TASK", "MEM"):
        env.pop(name, None)
    env["GPUS"] = "2"

    result = _run(["--backend", "slurm", "--dry-run", str(config)], env)
    command = shlex.split(result.stdout.splitlines()[-1])

    assert "--gres=gpu:2" in command
    assert "--array=0-0" in command
    for prefix in (
        "--partition=",
        "--time=",
        "--nodes=",
        "--ntasks-per-node=",
        "--cpus-per-task=",
        "--mem=",
    ):
        assert not any(arg.startswith(prefix) for arg in command)


def test_slurm_passes_ephemeral_environment_file_without_login_env_retrieval(
    tmp_path: Path,
) -> None:
    config, env = _write_fixture(tmp_path)
    state_root = tmp_path / "slurm-state"
    capture = tmp_path / "captured-environment"
    bin_dir = Path(env["HOME"]) / ".local" / "bin"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "found_export=0\n"
        "for arg in \"$@\"; do\n"
        "  if [[ \"$arg\" == --export=ALL ]]; then\n"
        "    found_export=1\n"
        "  fi\n"
        "done\n"
        "[[ \"$found_export\" == 1 ]] || exit 1\n"
        "env -0 > \"$SBATCH_EXPORT_CAPTURE\"\n"
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


def test_eval_only_runs_multiple_thinking_modes_in_one_queue(
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
            "EVAL_CHECKPOINT_STEP": "2000",
            "EVAL_THINKING_MODES": "false,true",
            "UV_CAPTURE": str(capture),
        }
    )

    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", str(config)],
        env,
    )
    state_dir = _state_dir(result.stdout)
    _wait_for_status(state_dir, env, "SUCCEEDED")

    invocations = [
        line
        for line in capture.read_text(encoding="utf-8").splitlines()
        if " gcw eval " in f" {line} "
    ]
    assert len(invocations) == 2
    assert "--enable-thinking false" in invocations[0]
    assert "step2000_paper7_thinking-off.json" in invocations[0]
    assert "--enable-thinking true" in invocations[1]
    assert "step2000_paper7_thinking-on.json" in invocations[1]


def test_eval_tasks_file_runs_heterogeneous_checkpoints_and_thinking_modes(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "uv-args"
    _, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"',
    )
    checkpoint_1000 = tmp_path / "run-a" / "checkpoint-1000"
    checkpoint_2000 = tmp_path / "run-b" / "checkpoint-2000"
    checkpoint_1000.mkdir(parents=True)
    checkpoint_2000.mkdir(parents=True)
    out_1000 = tmp_path / "results" / "run-a-corrected.json"
    out_2000 = tmp_path / "results" / "run-b-thinking.json"
    tasks = tmp_path / "eval-tasks.tsv"
    tasks.write_text(
        f"# checkpoint\\toutput\\tthinking\n"
        f"{checkpoint_1000}\t{out_1000}\tfalse\n"
        f"{checkpoint_2000}\t{out_2000}\ttrue\n",
        encoding="utf-8",
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
        [
            "--backend",
            "local",
            "--local-gpus",
            "GPU-test",
            "--eval-tasks-file",
            str(tasks),
        ],
        env,
    )
    state_dir = _state_dir(result.stdout)
    _wait_for_status(state_dir, env, "SUCCEEDED")

    invocations = [
        line
        for line in capture.read_text(encoding="utf-8").splitlines()
        if " gcw eval " in f" {line} "
    ]
    assert len(invocations) == 2
    assert f"--checkpoint {checkpoint_1000}" in invocations[0]
    assert f"--out {out_1000}" in invocations[0]
    assert "--enable-thinking false" in invocations[0]
    assert f"--checkpoint {checkpoint_2000}" in invocations[1]
    assert f"--out {out_2000}" in invocations[1]
    assert "--enable-thinking true" in invocations[1]
    assert all("--batch-size auto" in invocation for invocation in invocations)
    assert all("--max-batch-size 32" in invocation for invocation in invocations)
    assert (state_dir / "configs.txt").read_text(encoding="utf-8") == (
        f"{checkpoint_1000}\t{out_1000}\tfalse\n"
        f"{checkpoint_2000}\t{out_2000}\ttrue\n"
    )


def test_eval_tasks_file_rejects_global_checkpoint_or_thinking_overrides(
    tmp_path: Path,
) -> None:
    _, env = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint-2000"
    checkpoint.mkdir()
    tasks = tmp_path / "eval-tasks.tsv"
    tasks.write_text(
        f"{checkpoint}\t{tmp_path / 'result.json'}\ttrue\n",
        encoding="utf-8",
    )
    env.update(
        {
            "EVAL_ONLY": "1",
            "EVAL_CHECKPOINT_STEP": "2000",
        }
    )

    result = _run(
        [
            "--backend",
            "local",
            "--dry-run",
            "--local-gpus",
            "GPU-test",
            "--eval-tasks-file",
            str(tasks),
        ],
        env,
        check=False,
    )

    assert result.returncode != 0
    assert "cannot be combined with EVAL_CHECKPOINT_STEP" in result.stdout


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


def test_mixed_tasks_keep_training_modes_and_skip_lists_per_row(tmp_path: Path) -> None:
    capture = tmp_path / "uv-args"
    config, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s|%s\\n" "$GCW_EVAL_SKIP_TASKS" "$*" >> "$UV_CAPTURE"',
    )
    checkpoint = tmp_path / "existing checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text(
        f"train\t{config}\t{tmp_path / 'output/checkpoint'}\t"
        f"{tmp_path / 'trained.json'}\tfalse\t-\n"
        f"eval\t-\t{checkpoint}\t{tmp_path / 'eval.json'}\ttrue\ttheoremqa\n"
        f"eval\t-\t{checkpoint}\t{tmp_path / 'both.json'}\tfalse,true\t-\n"
    )
    env.update({"UV_CAPTURE": str(capture), "EVAL_THINKING_MODES": "true"})
    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", "--tasks-file", str(tasks)],
        env,
    )
    state = _state_dir(result.stdout)
    _wait_for_status(state, env, "SUCCEEDED")
    lines = capture.read_text().splitlines()
    assert sum(" gcw train " in line for line in lines) == 1
    evaluations = [line for line in lines if " gcw eval " in line]
    assert len(evaluations) == 4
    assert evaluations[0].startswith("|")
    assert "--enable-thinking false" in evaluations[0]
    assert "trained_thinking-off.json" in evaluations[0]
    assert evaluations[1].startswith("theoremqa|")
    assert "--enable-thinking true" in evaluations[1]
    assert f"--checkpoint {checkpoint}" in evaluations[1]
    assert all(line.startswith("|") for line in evaluations[2:])
    assert "--enable-thinking false" in evaluations[2]
    assert "--enable-thinking true" in evaluations[3]
    assert len((state / "completed_configs").read_text().splitlines()) == 3


def test_mixed_tasks_propagate_failure_and_continue_next_task(tmp_path: Path) -> None:
    capture = tmp_path / "uv-args"
    config, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"\n'
        '[[ "$*" != *" gcw train "* ]] || exit 17\n'
        '[[ "$*" != *"bad_thinking-off.json"* ]] || exit 23',
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text(
        f"train\t{config}\t{checkpoint}\t{tmp_path / 'train.json'}\tfalse\t-\n"
        f"eval\t-\t{checkpoint}\t{tmp_path / 'bad.json'}\tfalse,true\t-\n"
        f"eval\t-\t{checkpoint}\t{tmp_path / 'good.json'}\tfalse\t-\n"
    )
    env["UV_CAPTURE"] = str(capture)
    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", "--tasks-file", str(tasks)],
        env,
    )
    state = _state_dir(result.stdout)
    _wait_for_status(state, env, "FAILED")
    text = capture.read_text()
    assert "train_thinking-off.json" not in text
    assert "bad_thinking-on.json" not in text
    assert "good_thinking-off.json" in text
    failures = (state / "failed_configs.tsv").read_text().splitlines()
    assert failures[0].startswith("17\ttrain\t")
    assert failures[1].startswith("23\teval\t")
    assert len((state / "completed_configs").read_text().splitlines()) == 1


def test_mixed_tasks_reject_invalid_or_colliding_work(tmp_path: Path) -> None:
    config, env = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    out = tmp_path / "result.json"
    row = f"eval\t-\t{checkpoint}\t{out}\tfalse\t-\n"
    tasks = tmp_path / "tasks.tsv"
    invalid = [
        row + row,
        row.replace("\tfalse\t", "\tfalse,false\t"),
        row.replace("eval\t", "unknown\t"),
        row.replace("\t-\n", "\t\n"),
        row.replace(str(checkpoint), str(tmp_path / "missing")),
        f"train\t{config}\t{checkpoint}\t{out}\tfalse\t-\n"
        f"train\t{config}\t{checkpoint}\t{tmp_path / 'other.json'}\tfalse\t-\n",
    ]
    for content in invalid:
        tasks.write_text(content)
        result = _run(
            ["--backend", "slurm", "--dry-run", "--tasks-file", str(tasks)],
            env,
            check=False,
        )
        assert result.returncode != 0, content
    (tmp_path / "result_thinking-off.json").write_text("{}")
    tasks.write_text(row)
    result = _run(
        ["--backend", "slurm", "--dry-run", "--tasks-file", str(tasks)],
        env,
        check=False,
    )
    assert result.returncode != 0
    assert "Existing or duplicate evaluation output" in result.stdout


def test_mixed_tasks_slurm_export_preserves_dispatch_mode(tmp_path: Path) -> None:
    _, env = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text(
        f"eval\t-\t{checkpoint}\t{tmp_path / 'result.json'}\tfalse\t-\n"
    )
    bin_dir = Path(env["HOME"]) / ".local/bin"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        '#!/usr/bin/env bash\n'
        '[[ "$GCW_MIXED_TASK_MODE" == 1 ]] || exit 31\n'
        '[[ "$EVAL_AFTER_TRAIN" == 1 ]] || exit 32\n'
        "printf '999\\n'\n"
    )
    sbatch.chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    result = _run(["--backend", "slurm", "--tasks-file", str(tasks)], env)
    assert "Submitted array job 999" in result.stdout


def test_mixed_tasks_multiple_workers_claim_each_row_once(tmp_path: Path) -> None:
    capture = tmp_path / "uv-args"
    _, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"\nsleep 0.1',
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text("".join(
        f"eval\t-\t{checkpoint}\t{tmp_path / f'result-{i}.json'}\tfalse\t-\n"
        for i in range(4)
    ))
    env["UV_CAPTURE"] = str(capture)
    result = _run(
        ["--backend", "local", "--workers", "2", "--local-gpus", "0,1",
         "--tasks-file", str(tasks)],
        env,
    )
    state = _state_dir(result.stdout)
    _wait_for_status(state, env, "2 succeeded")
    completed = (state / "completed_configs").read_text().splitlines()
    assert len(completed) == len(set(completed)) == 4
    assert sum(" gcw eval " in line for line in capture.read_text().splitlines()) == 4


def test_mixed_tasks_defer_checkpoint_validation_only_for_afterok(tmp_path: Path) -> None:
    _, env = _write_fixture(tmp_path)
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text(
        f"eval\t-\t{tmp_path / 'future/checkpoint-2000'}\t"
        f"{tmp_path / 'result.json'}\tfalse,true\ttheoremqa\n"
    )
    for dependency, accepted in [("afterok:123", True), ("afterany:123", False), ("", False)]:
        env["DEPENDENCY"] = dependency
        result = _run(
            ["--backend", "slurm", "--dry-run", "--tasks-file", str(tasks)],
            env,
            check=False,
        )
        assert (result.returncode == 0) == accepted, result.stdout
        if accepted:
            assert "--dependency=afterok:123" in result.stdout

    env.update({
        "GCW_ARRAY_MODE": "worker",
        "GCW_MIXED_TASK_MODE": "1",
        "CONFIG_LIST": str(tasks),
        "STATE_DIR": str(tmp_path),
    })
    result = _run([], env, check=False)
    assert result.returncode != 0
    assert "future/checkpoint-2000" in result.stdout
    assert (tmp_path / "failed_configs.tsv").exists()


def test_korbench_workers_log_samples_with_ordered_thinking_modes(tmp_path):
    capture = tmp_path / "uv-args"
    _, env = _write_fixture(
        tmp_path, uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"'
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "tasks.tsv"
    tasks.write_text(
        f"eval\t-\t{checkpoint}\t{tmp_path / 'korbench.json'}\tfalse,true\t-\n"
    )
    env.update({
        "EVAL_SUITE": "korbench",
        "EVAL_LOG_SAMPLES": "1",
        "EVAL_APPLY_CHAT_TEMPLATE": "true",
        "EVAL_UV_EXTRAS": "--no-sync",
        "UV_CAPTURE": str(capture),
    })
    result = _run(
        ["--backend", "local", "--local-gpus", "GPU-test", "--tasks-file", str(tasks)],
        env,
    )
    _wait_for_status(_state_dir(result.stdout), env, "SUCCEEDED")
    invocations = [
        shlex.split(line)
        for line in capture.read_text().splitlines()
        if " gcw eval " in f" {line} "
    ]
    assert len(invocations) == 2
    for args, mode in zip(invocations, ["false", "true"], strict=True):
        assert args[args.index("eval") + 1] == "korbench"
        assert "--tasks" not in args
        assert args[args.index("--apply-chat-template") + 1] == "true"
        assert args[args.index("--enable-thinking") + 1] == mode
        assert "--log-samples" in args
        assert "--no-sync" in args


def test_custom_benchmark_submission_rejects_invalid_options(tmp_path):
    config, env = _write_fixture(tmp_path)
    for options, message in [
        ({"EVAL_LOG_SAMPLES": "1"}, "not supported for paper7"),
        ({"EVAL_LOG_SAMPLES": "true"}, "must be 0 or 1"),
        ({"EVAL_APPLY_CHAT_TEMPLATE": "1"}, "must be true or false"),
    ]:
        result = _run(
            ["--backend", "slurm", "--dry-run", str(config)],
            {**env, **options}, check=False,
        )
        assert result.returncode != 0
        assert message in result.stdout


def test_attach_eval_queue_preserves_existing_work(tmp_path: Path) -> None:
    _, env = _write_fixture(tmp_path)
    state = tmp_path / "existing"
    state.mkdir()
    row = "/checkpoint\t/already-completed.json\tfalse\n"
    contents = {"configs.txt": row, "claimed_configs": row, "queue.lock": "",
                "completed_configs": row}
    for name, value in contents.items():
        (state / name).write_text(value)
    env["EVAL_ONLY"] = "1"
    result = _run(["--backend", "slurm", "--dry-run", "--workers", "4",
                   "--attach-eval-queue", str(state)], env)
    assert f"Config list: {state}/configs.txt" in result.stdout
    assert "--array=0-3" in result.stdout
    for name, value in contents.items():
        assert (state / name).read_text() == value
    assert not list(state.glob("slurm-environment*"))
    rejected = _run(["--backend", "slurm", "--dry-run",
                     "--attach-eval-queue", str(state), "extra.yaml"], env, check=False)
    assert rejected.returncode != 0
    (state / "configs.txt").write_text("training.yaml\n")
    rejected = _run(["--backend", "slurm", "--dry-run",
                     "--attach-eval-queue", str(state)], env, check=False)
    assert rejected.returncode != 0


def test_eval_tasks_support_per_row_suites(tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    _, env = _write_fixture(
        tmp_path,
        uv_body='printf "%s\\n" "$*" >> "$UV_CAPTURE"',
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "eval.tsv"
    rows = [
        f"{checkpoint}\t{tmp_path / 'paper7-off.json'}\tfalse\tpaper7",
        f"{checkpoint}\t{tmp_path / 'korbench-off.json'}\tfalse\tkorbench",
        f"{checkpoint}\t{tmp_path / 'paper7-on.json'}\ttrue\tpaper7",
        f"{checkpoint}\t{tmp_path / 'korbench-on.json'}\ttrue\tkorbench",
    ]
    tasks.write_text("\n".join(rows) + "\n")
    env.update(EVAL_ONLY="1", UV_CAPTURE=str(capture))
    result = _run(
        [
            "--backend",
            "local",
            "--local-gpus",
            "0",
            "--eval-tasks-file",
            str(tasks),
        ],
        env,
    )
    state = _state_dir(result.stdout)
    _wait_for_status(state, env, "1 succeeded")
    calls = [
        line
        for line in capture.read_text().splitlines()
        if " gcw eval " in f" {line} "
    ]
    assert len(calls) == 4
    for call, row in zip(calls, rows, strict=True):
        _, out, thinking, suite = row.split("\t")
        assert f"eval {suite}" in call
        assert f"--enable-thinking {thinking}" in call
        assert f"--out {out}" in call
    assert (state / "completed_configs").read_text().splitlines() == rows


def test_eval_tasks_reject_unknown_per_row_suite(tmp_path: Path) -> None:
    _, env = _write_fixture(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tasks = tmp_path / "eval.tsv"
    tasks.write_text(
        f"{checkpoint}\t{tmp_path / 'out.json'}\tfalse\tmisspelled\n"
    )
    env["EVAL_ONLY"] = "1"
    result = _run(
        ["--backend", "slurm", "--dry-run", "--eval-tasks-file", str(tasks)],
        env,
        check=False,
    )
    assert result.returncode != 0
    assert "Unsupported evaluation task suite" in result.stdout
