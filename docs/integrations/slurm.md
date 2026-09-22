# Slurm

**Status:** Maintained

Slurm scripts are an operational layer under `scripts/cluster/slurm/`. They select resources, establish the environment, claim output paths, and invoke package commands. They must not own scientific loss or evaluation logic.

Run GPU work inside an allocation. Export the exact config path and required runtime extras. Capture Slurm job ID, node, GPU model, driver, environment, and command in run provenance.

Historical launch paths can participate in source identity. Move them only with a versioned source-identity migration.

## Training batch probe

`scripts/cluster/probe_batch.py` accepts one or more training YAML
paths. It probes those configs sequentially, in fresh subprocesses, without
modifying the source files.

The search starts at gradient accumulation 1, then considers 2, 3, and so on.
Per-GPU microbatch size is `global_rows / (gpus * grad_accum_steps)`. Values
that do not divide exactly are skipped so the global batch is unchanged. For
128 global rows, one GPU starts at batch 128 and two GPUs start at batch 64.
For 120 rows on two GPUs, the first candidates are 60, 30, and 20.
`global_batch_rows` takes precedence; otherwise rows are derived from
`total_batch_size / sequence_len`. If neither budget is set, the source
microbatch times GPU count defines the fixed global row budget.

The first candidate completing three optimizer steps with finite losses is
reported as `optimal_batch_size`, alongside `optimal_grad_accum_steps`.
`--steps N` changes this qualification length. Only allocation failures trigger
a smaller-batch retry. Other failures stop that config's search and probing
continues with the next config. Any failed config makes the controller exit
nonzero. This checks short-run memory fit, not peak throughput or every later
training batch.

`--gpus N` defaults to one. For multiple GPUs the config must already specify
DDP or FSDP; the probe uses `torchrun --standalone`. Run the controller once,
outside torchrun. The Slurm wrapper requests one GPU and six hours by default; when using
more GPUs, set both the Slurm allocation and `--gpus` accordingly:

```bash
sbatch scripts/cluster/slurm/probe_batch.sbatch \
  configs/experiments/reasoning_gym_128k/llama32_1b_*.yaml

sbatch --gres=gpu:2 scripts/cluster/slurm/probe_batch.sbatch \
  --gpus 2 path/to/ddp-config.yaml
```

To probe the 18 small Reasoning Gym paper configs in separate one-GPU jobs,
allowing Slurm to schedule them in parallel:

```bash
for config in configs/experiments/reasoning_gym_128k/{llama32_1b,llama32_3b,qwen35_2b}_*.yaml; do
  sbatch scripts/cluster/slurm/probe_batch.sbatch "$config"
done
```

Each invocation writes temporary configs, logs, and an incremental `report.json`
under `slurm_out/batch-probe-<job-id>-<pid>/`. `--result-dir PATH` selects a new,
nonexistent directory instead; concurrent invocations must use distinct paths.
Source model, loss, thinking, sequence length, and optimizer settings are
preserved. W&B, evaluation, and checkpoint saves are disabled in temporary
configs. Resuming training is rejected because attempts must start fresh.
Required local SFT datasets must exist before any attempt starts. The job uses
GPU/Qwen3.5 runtime extras and expandable allocator segments.

## Shared worker queues

`scripts/cluster/submit_array.sh --workers N --tasks-file tasks.tsv` accepts mixed
training and evaluation tasks in one queue. Each tab-separated row has six fields:
`ACTION`, `CONFIG`, `CHECKPOINT`, `OUTPUT_BASE`, `THINKING_MODES`, `SKIP_TASKS`.
Use `train` for training followed by evaluation, or `eval` with CONFIG `-` for
evaluation only. Modes are `false`, `true`, or `false,true`; outputs gain
`_thinking-off.json` / `_thinking-on.json` suffixes. SKIP_TASKS `-` evaluates the
full suite; `theoremqa` omits that task. Existing or duplicate output paths are
rejected. Each task owns its evaluation settings, and failures are recorded and
propagated to the worker exit status.

For training only, use `--configs-file` with `EVAL_AFTER_TRAIN=0`. To evaluate its
outputs later, submit an evaluation task queue with `DEPENDENCY=afterok:JOB_ID`.
Only this Slurm success dependency permits missing checkpoints at submission;
workers still require checkpoints when executing tasks. All training workers
must succeed before the dependent array starts. Use fresh queue state for retries.

Set `GPUS` for GPUs per worker and `EVAL_GPUS=1` for single-GPU evaluation.
Leave `CPUS_PER_TASK`, `MEM`, `ARRAY_TIME`, and `PARTITION` unset to use Slurm
defaults; clear inherited `SBATCH_*` resource overrides as well. Worker count
and job dependencies do not prescribe CPU, memory, or time limits.

For KOR-Bench, set `EVAL_SUITE=korbench`, `EVAL_APPLY_CHAT_TEMPLATE=true`
for instruction models, and `EVAL_LOG_SAMPLES=1` to retain responses. Sample logging
is supported for `all`, `math`, `general`, `korbench`, and `tasks`. These options
propagate to both local and Slurm workers. Use `EVAL_UV_EXTRAS="--no-sync"` with a
pre-synced environment to avoid dependency changes while workers run.
