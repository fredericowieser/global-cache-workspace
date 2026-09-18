# Slurm

**Status:** Maintained

Slurm scripts are an operational layer under `scripts/cluster/slurm/`. They select resources, establish the environment, claim output paths, and invoke package commands. They must not own scientific loss or evaluation logic.

Run GPU work inside an allocation. Export the exact config path and required runtime extras. Capture Slurm job ID, node, GPU model, driver, environment, and command in run provenance.

Historical launch paths can participate in source identity. Move them only with a versioned source-identity migration.

## Fair campaign batch probe

Submit `scripts/cluster/slurm/probe_fair_campaign_batch.sbatch` from the repository root with `sbatch`. The one-GPU job tests each `configs/experiments/*fair_campaign*.yaml` at batch sizes 128, 64, 32, 16, 8, 4, 2, and 1 until three optimizer steps complete. It retains each config's `total_batch_size` and sequence length, disables W&B and checkpoint saves, and writes per-attempt logs plus `report.json` under `slurm_out/fair-campaign-batch-<job-id>/`. The job uses the `gpu` and `qwen35` extras and `expandable_segments:True`.

The probe uses floating-point `cache_processor.gate_init` values in its temporary config copies, as required for a trainable gate tensor. The fair campaign BT and BGT source configs also use floating-point values. Set `GCW_PROBE_CONFIG` to one config filename to run a single config; set `GCW_PROBE_START_REPORT` to an existing report path to start each config at its previously selected batch size.

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
