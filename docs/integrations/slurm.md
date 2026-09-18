# Slurm

**Status:** Maintained

Slurm scripts are an operational layer under `scripts/cluster/slurm/`. They select resources, establish the environment, claim output paths, and invoke package commands. They must not own scientific loss or evaluation logic.

Run GPU work inside an allocation. Export the exact config path and required runtime extras. Capture Slurm job ID, node, GPU model, driver, environment, and command in run provenance.

Historical launch paths can participate in source identity. Move them only with a versioned source-identity migration.

## Fair campaign batch probe

Submit `scripts/cluster/slurm/probe_fair_campaign_batch.sbatch` from the repository root with `sbatch`. The one-GPU job tests each `configs/experiments/*fair_campaign*.yaml` at batch sizes 128, 64, 32, 16, 8, 4, 2, and 1 until three optimizer steps complete. It retains each config's `total_batch_size` and sequence length, disables W&B and checkpoint saves, and writes per-attempt logs plus `report.json` under `slurm_out/fair-campaign-batch-<job-id>/`. The job uses the `gpu` and `qwen35` extras and `expandable_segments:True`.

The probe uses floating-point `cache_processor.gate_init` values in its temporary config copies, as required for a trainable gate tensor. The fair campaign BT and BGT source configs also use floating-point values. Set `GCW_PROBE_CONFIG` to one config filename to run a single config; set `GCW_PROBE_START_REPORT` to an existing report path to start each config at its previously selected batch size.
