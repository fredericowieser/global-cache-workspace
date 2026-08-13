# Inspect runs

**Status:** Maintained

A run has two surfaces:

- `data/training-runs/` or `data/eval-runs/` contains human-readable logs, resolved config, metrics, environment, and provenance.
- `data/runs/` contains mutable checkpoint and resume state.

Check `status.json` first. Then compare `provenance.json`, `config.yaml`, and `git_hash.txt`. Read `metrics.jsonl` for the complete event stream and `metrics.json` for final aggregates.

`wandb.json` binds the local execution to its W&B run. Local records remain authoritative if W&B is unavailable.
