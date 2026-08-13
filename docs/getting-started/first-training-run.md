# First training run

**Status:** Maintained

Use the random-data smoke config before downloading a dataset.

```bash
uv run --extra cpu gcw train --config configs/train/smoke_tf.yaml
```

The command validates the config before it constructs the model. It writes mutable checkpoint state to the config's `output_dir`. New run-log contracts place configs, metrics, logs, and provenance under `data/training-runs/`; they keep weights under `data/runs/`.

Do not use a smoke result as scientific evidence. Admit reviewed artifacts through the evidence workflow after validation.
