# Weights & Biases

**Status:** Maintained

GCW mirrors local metric events to W&B. Local run records remain canonical.

`WandbConfig` controls project, entity, run name, group, tags, mode, ID, and resume policy. `resume: must` requires a stable run ID. Resumed training continues after the checkpoint's completed step and must not log that step twice.

W&B failures must not erase local metrics or checkpoint state. The local `wandb.json` record binds the execution ID to the remote run ID and URL. Do not place model weights in the run-log directory merely to upload them as W&B artifacts.
