# Slurm

**Status:** Maintained

Slurm scripts are an operational layer under `scripts/cluster/slurm/`. They select resources, establish the environment, claim output paths, and invoke package commands. They must not own scientific loss or evaluation logic.

Run GPU work inside an allocation. Export the exact config path and required runtime extras. Capture Slurm job ID, node, GPU model, driver, environment, and command in run provenance.

Historical launch paths can participate in source identity. Move them only with a versioned source-identity migration.
