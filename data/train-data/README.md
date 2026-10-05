# Training dataset registry

This directory records dataset sources, immutable revisions, preprocessing commands, licenses, checksums, and intended training use. It does not store raw datasets in Git.

A run manifest must bind the prepared dataset bytes outside this registry. A README entry alone does not make a run reproducible.

- [Reasoning Gym worked solutions](reasoning-gym.md): six balanced procedural
  tasks, pinned generators, complete SFT targets, and independent fresh-instance
  validation and test splits.
