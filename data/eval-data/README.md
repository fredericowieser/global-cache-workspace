# Evaluation dataset registry

This directory records evaluation dataset sources, immutable revisions, preparation rules, licenses, overlap policy, checksums, and intended tasks. It does not store raw datasets in Git.

Evaluation evidence must bind the exact prepared records and task definitions used by the run.

KOR-Bench source and prepared-record hashes are pinned in
`src/gcw/evaluation/korbench_sources.json`. Run `scripts/prepare_korbench.py` to
materialize the ignored `data/eval-data/korbench/` directory. See the
[KOR-Bench integration guide](../../docs/integrations/korbench.md).
