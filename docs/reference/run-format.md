# Run format

**Status:** Experimental schema

Training and evaluation logs use `gcw.run_record.v1`.

```text
data/{training-runs|eval-runs}/YYYY-MM-DD/<experiment>--<execution-id>/
├── config.yaml
├── git_hash.txt
├── environment.json
├── provenance.json
├── metrics.jsonl
├── metrics.json
├── logs.txt
├── status.json
├── wandb.json
└── state.json
```

The directory contains no model weights. `state.json` binds mutable state under `data/runs/`. An execution ID identifies one attempt. An experiment identity identifies scientific inputs and protocol, so several execution IDs can share one experiment identity.

Writers claim the directory atomically. Resume reopens the same execution. Finalization records success or failure without deleting the event stream.
