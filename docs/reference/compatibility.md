# Compatibility policy

**Status:** Maintained

The maintained API is intentionally small: `gcw.config`, `gcw.models`, `gcw.training`, and `gcw.evaluation`.

The source-layout migration removed physical aliases for the former flat package. Import canonical modules directly:

| Removed path | Canonical path |
|---|---|
| `gcw.train` | `gcw.training.core` |
| `gcw.eval` | `gcw.evaluation.core` |
| `gcw.training_checkpoint` | `gcw.training.checkpoint` |
| `gcw.models.{bt,bgt}` | `gcw.models.model` and `gcw.models.build_model` |
| `gcw.models.{llama,qwen3,gemma4}` | `gcw.models.backbones.*` |
| `gcw.models.{icarus,longshot}` | `gcw.models.cache_processors.*` |

Checkpoint compatibility remains strict. Processor-bearing state dictionaries retain `backbone.*` and `cache_processor.*` ownership. Processor-free backbone checkpoints retain unprefixed keys.
