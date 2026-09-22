# Data and tokenization

**Status:** Maintained

## FineWeb pretraining

The preparation pipeline streams FineWeb-Edu sample-10BT, tokenizes documents with Cosmo2, reserves the first 10,000 documents for validation, and writes contiguous `uint16` token shards. Training memory-maps those shards rather than loading the corpus into RAM.

A FineWeb cursor records `(epoch, shard_index, step_in_shard)`, where the step identifies the next logical global microbatch. “Global” is essential: every DDP rank takes a disjoint slice of that batch. Restoring the cursor therefore resumes data coverage rather than replaying rank-local iterators.

## Random smoke data

The random stream exists for deterministic tests and smoke runs. Each rank derives a different reproducible generator seed from the configured seed and rank. Targets use a one-position roll, including a circular final target; this makes a convenient synthetic contract, not a realistic language distribution.

## Supervised fine-tuning

OpenMath preparation normalizes each row to `{"prompt": ..., "answer": ...}`. Batching tokenizes prompt and answer, appends end-of-text, shifts labels for next-token prediction, and masks prompt and padding positions with `-100`. Each DDP rank receives a disjoint slice of a logical global row batch.

The SFT cursor stores `(epoch, next_global_row_offset)`. Restoring it reproduces row order for the same topology. Elastic world-size changes are intentionally unsupported for SFT because the implementation only defines that guarantee for FineWeb.

[Reasoning Gym preparation](../integrations/reasoning-gym.md) produces the same
JSONL interface with complete worked solutions and tagged final answers. It
checks full chat-rendered sequence lengths before admission and freezes balanced,
deduplicated train, validation, and test splits with generator provenance.

## Tokenizer abstraction

GCW supports the Cosmo2 tokenizer used by scratch models, a Hugging Face tokenizer adapter used by converted Llama/Qwen artifacts, and a byte tokenizer for small tests. Evaluation also asks tokenizers for byte lengths so it can convert token NLL into bits per byte.
