# Data and tokenization

**Status:** Maintained

## FineWeb pretraining

The preparation pipeline streams `HuggingFaceFW/fineweb-edu`, subset `sample-10BT`, in source order. It reserves the first 10,000 documents for validation by default and appends EOS to each plain-text document. Training memory-maps packed token shards rather than loading the corpus into RAM. Sequences can cross document boundaries; EOS separates documents but does not reset attention. Every next-token target contributes to loss, including EOS. Incomplete global microbatches at shard tails are skipped.

For instruction-tuned checkpoints that need an assistant prefix, set `data.fineweb_prefix` explicitly. The loader tokenizes this string once per stream and prepends it to every sampled row in training and validation, without modifying saved shards or inserting it at internal document boundaries. `sequence_len` still counts corpus input tokens; the actual input length grows by the prefix length and must fit `model.sequence_len`. Prefix positions receive ignored targets (`-100`), preserving the original next-token targets, supervised-token budget, rank partitioning, and cursor coverage. Resumable checkpoints record prefix token IDs in the data identity, and standalone FineWeb evaluation inherits them. The Gemma4-31B FineWeb configs use `<bos><|turn>model\n<|channel>thought\n<channel|>`; this is an assistant prefix with an empty thought block, not a full conversation.

Cosmo2 remains the default tokenizer. For a pretrained backbone, prepare a separate dataset using its matching fast Hugging Face tokenizer:

```bash
gcw prepare fineweb --out /path/to/fineweb-tokenized \
  --tokenizer hf:/path/to/backbone-tokenizer \
  --val-docs 10000 --shard-tokens 100000000
```

`--limit` caps the total documents consumed, including validation documents. Use a limit greater than `--val-docs` to produce training data. The output directory must be empty. Interrupted preparation leaves an incomplete marker and cannot be consumed; retry in a fresh location. A tokenizer without an explicit EOS is rejected.

For an exact token budget, add `--train-tokens 20000000`. This counts training tokens including document EOS separators, independently of the held-out validation documents. Preparation stops at the exact count, truncating the final document's encoded stream if necessary; the manifest records how many tokens were omitted. If the source or `--limit` runs out before the requested count, preparation fails and leaves the incomplete marker. `--shard-tokens` controls individual file sizes, not the total dataset size. Use `--dataset-revision <commit>` to pin the source revision across tokenizers.

To size a single-file dataset for exactly `N` optimizer steps without replay, store `N × global_sequences_per_step × sequence_len + 1` tokens and set `--shard-tokens` to at least that count. The extra token supplies the last next-token target. Gradient accumulation contributes to the global sequence count; it does not change this token budget. Multiple files can each discard an incomplete global microbatch at their boundary.

Use `--shuffle-seed <nonnegative integer>` when selecting a representative subset. The published `sample-10BT` documents are ordered in crawl-specific blocks, so a source-order prefix can have uneven crawl coverage. This option caches the complete source subset as a memory-mapped Hugging Face dataset and applies a seeded global document permutation without replacement before validation selection, tokenization, or `--limit`. The first `--val-docs` shuffled rows become validation and subsequent rows supply training tokens. This makes the splits disjoint by source row; it does not deduplicate identical text across different source rows. Only the selected documents are tokenized. The manifest records the sampling seed, source document count, and crawl counts for each split. Set `HF_DATASETS_CACHE` and `HF_HUB_CACHE` to local storage with space for the source Parquet files, Arrow cache, and shuffle index. This is a global shuffle, rather than a bounded streaming-buffer shuffle. Without this option, preparation retains source-order streaming.

Preparation selects little-endian `uint16` or `uint32` storage according to the tokenizer's maximum ID. `manifest.json` records source, preparation parameters, tokenizer content identity and EOS, dtype, and ordered shard inventories with counts and SHA-256 checksums. Readers validate the inventory and byte lengths; they do not rehash the entire corpus at each startup. The manifest digest participates in exact-resume identity. Treat published shards as immutable.

The tokenizer identity includes vocabulary and tokenization rules, including normalization, but excludes its filesystem location and chat formatting. Training checks it against `data.tokenizer`/`data.tokenizer_source` and checks the model vocabulary capacity. Existing manifest-free shards retain the historical Cosmo2 `uint16` interpretation and resume identity; they cannot be selected with an HF tokenizer. Their token range remains the responsibility of the original artifact producer.

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

GCW supports the Cosmo2 tokenizer used by scratch models, a Hugging Face tokenizer adapter used by converted backbones, and a byte tokenizer for small tests. FineWeb evaluation reports NLL and perplexity for compatible tokenizers, with the historical bits-per-byte metric additionally available for Cosmo2.
