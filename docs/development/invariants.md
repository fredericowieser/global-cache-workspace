# Operational invariants and sharp edges

**Status:** Maintained

These constraints deserve explicit attention because a plausible-looking run can otherwise answer the wrong question.

- The repository implements causal language modeling, not JEPA. Do not label cache-processing results as JEPA results.
- Pick exactly one of `cpu`, `gpu`, or `rocm`. CPU resolution should contain no NVIDIA packages; GPU resolution targets CUDA 12.8 and excludes CUDA 13.
- `configs/model/smollm360m_icarus_per_cache_head.yaml` is empty. It is a placeholder, not a runnable model definition.
- The `smollm360m_matched_icarus_20b` and `..._pe512` train files use 3,000 steps at 524,288 tokens per step, about 1.57 billion tokens, while their metadata claims about 20 billion. The executable budget and label disagree.
- FineWeb training identity uses names and byte sizes rather than file-content hashes. Never mutate prepared shards in place.
- FineWeb NLL/BPB uses the supported per-token loss contract and is covered by the model/evaluation CPU gates; network-dependent shard preparation and large-scale metric runs remain separate claims.
- Qwen sliding attention has no native equivalent here and fails conversion rather than silently changing semantics.
- Exact resume accepts only version-2 training checkpoints. Model-only artifacts cannot reconstruct optimizer, cursor, or random state.
- Elastic resume guarantees deterministic repartition and coverage, not bitwise equivalence across world sizes.
- Static generation and reference generation must remain numerically aligned. A speed improvement that changes stop, cadence, or truncation semantics is a correctness regression.
- `docs/lrc-archive` remains reachable only through integrated Git history. It does not belong in the active tree.
