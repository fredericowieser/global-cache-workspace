# Reasoning Gym worked-solution SFT

Prepared with `scripts/prepare_reasoning_gym.py`. The generated full dataset is
`data/train-data/reasoning-gym/`; its fixed proportional subset is
`data/train-data/reasoning-gym-128k/`. Both builders support `--replace` for
regeneration at the same paths. The [integration guide](../../docs/integrations/reasoning-gym.md)
contains the reproduction commands, task definitions, admission policy, and
all six training templates. [Example rows](reasoning-gym-examples.jsonl)
show one retained worked solution per task.

| Split | Rows | Bytes | Supervised tokens, including EOS |
|---|---:|---:|---:|
| Train | 700,032 | 1,668,593,259 | 269,025,878 |
| Validation | 6,000 | 14,330,670 | 2,309,576 |
| Test | 6,000 | 14,264,502 | 2,298,698 |

Each of the six tasks contributes 116,672 training rows and 1,000 rows per
holdout: string manipulation, base conversion, AB rewriting, cube rotation,
matrix manipulation, and GCD. The source is Reasoning Gym revision
`49b07130b3fcd12f2d064bba7c43869543a0e7e7`, licensed under Apache-2.0.
The tokenizer is Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, with thinking disabled.

Training sequences average 591.28 tokens, with a maximum of 3,916.
Complete solutions are admitted within 4,096 tokens. Matrix answers must be
nonconstant, have at least six cells and two rows/columns, and come from traces
with at least 75% productive operations. GCD pairs use a controlled common-factor
mixture with a common numeric input range across labels and require at least
four Euclidean divisions. Every base-conversion
digit accumulation and quotient/remainder division is rendered explicitly.

The [preparation receipt](reasoning-gym-preparation.json) records file checksums,
generator and implementation identity, admission policy, task statistics, and
verification results. `scripts/verify_reasoning_gym.py` verifies all 712,032 rows
for checksums, labels, prompt/input deduplication, seed isolation, and recorded
lengths; it regenerates and checks 100 examples per task per split (1,800 total)
with the pinned tokenizer and actual SFT loader.

The fixed 128,000-row subset uses sampling seed 24. Cube rotation and matrix
manipulation contribute 21,334 rows each; the other tasks contribute 21,333 each.
The [subset receipt](reasoning-gym-128k-preparation.json) records membership and
length evidence. At 1,536 tokens, 1,634 rows (1.277%) are truncated, none becomes
prompt-only, and 0.88106% of supervised tokens are removed. The subset templates
use global batch 64 and 2,000 steps for one pass.

All six configurations share the full validation split and evaluate its first
768 examples every 500 steps. The held-out data measure fresh instances of the
same mixture, not new families or harder compositions. No training run or
reasoning-transfer result is implied by data preparation. The shared native
backbone checkpoint and target GPU memory/throughput still need qualification.

[Training truncation and quality findings](../analysis/reasoning-gym-truncation/summary.md)
and [validation findings](../analysis/reasoning-gym-validation/summary.md)
are reproducible using `scripts/audit_reasoning_gym.py`.
