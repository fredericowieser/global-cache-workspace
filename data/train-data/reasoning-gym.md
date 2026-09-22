# Reasoning Gym worked-solution SFT

Prepared locally on 2026-09-22 using
`scripts/prepare_reasoning_gym.py`. The generated files live in the Git-ignored
`data/train-data/reasoning-gym/` directory. The
[integration guide](../../docs/integrations/reasoning-gym.md) contains the
reproduction command, format, filtering policy, and training templates.
For a small reviewable sample, see
[one retained example per task](reasoning-gym-examples.jsonl).

A fixed stratified 128,000-row subset is also prepared at
`data/train-data/reasoning-gym-128k/train.jsonl`, using sampling seed 24.
See the [subset receipt](reasoning-gym-128k-preparation.json) and the
[128k setting](../../docs/integrations/reasoning-gym.md#fixed-128000-example-setting)
for reproduction, task quotas, training configurations, and the 1,536-token audit.

| Split | Rows | Bytes | Supervised tokens, including end tokens |
|---|---:|---:|---:|
| Training | 700,032 | 1,866,639,403 | 270,851,744 |
| Validation | 6,000 | 16,040,994 | 2,333,160 |
| Test | 6,000 | 15,961,284 | 2,312,944 |

All six tasks have equal retained counts: 116,672 training examples and 1,000
examples per holdout. Training has mean complete sequence length 655.63 tokens
and maximum 3,916, below the 4,096-token admission cap. Solutions were never
truncated. The training size is approximately the same number of examples as
the documented 697,744-row math experiment; it is not a matched token budget.

The source is Reasoning Gym revision
`49b07130b3fcd12f2d064bba7c43869543a0e7e7`, licensed under Apache-2.0.
The tokenizer is Qwen/Qwen3.5-2B revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, with thinking disabled. The private
generator subset retains the upstream license and third-party notice.

[Preparation receipt](reasoning-gym-preparation.json) records file checksums,
tokenizer file hashes, generator/implementation identity, per-task admission
statistics, and audit results. The full generated `manifest.json` also retains
the task configurations and source-file hashes. All 712,032 rows were audited
for counts, answer labels, unique prompts and task inputs, nonoverlapping
effective seeds, and complete sequence lengths as recorded at admission. File
and ordered source-ID checksums were recomputed independently. A further 100
examples per task per split (1,800 total) were regenerated and checked through
the actual tokenizer and `sft_batch`, including prompt masking and the end token.

The mixture excludes empty answers, traces with fewer than two productive
steps, generation beyond the transition bound, duplicate inputs, and sequences
over the context cap. The training build excluded 35,616 duplicate inputs,
33,130 insufficient-depth examples, 257 empty outputs, four overlong sequences,
and one bounded string-generation failure. Extra candidates after a task quota
was filled are counted separately in the receipt.

The held-out splits are fresh instances of the same six-task distribution.
Three tasks are in the rewriting family. These data do not establish family
generalisation, harder-instance performance, or measured reasoning transfer.
No experiment training was launched. The three training configurations pass
configuration and capability admission but still require a shared native
backbone checkpoint and GPU qualification.

All 53 focused pipeline, configuration, existing loader, and SFT compatibility
tests pass, as do lint, formatting, and the warnings-as-errors documentation
build. A broader run of existing
training/checkpoint tests encountered Windows filesystem permission and
checkpoint file-descriptor errors outside the modified data pipeline.
