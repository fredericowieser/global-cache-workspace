# Reasoning Gym supervised data

The offline preparation pipeline creates a frozen, balanced mixture of six
Reasoning Gym tasks with deterministic worked solutions and labelled final
answers. It uses the existing JSONL SFT loader and answer-only loss. It does not
require teacher inference or reinforcement learning.

The default training size is **700,032 examples**, close to the 697,744-row math
experiment recorded in [Qwen math adaptation](../experiments/qwen3-math-adaptation.md).
This matches approximate example count, not supervised token count. Validation
and test each contain 6,000 examples. Each training task contributes 116,672
rows; each holdout task contributes 1,000. The training count is divisible by
both six tasks and the example configurations' global batch of 64.

## Prepare

Install GCW with a CPU runtime and the tokenizer extra:

```bash
uv sync --extra cpu --extra convert --group dev
```

Download only the pinned tokenizer files used by the Qwen3.5-2B configurations:

```bash
uv run --extra cpu --extra convert python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-2B', revision='15852e8c16360a2fea060d615a32b45270f8a8fc', allow_patterns=['tokenizer*', 'chat_template*', 'config.json', 'special_tokens_map.json'], local_dir='artifacts/tokenizers/Qwen3.5-2B')"
```

Generate the full dataset:

```bash
uv run --extra cpu --extra convert python scripts/prepare_reasoning_gym.py \
  --tokenizer artifacts/tokenizers/Qwen3.5-2B \
  --out-dir data/train-data/reasoning-gym \
  --workers 4
```

For a small check, use a different output directory and add
`--train-rows 6000 --validation-rows 600 --test-rows 600`.
Preparation works on Windows and Linux. The six pinned generators and their
small shared dependency subset are bundled privately in
`src/gcw/data/_vendor/reasoning_gym/`. NumPy is already a GCW dependency.
Installing the full Reasoning Gym package is unnecessary; this also avoids
unrelated native puzzle dependencies such as `pycosat` on Windows.

The upstream revision is
[`49b07130b3fcd12f2d064bba7c43869543a0e7e7`](https://github.com/open-thought/reasoning-gym/tree/49b07130b3fcd12f2d064bba7c43869543a0e7e7).
`SOURCES.json` records original and adapted file hashes and the import-only
adaptations and removal of unused APIs. Generator, transition, and direct scorer
logic is preserved. The upstream Apache-2.0 license and attribution notice ship
with the subset. The GCW wrapper bounds string rewriting at 256 transitions.
The source manifest is itself hash-pinned, including hashes of package wiring,
license, and notice files. Implementation and vendor files use LF line endings
for portable hashes. Existing receipts retain the implementation hashes from
their original build; later publication and audit fixes do not rewrite history.

## Mixture and labels

| Task | Generator settings | Worked solution |
|---|---|---|
| `string_manipulation` | String length 8 to 28; 6 to 12 sampled rules | First applicable rule, its position in the displayed list, and each new string; explicit termination |
| `string_splitting` | Each initial machine count 0 to 64; at most 256 applications | Priority-ordered dismantling; consecutive applications of one rule are grouped with per-application deltas and resulting counts |
| `ab` | 18 input tokens | Leftmost applicable pair, replacement, and resulting program |
| `color_cube_rotation` | 2 to 8 rotations | All six face colours after each rotation, followed by the requested face |
| `manipulate_matrix` | 3 to 6 initial rows and columns; 3 to 8 operations | Each operation and its resulting matrix |
| `gcd` | 2 to 4 integers between 10 and 100,000 | Euclidean divisions, pairwise GCD reductions, and final GCD |

Every admitted record has `prompt`, `answer`, and `final_answer`. The `answer`
is the entire SFT target:

```text
Worked solution:
...operations and intermediate states...

Final answer:
<answer>...result...</answer>
```

`extract_final_answer` in `gcw.data.reasoning_gym_traces` extracts the tagged
answer, including multiline matrices, and rejects multiple answer blocks.
The ordinary worked solution is supervised with thinking disabled. Upstream
final-answer-only wording is reconciled with the worked-solution instruction.
Prompts clarify leftmost AB rewrites, clockwise matrix rotations, mirror axes,
the right-middle character of even strings, and cube rotation axes. In
particular, bottom-to-top rotation swaps front/back and top/bottom while leaving
left/right fixed.

Trace construction replays prompt-visible inputs and operations. It does not
copy hidden `solution`, `result`, or `states` metadata into explanations. Every
reconstructed final answer must exactly equal the generator answer and receive
full direct upstream reward; disagreement aborts preparation. Tests also check
hand-computed transitions, because agreement with a reused upstream operation
alone is not an independent proof of its semantics.

## Admission, splits, and provenance

Admission requires at least two productive transitions (Euclidean divisions for
GCD). Matrix operations that leave the matrix unchanged do not count towards
this minimum, although their intermediate state remains in the trace. Empty
final answers, including completely cancelled AB programs, are explicitly
excluded. Repeated string states are discarded according to the prompt; a
dismantling iteration limit returns the current counts. Bounded string generation
failures are counted as exclusions.

Complete chat-rendered prompts and targets, including the end token, must fit
4,096 tokens. No solution is truncated. Token counting uses the same prompt
template, thinking setting, separate answer encoding, and end token as
`sft_batch`. Changing tokenizer, prompt suffix, response prefix, chat policy,
or using a shorter context requires preparing and auditing the data again.

For task ordinal `t` and split ordinal `s` (`train`, `validation`, `test`), the
generator seed is `seed + (3*t + s)*100000000`. Candidate index `i` therefore
uses effective seed `generator_seed + i`. Generation cannot exhaust the reserved
range. The defaults use seed 24 and at most 20 attempts per requested row per
task. Both normalized full prompts and structured task inputs are deduplicated
globally across all splits; this also catches cube stories with different
wording but the same inputs and rotations. GCD input order remains part of the
problem identity. Each split is shuffled deterministically after admission.
Worker count does not change the generated bytes. Keep batch size fixed for
identical attempt accounting and output order.

The output directory contains `train.jsonl`, `validation.jsonl`, `test.jsonl`,
and `manifest.json`. Rows retain task, family, source identity, seed, index,
difficulty, trace depth, termination, token counts, and prompt/input hashes.
The manifest records generator configurations, source and implementation hashes,
tokenizer identity, runtime versions, requested and retained counts, all exclusion
counts, difficulty histograms, length/depth statistics, and JSONL checksums.
Effective seed intervals in the report are half-open: `[start, end)`.

Preparation refuses an existing destination. Files are staged and the completed
directory is published only after every quota is met. Failures leave a sibling
`<output-name>.failed.json` diagnostic, rather than an apparently complete
undersized dataset. Generated data and temporary staging directories are ignored
by Git; the registry contains the small preparation receipt.
Staging uses an ordinary sibling directory, so published directories inherit
destination-parent permissions, subject to the operating system's ACL and umask.
On a shared cluster, prepare beneath the intended shared dataset parent.

Validation and test measure **fresh instances from the same task distribution**.
They are not harder-instance, composition, or family holdouts. Three tasks share
the rewriting family, so half of the balanced mixture is rewriting. KOR-Bench
is not used for generation, filtering, mixture tuning, or checkpoint selection.
No transfer advantage or learning result is established by data preparation.

## Training configurations and checks

### Fixed 128,000-example setting

Create the subset once from the prepared full training split:

```bash
uv run --extra cpu python scripts/subset_reasoning_gym.py
```

This writes `data/train-data/reasoning-gym-128k/train.jsonl` and
`manifest.json`, refusing an existing destination. It verifies the original
training file against its manifest, samples uniformly without replacement
within each task using seed 24, and shuffles the selected rows once. Task quotas
are proportional to the source counts, rounded by largest remainder with seeded
tie-breaking. GCD and string manipulation each receive 21,334 rows; each other
task receives 21,333. No length or difficulty filtering is applied.

All configurations use the same frozen file. Changing the training seed does
not change membership. Records retain their original `source_id` and
`experiment_order`; the latter refers to the full training file, while physical
line order determines subset order. The manifest records source and subset
checksums, the sampling policy, ordered source-ID checksum, tokenizer identity,
task counts, and truncation statistics. The small
[subset receipt](../../data/train-data/reasoning-gym-128k-preparation.json)
is retained outside the ignored generated directory.
The optional `--sequence-len` argument sets the receipt's audit cap (default
1,536). It does not filter, truncate, or change selected rows. Match it to the
training configuration when auditing another context length.

Use `configs/train/reasoning_gym/full_sft_128k.yaml`, `lora_128k.yaml`, or
`longshot_128k.yaml`. These use the same original validation/test splits and
starting backbone as the full-data setting, with a 1,536-token training cap,
microbatch size 16, global batch of 64 rows, and 2,000 optimizer steps for one
pass. Output directories are separate from the full-data runs. The original
4,096-token configurations remain available.

Microbatch size is per rank. Keep `global_batch_rows` at least
`batch_size * world_size` and divisible by that product. Adjust `max_steps`
to `128000 / global_batch_rows` when changing the global batch for a one-pass
run, choosing a divisor of 128,000. For example, global batch 128 uses 1,000
steps. Actual GPU memory fit and throughput still require qualification.

Stored solutions remain complete. The loader truncates during training:
1,527 rows (1.19297%) exceed 1,536 tokens, comprising 833 string-manipulation
and 694 AB examples. Among truncated rows, original length averages 1,808.52
tokens and removed length averages 272.52 tokens; both have population standard
deviation 262.37. Four rows lose only EOS. No row becomes prompt-only, and
0.84142% of target tokens are removed. Counts use the authenticated admission
token lengths for the pinned tokenizer and existing chat settings.

### Full training split

`configs/train/reasoning_gym/` supplies `full_sft.yaml`, `lora.yaml`, and
`longshot.yaml`. All use the same dataset, tokenizer, Qwen3.5-2B starting
checkpoint path, seed, 4,096-token cap, global batch of 64, and 10,938 optimizer
steps (one pass through the default training rows). LoRA and Longshot reuse
existing approximately 22-million-parameter configurations. These are starting
templates, not tuned or GPU-qualified training results. Set the shared native
backbone checkpoint path for the target machine before launch. Keep the
pretrained source and revision matched across methods; the Longshot template
does not start from a separate maths-SFT checkpoint.

The `validation_dataset` is used for SFT loss monitoring. All six templates
evaluate the same **first 768 validation rows** every 500 optimizer steps:
768 batches at microbatch size 1, or 48 batches at size 16. That fixed prefix
contains 122 to 142 examples per task; it is not exactly stratified. The loader
uses `batch_size * eval_batches` rows, not the entire 6,000-row split. Keep that
product fixed when changing microbatch size to preserve validation membership.
The training context cap also applies during evaluation, so loss at 1,536 and
4,096 tokens does not cover identical target tokens. Test rows are separate
and are not configured for checkpoint selection. A model-generation evaluation
suite and harder/family holdouts are outside this initial data integration.

```bash
uv run --extra cpu --extra convert --group dev pytest -q tests/test_reasoning_gym.py tests/test_reasoning_gym_subset.py tests/test_reasoning_gym_audit.py
```

The tests cover hand-computed task semantics, 1,200 seeded reconstructions,
bounded generation, empty outputs, deterministic preparation, global
deduplication, split isolation, admission accounting, refusal to publish partial
datasets, and complete answer-only SFT targets.
The real-tokenizer branch test uses the pinned local tokenizer downloaded above
and skips explicitly if those files are absent; it compares batched and scalar
counting with actual SFT batches for all six shipped examples.

## Reproduce the analysis

```bash
uv run --extra cpu --extra convert python scripts/audit_reasoning_gym.py
uv run --extra cpu --extra convert python scripts/audit_reasoning_gym.py \
  --split validation --out-dir data/analysis/reasoning-gym-validation
```

The command verifies the source checksum, row count, per-task length totals,
and tokenizer identity before writing reports. It checks actual SFT batches
for every available task/cap/truncated-or-complete combination. Use
`--sequence-lens 512 1024 1536 2048 4096` to select audit caps. Output includes
`report.json`, `summary.md`, and exact distribution counts in `quality.json`.
Re-running overwrites those analysis files, never the prepared dataset.
Published {download}`training truncation findings <../../data/analysis/reasoning-gym-truncation/summary.md>`
and {download}`validation findings <../../data/analysis/reasoning-gym-validation/summary.md>`
are retained in Git; the local NumPy length cache is unnecessary to reproduce them.

## Pre-training review and limitations

The September 2026 review retained the frozen training file, fixed subset,
holdouts, and labels. Publication, validation monitoring, audit commands, and
provenance checks were improved without regenerating the data. Exact counts
come from all prepared [training rows](../../data/analysis/reasoning-gym-truncation/quality.json)
and [validation rows](../../data/analysis/reasoning-gym-validation/quality.json),
rather than fresh generator samples.

| Review points | Finding and decision |
|---|---|
| 1, validation size | Increased every template to the same 768-row prefix. This improves coverage but is still a sample and does not establish statistical reliability or answer accuracy. |
| 2, public analysis | Kept the user's public analysis policy, including the math row-length evidence. Generated analysis may be retained as reproducible evidence. |
| 3, audit reproduction | Added the script above; reproduced all published training truncation counts and statistics, within floating-point rounding. |
| 4, production token counts | Added a pinned local Qwen tokenizer test comparing both counting paths against `sft_batch`, including EOS and target masking. |
| 5, subset configurations | Tests now cover all six templates, exposure, cap, dataset paths, and validation row budget. There are three subset templates and three full-data templates. |
| 6, matrix degeneracy | Training answers have one cell in 18.79%, at most four cells in 41.75%, and only zeros in 9.96%; 16.44% of rendered operations leave the state unchanged. Retained these valid solutions and their earlier intermediate states. A stricter size/nonzero admission policy would define a new dataset version. |
| 7, GCD skew | `1` accounts for 49.13% of training and 49.8% of validation answers; `1` or `2` accounts for 74.18% of training. The supervised Euclidean divisions still vary. Retained the distribution; final-answer accuracy must be compared with the constant baseline. |
| 8, dismantling input/output space | Training uses 116,672 of 274,625 possible initial triples (42.48%); all splits occupy 43.21%. A is always zero; B/C are zero in about 75%, Y in 91.79%. Exact-input dedup prevents overlap, not near-neighbour generalisation. This is an interpolation task. |
| 9, grouped dismantling traces | Retained the documented arithmetic compression: mean 111.49 rule applications in 4.41 rendered groups. These are worked solutions, not supervision of every individual transition. Expanding them would materially change length and truncation statistics. |
| 10, cube shortcuts | Ignoring rotations yields the correct queried colour on 21.32% of training and 19.4% of validation. Intermediate face states are supervised; final-colour accuracy alone is a weak metric. |
| 11, clarifications | Retained uniform semantic clarifications. Their redundant tokens are already included in the length audit; making them conditional would change prompt identities and prepared bytes for little expected benefit. |
| 12, publication permissions | Confirmed protected owner-restricted Windows ACLs despite sandbox readability. Restored parent inheritance on the existing full and subset folders and their files. Future staging inherits the destination parent instead of a private temporary-directory ACL. |
| 13, subset audit cap | Added `--sequence-len`, default 1,536. Changing it only changes audit statistics. |
| 14, portable provenance | LF-pinned implementation files and pinned the vendor source manifest plus package initialisers, license, and notice. Original receipts remain historical. |
| 15, throughput | Kept full-data microbatch 1 as a memory-conservative template, pending target-GPU measurement. Increase it when memory permits and adjust `eval_batches` to retain the evaluation budget. Length bucketing was deferred because it changes training order and needs trainer-level qualification. |
| 16, row size | Retained provenance alongside targets. A slim file or persisted index is an optimisation, not a correctness prerequisite; neither has a measured startup bottleneck here. |
| 17, evaluation/format | No Reasoning Gym generation suite was added. Current monitoring measures target-token SFT loss, not final-answer accuracy. Full SFT and LoRA can change output style; Paper7 transfer and extraction success need actual generated-output evaluation. Frozen-backbone cache processors can also change generated behaviour. |
| 18, transition bound | Current generation and rendering both admit exactly 256 transitions; added tests at 255, 256, and 257. No documentation correction to 255 was warranted. |
| 19, memory/diagnostics | Retained full digest strings because this offline build fits the available environment. Failure reports for custom output names are covered by the dataset-directory `*.failed.json` ignore rule. |

These data are suitable for the intended worked-solution SFT experiment, with
the limitations above. They do not constitute a difficult, shortcut-free
reasoning benchmark. A future quality-filtered mixture should use a new output
directory, preparation receipt, subset, and length audit instead of silently
replacing this frozen experiment.
