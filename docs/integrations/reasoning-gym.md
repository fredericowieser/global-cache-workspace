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
  --workers 4 --replace
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
adaptations and removal of unused APIs. Vendored generator, transition, and direct scorer
logic is preserved; the GCW GCD wrapper supplies the number-sampling policy
described below. The upstream Apache-2.0 license and attribution notice ship
with the subset. The GCW wrapper bounds string rewriting at 256 transitions.
The source manifest is itself hash-pinned, including hashes of package wiring,
license, and notice files. Implementation and vendor files use LF line endings
for portable hashes. Receipts record the implementation that generated the
current files.

## Mixture and labels

| Task | Generator settings | Worked solution |
|---|---|---|
| `string_manipulation` | String length 8 to 28; 6 to 12 sampled rules | First applicable rule, its position in the displayed list, and each new string; explicit termination |
| `base_conversion` | Different source/target bases from 2 to 16; values 256 to 1,048,575 | Accumulate the source digits into a decimal value; show every quotient/remainder division in the target base, then reverse the digits |
| `ab` | 18 input tokens | Leftmost applicable pair, replacement, and resulting program |
| `color_cube_rotation` | 2 to 8 rotations | All six face colours after each rotation, followed by the requested face |
| `manipulate_matrix` | 3 to 6 initial rows and columns; 3 to 8 operations; nonconstant final matrix with at least 6 cells and 2 rows/columns; at least 75% productive operations | Each operation and its resulting matrix |
| `gcd` | Two integers from 10 to 100,000; 25% coprime candidates, otherwise a uniform common factor from 2 to 100; at least 4 divisions | Every Euclidean quotient/remainder division and the final GCD |

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

Admission requires at least two productive transitions; GCD requires at least
four Euclidean divisions. Matrix operations that leave the matrix unchanged do not count towards
this minimum, although their intermediate state remains in the trace. Empty
final answers, including completely cancelled AB programs, are explicitly
excluded. Repeated string states are discarded according to the prompt;
bounded string generation failures are counted as exclusions.

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

Preparation refuses an existing destination unless `--replace` is supplied.
Files are staged and the completed directory is published only after every
quota is met. Replacement accepts only a recognised prepared dataset directory
containing its data files and manifest, and removes the preceding files after
successful publication. Use the same paths and flags on every machine.
Failures leave a sibling
`<output-name>.failed.json` diagnostic, rather than an apparently complete
undersized dataset. Generated data and temporary staging directories are ignored
by Git; the registry contains the small preparation receipt.
Staging uses an ordinary sibling directory, so published directories inherit
destination-parent permissions, subject to the operating system's ACL and umask.
On a shared cluster, prepare beneath the intended shared dataset parent.
Cleanup retries transient filesystem failures. If the superseded backup or
staging directory remains locked after publication, the command exits with
`PublishedDatasetCleanupError`, reporting the valid published dataset and the
exact directory to remove after closing file handles. It does not write a
build-failure diagnostic or silently leave a successful command with an
undisclosed directory. If a build fails before publication, a staging cleanup
failure preserves the original build error and reports the cleanup error as
its cause.

Validation and test measure **fresh instances from the same task distribution**.
They are not harder-instance, composition, or family holdouts. Two tasks share
the rewriting family and two involve arithmetic; those two families each
contribute one third of the balanced mixture. Spatial tracking and matrix
transformations each contribute one sixth. KOR-Bench is not used for generation,
filtering, mixture tuning, or checkpoint selection.
No transfer advantage or learning result is established by data preparation.

## Training configurations and checks

### Fixed 128,000-example setting

Create the subset once from the prepared full training split:

```bash
uv run --extra cpu python scripts/subset_reasoning_gym.py --replace
```

This writes `data/train-data/reasoning-gym-128k/train.jsonl` and
`manifest.json`; `--replace` permits regeneration at that path. It verifies the
full
training file against its manifest, samples uniformly without replacement
within each task using seed 24, and shuffles the selected rows once. Task quotas
are proportional to the source counts, rounded by largest remainder with seeded
tie-breaking. Cube rotation and matrix manipulation each receive 21,334 rows;
each other task receives 21,333. No length or difficulty filtering is applied.

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
`longshot_128k.yaml`. These use the same full validation/test splits and
starting backbone as the full-data setting, with a 1,536-token training cap,
microbatch size 16, global batch of 64 rows, and 2,000 optimizer steps for one
pass. Output directories are separate from the full-data runs.
The full-data configurations use 4,096 tokens.

Microbatch size is per rank. Keep `global_batch_rows` at least
`batch_size * world_size` and divisible by that product. Adjust `max_steps`
to `128000 / global_batch_rows` when changing the global batch for a one-pass
run, choosing a divisor of 128,000. For example, global batch 128 uses 1,000
steps. Actual GPU memory fit and throughput still require qualification.

Stored solutions remain complete. The loader truncates during training:
1,634 rows (1.27656%) exceed 1,536 tokens, comprising
876 string-manipulation and 758 AB examples. Among truncated rows, original
length averages 1,801.44 tokens and removed length averages
265.44 tokens; both have population standard deviation 254.63.
Seven rows lose only EOS. No row becomes prompt-only, and
0.88106% of target tokens are removed. Counts use the authenticated admission
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
contains 119 to 141 examples per task; it is not exactly stratified. The loader
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
uv run --extra cpu --extra convert python scripts/verify_reasoning_gym.py
uv run --extra cpu --extra convert python scripts/audit_reasoning_gym.py
uv run --extra cpu --extra convert python scripts/audit_reasoning_gym.py \
  --split validation --out-dir data/analysis/reasoning-gym-validation
```

`verify_reasoning_gym.py` checks all split checksums, global deduplication,
labels, seed isolation, and token counts, then regenerates 100 examples per
task/split through the actual tokenizer and SFT loader. Only after these checks
pass does it write the preparation receipt and six example rows.

The truncation command verifies the source checksum, row count, per-task
length totals,
and tokenizer identity before writing reports. It checks actual SFT batches
for every available task/cap/truncated-or-complete combination. Use
`--sequence-lens 512 1024 1536 2048 4096` to select audit caps. Output includes
`report.json`, `summary.md`, and exact distribution counts in `quality.json`.
Re-running overwrites those analysis files, never the prepared dataset.
Published {download}`training truncation findings <../../data/analysis/reasoning-gym-truncation/summary.md>`
and {download}`validation findings <../../data/analysis/reasoning-gym-validation/summary.md>`
are retained in Git; the local NumPy length cache is unnecessary to reproduce them.

## Data-quality policy and interpretation

Matrix admission rejects constant or undersized answers and traces in which
more than one quarter of operations leave the state unchanged. These conditions
are checked in addition to trace depth. The renderer records final rows,
columns, and distinct-cell count in `difficulty`; admission consumes these
structural facts. This filtering also changes the operation mixture, especially
by excluding crop-heavy sequences. The quality report records exact admitted
per-transform operation counts and row frequencies, with an unfiltered sample
of the first 4,000 candidates (or all candidates if fewer) from each split
for comparison. Rotation angles share the `rotate` category; the initial
identity instruction is excluded. In the training audit, crops account for
10.45% of operations in the candidate sample and 1.33% of admitted operations.

GCD generation samples a factor: 1 with probability 0.25, otherwise uniform
from 2 to 100. For the actual factor `g`, each cofactor is drawn from
`ceil(10 / g)` through `floor(100000 / g)`, rejecting pairs whose GCD is not 1,
before multiplying both by `g`. Both branches therefore generate inputs in
the same numeric interval, rather than giving coprime inputs a smaller range.
The prompt and scorer are reused from the pinned GCD generator. Trace-depth admission and
global deduplication can slightly alter the final coprime proportion; the exact
label histogram is published in the analysis report. The prepared training
split contains 26.08% coprime GCD pairs, with all 100 final GCD values represented.
The coprime rate is 30.04% among 1,032 examples whose larger input has four
digits and 26.04% among 115,635 examples with at least five digits. Only five
examples have smaller maxima, so that bin is too small for a useful estimate.
Matching input ranges removes the near-deterministic size shortcut; it does
not imply exact statistical independence after admission.
All admitted matrix answers satisfy the size, diversity, and productivity
criteria. Base conversion has 115,445 distinct final answers in 116,672 training
rows; different inputs may legitimately have the same final representation.

Base conversion spans 15 source bases and 14 different target bases over
1,048,320 possible values, giving 220,147,200 input combinations. Its worked
solutions contain each digit accumulation and each division, without grouped
transition shortcuts. All labels are reconstructed from the visible prompt.
Base conversion and GCD share quotient/remainder arithmetic in their traces;
the two tasks provide related arithmetic supervision, not independent reasoning
families.

The manifest records these generator settings and admission criteria. The
[training distribution counts](../../data/analysis/reasoning-gym-truncation/quality.json)
and [validation distribution counts](../../data/analysis/reasoning-gym-validation/quality.json)
report final-answer diversity, GCD labels and coprime rates by input magnitude,
matrix admission checks and operation frequencies, and the cube
initial-colour baseline. Cube rotations can legitimately return the queried
face to its initial colour; intermediate face states remain supervised.

These are procedural worked-solution SFT data. Validation loss is measured on
supervised target tokens, not final-answer accuracy. The six tasks do not
establish broad reasoning transfer, harder-instance generalisation, or family
holdout performance. Full SFT, LoRA, and cache processing can all change output
behaviour; evaluate actual generated answers when measuring transfer.

The full-data microbatch size of 1 is a memory-conservative template. Increase
it when the target GPU permits, preserving global batch/exposure and the
validation row budget. Length bucketing and compact training files are not
implemented; rows retain provenance alongside targets.

## 128k experiments initialized from maths checkpoints

`configs/experiments/reasoning_gym_128k_maths_init/` contains six Qwen3.5-4B
runs initialized from the step-2000 OpenMathInstruct-2 checkpoints selected for
the paper in `TODO.md`: Fred's SFT and Martin's `pct015` LoRA, BT, BGT,
Longshot, and Diffusion-Longshot. Paths point to the local copies under
`/shared/gwt/checkpoints/`.

These configs use `init_checkpoint` with `backbone_checkpoint: null` and
`resume: null`. This loads all backbone and adapter/processor weights, while
starting a new optimizer, schedule, data cursor, and 2,000-step training run.
They retain the original Reasoning Gym settings and probed batch sizes:
SFT/Longshot/Diffusion-Longshot 16, LoRA/BT/BGT 32. Outputs and W&B run names
use a separate `reasoning_gym_128k_maths_init` namespace. The obsolete
`legacy_bottleneck_generate` alias is cleared while retaining its resolved
`processor_generation_mode`, matching the checkpoint loader's configuration.

The campaign submits six one-GPU training workers with KOR-Bench evaluation,
then six one-GPU Paper7 evaluation workers. Both use thinking off; KOR-Bench
logs samples and automatically records diffusion cache-update metrics where
applicable. Paper7 excludes TheoremQA. Only the GPU count is requested;
CPU, memory, partition, and time come from Slurm defaults.
