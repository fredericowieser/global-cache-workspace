# Reasoning Gym supervised traces

The six-task pipeline supplies deterministic worked solutions for supervised
fine-tuning without teacher inference or reinforcement learning. Preparation,
reproduction commands, current dataset statistics, and all six training
templates are documented in the [integration guide](../integrations/reasoning-gym.md).
No training or reasoning-transfer result is established by preparation.

## Task selection and trace construction

The bundled generators come from Reasoning Gym revision
[`49b07130b3fcd12f2d064bba7c43869543a0e7e7`](https://github.com/open-thought/reasoning-gym/tree/49b07130b3fcd12f2d064bba7c43869543a0e7e7).
Two tasks involve rewriting, two involve arithmetic, and the remaining tasks
cover spatial tracking and matrix operations. GCD and the division phase of
base conversion share quotient/remainder trace structure, so these arithmetic
tasks should not be treated as independent forms of reasoning. The mixture is deliberately
selected rather than representative of the entire Reasoning Gym catalog.

| Task | Trace construction and admission |
|---|---|
| `string_manipulation` | Replay the first applicable rule in prompt order, show every new string, and explain termination. Require at least two productive transitions and bound generation at 256. |
| `base_conversion` | Accumulate source digits into a decimal value and show every target-base division and remainder. Bases 2 to 16 and values 256 to 1,048,575 provide a large input space. |
| `ab` | Replay leftmost rewrites with explicit program states. Exclude empty final programs and insufficient depth. |
| `color_cube_rotation` | Replay the visible rotations and show all six face colours after every move. Read the requested face from the final state. |
| `manipulate_matrix` | Replay each operation. Admit only nonconstant final matrices with at least six cells and two rows/columns, and require at least 75% of operations to change the state. |
| `gcd` | Use a controlled mixture of coprime and common-factor pairs; show every Euclidean division and require at least four. The wrapper reuses the upstream prompt and scorer with its own number sampler. |

Solutions are constructed from prompt-visible inputs, not copied from hidden
answer metadata. Every reconstructed endpoint must exactly equal the generator
answer and receive full direct reward. Tests include hand-computed semantics,
seeded reconstructions, quality admission, both tokenizer counting paths,
publication failures, and all full/subset training templates.

## Preparation and evaluation boundaries

The builder admits complete prompt/target sequences up to 4,096 tokens, assigns
disjoint effective seed ranges, and deduplicates prompts and structured task
inputs globally across training, validation, and test. Receipts record pinned
source and implementation hashes, generator settings, quality criteria,
exclusions, token counts, and checksums. The prepared full split and fixed
128,000-row subset are shared across methods.

The SFT answer includes the worked solution and a labelled final answer.
Answer-only loss masks prompt tokens; thinking is disabled consistently with
the pinned tokenizer template. The 1,536-token subset configurations truncate
at load time, with measured rates in the integration guide.

Validation and test contain fresh instances of the same mixture. They do not
measure harder instances, unseen task families, or unseen compositions.
Validation monitoring uses SFT loss; a generation-based Reasoning Gym suite is
not included. Keep KOR-Bench outside mixture tuning and checkpoint selection,
and measure generated-answer performance before claiming reasoning transfer.

## Released teacher traces: alternative assessed

The [SYNTHETIC-2 verified SFT release](https://huggingface.co/datasets/PrimeIntellect/SYNTHETIC-2-SFT-verified)
contains teacher-generated responses from DeepSeek-R1-0528. The public dataset
statistics inspected on 2026-09-22 reported **4,055 rows** with
`task_type=reasoning_gym` within 104,913 total rows. This is a subset of a mixed
dataset, not 104,913 Reasoning Gym examples.

A sample of 300 Reasoning Gym rows, drawn from offsets 59000–59099,
60000–60099, and 61000–61099 of the default training split, had:

- Reward 1 and a closing `</think>` delimiter in all 300 responses.
- Response lengths of 5,269–60,104 characters, with median 26,962 characters.
  These are character counts, not token counts; model-specific tokenisation
  remains necessary.
- Null `problem_id` fields in all 300 rows. The SFT schema contains messages,
  task type, reward, and problem ID, but lacks per-task generator metadata.
- Prompts covering maze navigation, button/state puzzles, bitwise arithmetic,
  truthfulness constraints, grid-rule induction, and rotten-oranges propagation.

This makes a small distillation experiment possible without generating teacher
responses. It is less attractive for the proposed controlled mixture because
the pool is limited, responses are long, and generator-level provenance is not
available in the SFT rows. Recovering provenance or independently re-verifying
answers would require additional work. The dataset card also allows partial
reward for some tasks; do not assume the release name guarantees strict
correctness. Even full final-answer reward does not validate every reasoning
step. The inspected release and sample were not pinned to an immutable dataset
revision, so these observations must be rechecked and pinned before reuse.
