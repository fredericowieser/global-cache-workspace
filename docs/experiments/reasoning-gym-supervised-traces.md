# Reasoning Gym supervised traces: implementation review

**Status:** Proposal, with source inspection and local feasibility checks,
2026-09-22. No GCW integration or training run has been completed. This review
supports the priorities in [reasoning transfer](reasoning-transfer.md).

## Assessment and scope

A selected six-task mixture with deterministic worked solutions is feasible
within an estimated **4–6 hours of implementation and validation**. Reasoning Gym
has no uniform worked-solution field, but some generators retain intermediate
states or expose operations that can be replayed. This supports supervised
training without teacher inference or an RL loop.

The proposed targets are algorithmic worked solutions: identify an operation,
show the resulting state, and explain termination or read out the result. They
are different from exploratory teacher reasoning and from formal deduction
proofs. Three initial tasks involve rewriting; the mixture adds spatial state
tracking, matrix transformations, and arithmetic, but does not represent the
full breadth of Reasoning Gym. No transfer advantage has been measured for it.

## Inspected source and candidate tasks

The audit used upstream revision
[`49b07130b3fcd12f2d064bba7c43869543a0e7e7`](https://github.com/open-thought/reasoning-gym/tree/49b07130b3fcd12f2d064bba7c43869543a0e7e7),
downloaded and installed separately under `/tmp`. The project environment and
core code were not modified. Sources below refer to that revision.

| Task | Available information | Required trace construction |
|---|---|---|
| [`string_manipulation`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/string_manipulation.py) | Intermediate strings and selected rules in metadata | Describe each transition using the applicable rule and its priority in the prompt; explain termination |
| [`string_splitting`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/string_splitting.py) | Machine/part counts after successive rule applications | Label counts and applied rules; explain why the procedure stops |
| [`ab`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/ab.py) | Existing `compute_steps()` returns program states | Recover the question's input program, replay the solver, and describe rewrites |
| [`color_cube_rotation`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/cognition/color_cube_rotation.py) | Initial face colours, rotation sequence, and rotation functions | Replay rotations, render updated face colours, and read the requested face |
| [`manipulate_matrix`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/algorithmic/manipulate_matrix.py) | Initial matrix, operation parameters, and transformation functions | Replay operations and render each intermediate matrix |
| [`gcd`](https://github.com/open-thought/reasoning-gym/blob/49b07130b3fcd12f2d064bba7c43869543a0e7e7/reasoning_gym/arithmetic/gcd.py) | Input numbers and final result | Generate Euclidean-algorithm division steps, then check the result against the generator |

For the first five tasks, reuse upstream state and transformation logic where
possible. GCD needs a small trace-producing algorithm. Do not present final
solutions stored in metadata as if they were intermediate reasoning.

## Local feasibility observations

Checks used `seed=123456`, indices 0–199, and default task configurations unless
specified below. They generated 200 examples per task, inspected metadata and
prompt uniqueness, and checked gold answers with upstream scorers. A separate
reconstruction check exercised the six proposed trace paths.

| Task | Distinct prompts among 200 | Reconstruction endpoints matching gold |
|---|---:|---:|
| `string_manipulation` | 200 | 200/200 |
| `string_splitting` | 124 | 200/200 |
| `ab` | 200 | 200/200 |
| `color_cube_rotation` | 200 | 200/200 |
| `manipulate_matrix` | 200 | 200/200 |
| `gcd` | 200 | 200/200 |

All 1,200 reconstructed endpoints matched the generated answers. All gold
answers in those samples received full upstream reward. These checks establish
that trace reconstruction is practical; most reuse upstream operations, so they
do not independently prove agreement between the generator and its natural
language specification. They also do not qualify larger difficulty settings,
full textual explanations, tokenisation, training, or transfer.

The checks exposed limitations that preparation must address:

- **Trivial instances:** 59/200 default `string_manipulation` examples needed no
  transformation. Its median number of transformations was one. Filter or
  explicitly limit trivial instances and record the resulting distribution.
- **Small input spaces:** default `string_splitting` samples only three initial
  counts, each from zero to five, giving 216 possible inputs. Increasing the
  requested row count does not create new diversity. Adjust difficulty within
  length limits, deduplicate, and measure retained unique examples.
- **Other candidates are not automatically better:** `string_synthesis` also
  produced 124 distinct prompts among 200. Tower of Hanoi, checked with
  `visualize=True`, exposes intermediate states but produced only 72 distinct
  prompts among 200; some examples had 127 moves. Neither is needed for the
  initial six-task scope.

These were temporary exploratory checks rather than a committed validation
suite. The integration must preserve reproducible tests and preparation reports.

## Preparation and correctness requirements

1. **Trace semantics:** use only prompt-visible inputs and rules to construct
   explanations. Check rule priority, transition validity, repeated-state and
   iteration-limit behaviour, and termination. In string manipulation, internal
   rule identifiers differ from the sampled ordering shown in the prompt.
   Include independent, hand-checkable cases; matching an upstream endpoint
   alone is insufficient.
2. **Data admission:** enforce useful trace depth, nonempty targets, generation
   bounds, and prompt uniqueness. Report retained counts by task and difficulty
   rather than assuming the requested volume is available. Handle empty-state
   outputs explicitly if admitting tasks that can produce them.
3. **Splits and holdouts:** allocate nonoverlapping effective generator seed
   ranges and deduplicate across splits. Several generators use `seed + index`.
   Fresh instances, greater difficulty, and held-out operation compositions are
   separate conditions. Treat related rewriting tasks as a shared family when
   making family-level claims.
4. **Supervision format:** consistently request a worked solution followed by
   an extractable final answer. Reconcile upstream final-answer-only instructions
   with this format. Choose ordinary solution text or thinking delimiters
   consistently with the model's chat template; toggling thinking mode does not
   create a reasoning trace.
5. **Length and provenance:** tokenise complete prompt/solution pairs before
   admission. Reject overlong examples rather than truncating solutions. Save
   generator revision, task configuration, sample identity, split, formatting
   policy, length statistics, exclusions, and dataset checksums.
6. **Matched comparisons:** prepare one frozen dataset for full fine-tuning,
   LoRA, and Longshot. Preserve the common starting backbone and comparable
   tuning budgets. Keep KOR-Bench outside data-mixture and checkpoint selection.

GCW already accepts JSONL `prompt`/`answer` or `question`/`answer` records through
`src/gcw/data/sft.py` and `src/gcw/data/training.py`. The answer field can contain
the complete worked solution; `sft_loss_scope: answer_only` masks the prompt.
The existing OpenR1 preparation script provides patterns for complete-trace
admission, token lengths, and manifests, but its source-specific verification
requirements do not directly apply to Reasoning Gym. Reuse established
infrastructure without coupling new task logic to OpenR1 or the core trainer.

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

## Implementation budget and decision

| Work | Planning estimate |
|---|---:|
| Six trace renderers, consistent prompts, and answer checks | 1–2 hours |
| Difficulty filters, deduplication, splits, length checks, and manifest | 1–2 hours |
| Correctness tests, three training configs, and documentation | 1–2 hours |

Budget **4–6 hours overall** for a usable preparation implementation and CPU
validation, allowing integration overhead. This excludes training, GPU
qualification, teacher inference, exhaustive holdout design, and a comprehensive
new evaluation suite. It is an estimate, not a completed integration or a
guarantee that the resulting mixture is sufficient for the intended claim.

Recommend the six-task mixture before FineWeb for the immediate reasoning-transfer
question, and provisionally before PrOntoQA-OOD for greater procedural variety.
The preference over PrOntoQA is narrower than for an unrestricted Reasoning Gym
mixture. Before committing a large campaign, require enough unique, nontrivial,
complete examples per task and evidence of learning on a small validation set.
If meeting those conditions requires substantial new solver work, switch to
PrOntoQA-OOD for generated proof supervision and controlled generalisation.
The comparison and proposed experiment order are in
[reasoning transfer](reasoning-transfer.md#prontoqa-as-an-alternative).
