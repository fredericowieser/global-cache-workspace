# KOR-Bench

**Status:** Experimental — core zero-shot integration implemented; production-model
accuracy and runtime have not been measured with this integration.

GCW evaluates the five KOR-Bench categories through its native LM Evaluation
Harness adapter: operation, logic, cipher, puzzle, and counterfactual. The
`korbench` group covers 1,250 questions (250 per category, ten per rule).
Multi-question, multi-rule, few-shot, and self-correction protocols are not included.

## Prepare data

Install the evaluation extra alongside the appropriate Torch runtime. Symbolic
scoring requires `sympy` and `antlr4-python3-runtime==4.11.1`, both declared in
`eval`.

```bash
uv sync --extra cpu --extra eval
uv run --no-sync python scripts/prepare_korbench.py
```

Preparation downloads only the original rules, questions, and zero-shot prompt
file from upstream revision `bb8194dff8e51e2e2247cf62aa9932c3d57d53d7`. It checks
pinned SHA256 hashes, joins rules by ID, validates the population, and writes
`data/eval-data/korbench/` with a manifest. It refuses to overwrite an existing
destination. Raw/prepared records are excluded from Git.

For an offline upstream snapshot or alternate destination:

```bash
uv run --no-sync python scripts/prepare_korbench.py \
  --source-dir /path/to/KOR-Bench --out /path/to/prepared-korbench
export GCW_KORBENCH_DATA=/path/to/prepared-korbench
```

The loader verifies prepared files against code-pinned hashes. Prepared prompts
match upstream formatting exactly and contain only the rule and question, without
solution hints, `needle`, `trick`, or gold answers. Question and rule IDs remain
strings because the upstream scorer uses them for special cases.

## Evaluate

Run from the repository root so `configs/tasks/` is discoverable:

```bash
uv run --no-sync gcw eval korbench \
  --checkpoint /path/to/checkpoint \
  --apply-chat-template true --enable-thinking false \
  --batch-size auto --log-samples \
  --out results/model_korbench.json
```

Use `--hf-model` instead of `--checkpoint` for a Hugging Face baseline. Add
`--processor off` to bypass a native checkpoint's cache processor. Select an
individual category with, for example, `gcw eval tasks --tasks korbench_logic`. `--limit 2`
evaluates two questions **per category**, not two for the whole group.

Tasks inherit GCW's generation limits: 2,048 new tokens with thinking disabled
and 4,096 with thinking enabled, including both thinking and the final answer.
EOS still stops generation early. These differ from the upstream 2,000-token
default. The task-group runtime metadata records the effective cap and
`gcw.korbench.zero_shot.v1`; record thinking-enabled conditions separately.
Overlong prompts fail instead of silently removing rule text. The loaded model's
context limit must accommodate the complete formatted prompt; no checkpoint
configuration is changed by this integration.

Generation progress prints completed requests out of 1,250, generated tokens,
and elapsed time at batch boundaries, at most once per minute plus the start and
finish. It does not write progress files; see
[live generation progress](../guides/evaluate.md#live-generation-progress).

## Results and scoring

The output includes category and size-weighted overall `accuracy,none` and
`parse_failure,none`. Counterfactual additionally reports
`real_life_answer_rate,none`: how often the familiar real-world answer is returned
instead of the supplied rule's answer. Scores are fractions, not percentages.
Parse failure measures missing/empty answer delimiters, not every possible
symbolic parsing failure.

`--log-samples` is also available for the standard `all`, `math`, `general`, and
`tasks` suites. It retains harness `samples` and `configs` in the result JSON and
existing authenticated partial-result files. Samples contain source IDs, prompts,
raw/filtered responses, gold targets, and per-example metrics; KOR-Bench also adds
`extracted_answer`. Existing runtime summaries retain response lengths, cap hits,
processor calls, memory, and elapsed time. Generation tokens are not logged
individually. Sample logging is opt-in and can substantially enlarge results.

With `--log-samples --out results/run.json`, each successful batch also saves
request outputs under `results/run.partial-samples/`, with a separate journal per
process/attempt. Records include source documents, responses, generation lengths,
and collected diffusion metrics. Writes are flushed and synced once per batch;
completed responses survive interruption without waiting for the whole group.
Put `--out` on shared storage to preserve them across node shutdown. These are
raw outputs for offline scoring, not automatic generation-resume checkpoints.
See [prediction logging](../guides/evaluate.md#kor-bench-and-prediction-logging)
for the format, interruption handling, and a KOR-Bench scoring example.

For native Longshot diffusion checkpoints, the CLI currently enables cache update
metrics automatically when `--out` is supplied, including for KOR-Bench. Explicit
`--cache-update-metrics` selects the same default sidecar path. Metrics require
an enabled diffusion processor with eager recurrence. Join each sidecar row's
`(task, example_id)` to `(task, str(doc_id))` in the same run's logged samples.

Changing sample logging or its recorded evaluation settings requires a fresh
partial-result destination. KOR-Bench also checks its scorer, task, source, and
scoring-dependency identities before reusing completed partial results. These
checks extend the existing evaluator; checkpoint paths alone are not immutable
weight identities. Use immutable checkpoints for comparisons.

The scorer follows the pinned [upstream implementation](https://github.com/KOR-Bench/KOR-Bench/blob/bb8194dff8e51e2e2247cf62aa9932c3d57d53d7/eval/eval_utils.py),
including category/rule/question exceptions and first-match answer extraction.
Arithmetic puzzles use restricted arithmetic parsing rather than Python `eval`.
Missing/empty answers fail explicitly; these are documented GCW deviations for
malformed responses. Other upstream comparison rules remain unchanged, including
its permissive string normalisations and symbolic fallback. Attribution is in
`licenses/KOR-Bench-NOTICE.md`.

## Validation and limits

- All five tasks load with 250 questions each. A deterministic gold-response
  harness smoke checks generation requests, scoring, aggregation, and samples.
- CPU tests exercise a small native Longshot with its processor enabled/bypassed,
  plus data integrity, prompt preservation, scorer cases, and sample resume.
- Replaying 1,250 released Qwen2.5-7B-Instruct responses agrees with the pinned
  upstream scorer on every item. Regression fixtures retain one response per rule.
  Recomputed scores differ from **27 archived labels** (24 operation, three
  counterfactual); those archived scores are not a reliable parity target for
  this pinned code/data combination. The cause has not been isolated.

No trained-checkpoint evaluation or GPU timing pilot has been run. See the
[reasoning-transfer proposal](../experiments/reasoning-transfer.md) for the
scientific motivation and provisional runtime estimates.
