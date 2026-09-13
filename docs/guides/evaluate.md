# Evaluate a checkpoint

**Status:** Maintained

1. Install evaluation dependencies.

   ```bash
   uv sync --extra cpu --extra eval
   ```

2. Run a bounded suite.

   ```bash
   uv run --extra cpu --extra eval gcw eval all \
     --checkpoint /path/to/checkpoint \
     --limit 1 \
     --out data/eval-runs/manual-smoke.json
   ```

3. Inspect the JSON output and provenance before increasing the limit.

Evaluation constructs independent state. It does not mutate a training checkpoint. Experiment pages define stricter task sets, controls, and evidence gates.

Standard `all`, `math`, `general`, and `tasks` evaluation uses plain prompts by
default. Add `--apply-chat-template` for chat formatting. When chat formatting
is active and `--enable-thinking` is omitted, native checkpoints inherit
`data.sft_enable_thinking` from a recorded training fingerprint or a
published `run_config.json`; checkpoints without that setting and Hugging Face
targets default to false. Explicit `--enable-thinking true|false` always wins.
Chat and GSM8K reproduction always apply chat formatting, so they expose only
`--enable-thinking`. Paper7 exposes both settings and defaults
`--apply-chat-template` to true for legacy compatibility. FineWeb exposes
neither option.

For a cache-processor checkpoint, `--generation-process-every N` overrides the processor invocation cadence during decoding without changing the checkpoint or its prefill cadence. The option is available for `all`, `math`, `general`, `tasks`, `chat`, and the paper7 compatibility suites. It requires an enabled GCW cache processor and does not apply to Hugging Face targets or `fineweb` evaluation.

## Generation stopping

Generation stops at any configured EOS/end-of-turn token and pads finished rows
independently. Native checkpoint EOS/pad overrides source `generation_config.json`,
then `config.json` (including `text_config`), with tokenizer IDs as fallback.
Tokenizer selection is explicit `tokenizer_path`, then checkpoint `tokenizer_source`,
then `source`; these origins must remain accessible. HF targets use source/tokenizer
metadata without checkpoint overrides; native tokenizers retain EOT. Padding defaults to the
first EOS when absent. ID zero is valid; an empty EOS list disables EOS stopping
and requires padding metadata. Likelihood's scalar EOT is unchanged. Results record
resolved IDs; historical evaluations with incorrect stopping are not comparable.
Model-config fallback also retains tokenizer EOS/end-of-turn; an explicit
generation file is authoritative. Tokenizer-only overrides without EOS metadata
recover generation settings from the recorded model source.
Task stop sequences match only generated tokens, including in reference and HF
generation; scoring trims at the earliest EOS or task stop.

## Processor provenance in GSM8K and chat results

GSM8K schema `gcw.gsm8k_reproduction.v4` and chat schema
`gcw.chat_benchmark.v3` distinguish three booleans:

- `processor_requested`: the requested `--processor on/off` policy.
- `processor_present`: an actual GCW cache processor was loaded, before bypass.
- `processor_enabled`: that loaded processor remains active (present and requested).

These fields live in `target`; chat's top-level fields mirror that identity.
GSM8K `config` records the policy as `processor_requested`. The Python input
argument remains `processor_enabled` for compatibility; it expresses a request,
not evidence of a processor. Effective enabled state is not a processor call count.

Plain TF (including full-SFT) and HF targets have present/enabled both false under
either policy; TF retains its loaded model without a bypass wrapper. Use
`--processor off` for full-SFT controls. Composed checkpoints retain their processor
when on and bypass to their exact loaded backbone when off. Processor-only cadence
and diffusion solver overrides require an applicable active processor. Native
loading, generation cadence, and stopping are unchanged. Older schemas retain
their historical meaning and must not be interpreted as effective processor state.

## Answer-only teacher-forced NLL

Use the small native evaluator for token-weighted answer likelihood, not generation
accuracy or benchmark scores. It has no campaign-specific rank, cadence, step,
precision, or solver policy:

```bash
uv run --extra cpu gcw eval answer-nll \
  --checkpoint /path/to/version-2-checkpoint \
  --expected-manifest-sha256 <trusted-manifest-sha256> \
  --data /path/to/answers.jsonl \
  --expected-data-sha256 <trusted-whole-file-sha256> \
  --expected-count 100 \
  --out data/eval-runs/answer-nll.json
```

The input is canonical JSONL: exactly string `prompt` and `answer` fields. Blank
lines do not count as rows but remain part of the whole-file byte hash. Empty
answers retain the native EOT target. Duplicate Unicode-whitespace-normalized
prompts are rejected (case is preserved). Repeat `--disjoint-from JSONL SHA256
COUNT` to reject overlap with additional anchored datasets. This establishes only
exact normalized-prompt disjointness, not paraphrase or pretraining independence.

By default, sequence length, tokenizer, prefix, suffix, chat formatting, and
thinking come from the checkpoint's training fingerprint. Explicit chat or
thinking settings that disagree with the fingerprint are rejected. To specify
a complete alternative data policy, supply both
`--sequence-len N` and `--tokenizer int|cosmo2|hf:SOURCE`, plus any
`--response-prefix` or `--prompt-suffix`. Use `--apply-chat-template` and
`--enable-thinking` to state the matching fingerprint settings explicitly.
Non-answer-only input policies are rejected. Tokenization is exactly
`gcw.data.sft.sft_batch`; the labels are
already causally shifted. The evaluator does **not** shift again. It scores all
non-`-100` targets using the existing bounded native reference loss. Vocabulary
projection covers at most 32 positions at once, including ignored positions;
FP32 per-token losses are then summed over answer targets. Total NLL divides by
total targets, not examples. This avoids allocating full-sequence vocabulary
logits. Each row runs unpadded at batch size one; zero-target rows and nonfinite
scored losses fail rather than being silently dropped. `at_length_cap` indicates that
the input reached the limit, not proof of truncation.

The native inference checkpoint loader handles TF/SFT, ordinary LoRA, processor,
and joint checkpoints without bypassing processors. A complete version-2 training
manifest and every declared payload hash must verify; optimizer/RNG files are
hashed but never deserialized. Model-only directories without that manifest are
not admitted by this stricter protocol. Processor-only payloads additionally need
`--expected-loaded-state-sha256` to bind the exact restored external backbone and
processor state; use `gcw.provenance.tensor_state_sha256(model.state_dict())` from
a trusted reference load. This is inference, not processor-only training resume.
Only explicit native legacy generation/tensor-name migrations are admitted;
arbitrary loaded-config differences and partial backbone/adapter payloads fail.

`--device` defaults to CPU and `--dtype` to native `auto`; FP32, FP16, and BF16
overrides are available. The trained solver configuration is retained unless
`--solver-steps N` explicitly requests the native diffusion evaluation override.
The receipt records actual scheduled field calls per row and in total; zero
means no scheduled fields ran, not necessarily no processor calls. Stochastic
diffusion and random head selection are rejected by this deterministic protocol.
No processor-disabled control is silently substituted. Optional
`--expected-step` and `--expected-tokenizer-sha256` add external identity checks.
Tokenizer identity includes the actual serialized tokenization pipeline and
special-token/template settings (HF tokenizers must have a serializable fast
backend).

One complete JSON receipt contains the loaded state hash, original/effective
configuration, manifest and payload hashes, dataset identities, tokenizer and
package-source hashes, data policy, per-row input/target/mask identities, counts,
and token-weighted totals. Source files, checkpoint files, data, tokenizer, and
loaded model state are rechecked before atomic publication. Existing output is
never overwritten; failures publish no success receipt. SHA256 hashes require
trusted external anchors; they are integrity bindings, not signatures.

Library entry points live in `gcw.evaluation.answer_nll`: `answer_nll`,
`evaluate_answer_nll`, and `paired_answer_nll`. For paired comparisons, first use
`load_answer_nll_rows(path, expected_sha256=trusted_receipt_hash)` on each receipt,
then call `paired_answer_nll(reference, candidate, expected_count=N)` with the
independently known dataset size. Missing, duplicate, reordered, or
identity-mismatched rows fail in linear time, including identical omissions from
both sides.
`delta_nll` is candidate minus reference, weighted by the shared target counts;
negative is better. All comparisons must use the same data and tokenization
policy. Receipts and parsed rows are retained in memory; this is a bounded,
auditable batch-one evaluator, not a high-throughput streaming service.
`scripts/evaluate_answer_nll.py` remains a thin entrypoint accepting the same
arguments as `gcw eval answer-nll`.

## Paper7 protocols

`gcw eval paper7` is the corrected protocol for new evaluations. It applies the
selected chat-template and thinking settings to GSM8K OpenMath, SVAMP,
TheoremQA, and GSM8K Hard. LogiQA and Gaokao MathQA always use plain prompts,
thinking disabled, and benchmark-native option likelihood. One model load is
shared across all six tasks. Its result includes a `paper7_mixed_prompt_v1`
protocol block recording the effective booleans and request type for every
evaluated task.

```bash
uv run --extra gpu --extra eval --extra qwen35 gcw eval paper7 \
  --checkpoint /path/to/checkpoint \
  --batch-size 16 \
  --out results/<run>_paper7.json
```

For corrected Paper7, `--apply-chat-template false` disables chat formatting
for the four generation tasks as well. It has no effect on LogiQA or Gaokao,
which remain plain by definition.

`gcw eval paper7-legacy` exists for direct compatibility with martin-branch
Paper7 evaluation runs. Use it for old-checkpoint replication and for new
checkpoints that must be compared against those legacy result files. It applies
one suite-wide chat-template setting to all six tasks and retains the legacy
lm-eval-compatible top-level result shape.

For a Longshot random-routing ablation, add `--head-selection-method random`.
This runtime-only override applies to both the initial head selection and every
reachable internal reselection. Learned-selector parameters remain loaded from
the checkpoint but are not used for routing.

The legacy compatibility path preserves the old task and scoring protocol while
making its thinking behavior explicit:

- task YAMLs are loaded from `configs/tasks_legacy/paper7`;
- every generative task receives up to 2,048 new tokens;
- chat templates are applied by default when the tokenizer provides one;
- thinking inherits the checkpoint's training setting when present and
  otherwise defaults to false; `--enable-thinking true|false` overrides it;
- legacy Qwen3.5 and Gemma4 checkpoints keep the old generate-time special-token behavior;
- result JSON keeps the old lm-eval-compatible top-level shape.

Run Qwen3.5 legacy paper7 evaluation with the Qwen3.5 runtime extra:

```bash
GCW_EVAL_SKIP_TASKS=theoremqa \
uv run --extra gpu --extra eval --extra qwen35 gcw eval paper7-legacy \
  --checkpoint /path/to/checkpoint \
  --batch-size 16 \
  --out results/<run>_paper7.json
```

On a pre-synced Slurm environment, prefer `uv run --locked --no-sync --extra gpu
--extra eval --extra qwen35 ...` so the job does not rebuild dependencies on the
compute node.

Report legacy paper7 metrics in the martin table order:

| Column | Task | Metric |
|---|---|---|
| GSM8K | `gsm8k_openmath` | `exact_match,flexible` |
| SVAMP | `svamp` | `exact_match,flexible` |
| GSM-Hd | `gsm8k_hard` | `exact_match,flexible` |
| ThmQA | `theoremqa` | `exact_match,none` |
| LogiQA | `logiqa` | `acc_norm,none` |
| Gaokao | `agieval_gaokao_mathqa` | `acc_norm,none` |

If TheoremQA is skipped, leave ThmQA blank and compute `AVG` over the remaining
present task columns. `scripts/extract_paper_table.py` reproduces this extraction:

```bash
uv run --extra eval python scripts/extract_paper_table.py \
  --json results/<run>_paper7.json \
  --out results/<run>_paper7.md \
  --csv results/<run>_paper7.csv
```

## Canonical scores versus numeric thinking comparisons

Keep two protocols separate:

- Multiple-choice scores use full LM Evaluation Harness datasets, no chat template, and benchmark-native option likelihood. They are comparable to the corresponding harness protocol and have no thinking dimension.
- Numeric thinking comparisons use the official tokenizer chat template, deterministic generation, numeric-last-answer scoring, and the same 2,048-token cap in both conditions.

Generated multiple-choice answers are useful only as supplementary instruction-following diagnostics. Do not substitute them for canonical likelihood scores or schedule them across every experiment arm.

```bash
uv run --extra gpu --extra eval gcw eval general \
  --checkpoint /path/to/checkpoint \
  --processor on \
  --out data/eval-runs/standard-general.json

uv run --extra gpu --extra eval gcw eval chat \
  --checkpoint /path/to/checkpoint \
  --tokenizer-path /path/to/qwen-tokenizer \
  --tasks gsm8k_openmath svamp gsm8k_hard \
  --enable-thinking \
  --max-new-tokens 2048 \
  --out data/eval-runs/chat-thinking-on.json
```

The maintained Qwen3 experiment evaluates HellaSwag, PIQA, WinoGrande, ARC, MMLU, LogiQA, and Gaokao MathQA in their full no-chat likelihood protocols. Chat generation is reserved for the full GSM8K OpenMath, SVAMP, and GSM8K Hard datasets. Generated results record parse failures, truncations, response tokens, processor state, selected-row hashes, and per-task accuracy. LAMBADA is excluded because next-word completion has no meaningful thinking-on/chat analogue.

## Qwen3 GSM8K reproduction

Use `gcw eval gsm8k-reproduction` when exact generated rows and protocol identity matter. One invocation evaluates one thinking condition. It supports HF directories and native GCW checkpoints, processor bypass, separate tokenizer ownership, and both maintained scorers.

```bash
uv run gcw eval gsm8k-reproduction \
  --target-kind gcw \
  --model-path /path/to/checkpoint \
  --tokenizer-path /path/to/qwen-tokenizer \
  --processor off \
  --scoring-protocol lm_eval_flexible_exact \
  --output-dir data/eval-runs/gsm8k-cell
```

`scripts/analysis/run_benchmarks.py` remains a thin entrypoint accepting the
same arguments as `gcw eval gsm8k-reproduction`.

`numeric_last_answer` and `lm_eval_flexible_exact` are different scientific protocols. The historical Qwen3 BGT cells used flexible exact match. Results retain scorer, batch size, generation cap, tokenizer hash, checkpoint hash, `apply_chat_template`, `enable_thinking`, and processor state. Pass an explicit `--enable-thinking true|false` for controlled comparison cells and write each condition to a separate output directory.

The maintained fast path uses length-sorted, left-padded Hugging Face generation for HF targets and preallocated static-cache generation for native GCW targets. The default batch size is 64. An authenticated Qwen3-0.6B GSM8K validation completed both 1,319-row conditions in 35 minutes and scored 781 with thinking disabled and 899 with thinking enabled. The earlier exact-length run scored 782 / 888. Batch 128 was slower and drifted further with thinking enabled, so 64 is the calibrated default. Treat these as close downstream-score parity, not token-level parity: batch shape changes floating-point execution and therefore greedy trajectories.

## Deferred extension: high-throughput generation

Small autoregressive models can leave most accelerator memory unused while remaining decode-bound. Memory occupancy is not the bottleneck: every generated token depends on the previous token, and small matrix operations underutilize an H200. The current Qwen3 campaign keeps its validated evaluator unchanged; throughput work belongs in a separately versioned protocol.

Candidate improvements, in implementation order:

1. Separate likelihood and generation batch controls. Keep memory-heavy MMLU likelihood cells conservative while calibrating task-specific numeric-generation batches.
2. Add padded native generation with correct attention and cache masks. Native checkpoints currently group rows by exact prompt-token length, which fragments batches.
3. Add continuous batching and paged KV-cache support, either natively or through a version-pinned vLLM/SGLang backend.
4. Evaluate CUDA graphs, compiled decode steps, and optimized attention kernels after batching semantics are stable.

Do not treat these as transparent operational changes. Batch shape and decoding kernels alter floating-point execution and can change greedy trajectories. A production implementation must:

- assign a new protocol identity;
- freeze requested and effective batch/grouping rules per task;
- rerun the untouched baseline before comparing trained checkpoints;
- verify processor-off parity and processor-on cache semantics;
- record backend and kernel versions, peak memory, generated tokens, throughput, parse rate, and truncation rate;
- restart every affected cell from row zero after an OOM or batch change.
