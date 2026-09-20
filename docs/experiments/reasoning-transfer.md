# Reasoning transfer and novel-rule evaluation

**Status:** Proposal — living experiment note, 2026-09-20. [KOR-Bench integration](../integrations/korbench.md) is implemented and smoke-tested; no trained-checkpoint runs have been completed for this proposal.

## Motivation

The paper proposes a processor that integrates available information into reusable
reasoning. Training on OpenMathInstruct and evaluating on maths measures domain
adaptation, but does not isolate general reasoning from familiar problem patterns.
SFT and LoRA can learn transferable procedures; freezing the backbone does not
prevent the processor from learning domain-specific patterns or exposing answers
already represented in the backbone.

The hypothesis is **whether Longshot learns improvements that transfer across
domains and unfamiliar rule systems more effectively than comparable adaptation
methods**. Baselines need not remain unchanged for this hypothesis to receive support.

## Complementary experiments

1. **Change the adaptation domain.** Compare maths-only, general-text-only, and a
   fixed maths/general mixture, then evaluate on maths and non-mathematical
   reasoning. Start from the same pretrained backbone; compare full fine-tuning,
   parameter-matched LoRA, and frozen-backbone Longshot. Raw-text full-weight
   training is continued pretraining, rather than SFT.
2. **Test unfamiliar rules supplied in context.** Use fictional facts, newly
   defined operators, or explicit rule systems. Test fresh instances, held-out
   compositions, longer deductions, and pairs where changing a relevant premise
   changes the answer. These probe the use of available information more directly.

[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is a candidate
corpus, but broad web provenance does not establish absence of maths or benchmark
solutions. Audit exclusions and contamination before calling it out-of-domain.
A mixture still contains maths supervision, so its gains do not isolate general-text
transfer. Weak transfer from prose could also reflect an insufficient learning
signal for multistep computation, rather than an architectural limitation.

## Candidate resources

KOR-Bench is now implemented; the other effort estimates reflect code inspection
and upstream documentation, not completed integrations. “Low” means mostly data conversion, task configuration, and label
extraction; “moderate” includes custom scoring or split construction.

| Resource | What it tests | Training/evaluation availability | Ease of evaluation/implementation in GCW |
|---|---|---|---|
| **[KOR-Bench](https://kor-bench.github.io/)** | Supplied unfamiliar rules across operations, logic, ciphers, puzzles, and counterfactuals; closest match to the broad claim. | Primarily evaluation: 125 rules, 1,250 questions; released [data and evaluation code](https://github.com/KOR-Bench/KOR-Bench). | **Implemented.** Five tasks reuse native GCW generation, pinned prompts/data, and an adapted upstream scorer. Preparation and evaluation commands are in the [integration guide](../integrations/korbench.md). |
| **[PrOntoQA / PrOntoQA-OOD](https://github.com/asaparov/prontoqa)** | Deduction over fictional or counterfactual worlds, with controllable depth and premise ordering. | Released examples and a JSON generator with reasoning chains; suitable for generating training and evaluation data. | **Low for answer accuracy; moderate for controlled holdouts.** Convert records, add prompts and label extraction. Proof checking and strict composition holdouts require additional work. Best first controlled study. |
| **[ProofWriter](https://aclanthology.org/2021.findings-acl.317/)** | Conclusions and proofs derived from supplied natural-language facts and rules. | Training/evaluation data with proof annotations; supports studies of generalisation to greater proof depth. | **Low–moderate for answer accuracy; higher for proofs.** Flatten theory/question records, preserve open-/closed-world semantics and depth metadata, and score labels. Proof correctness needs a separate verifier. Group questions by theory when constructing splits. |
| **[Reasoning Gym](https://github.com/open-thought/reasoning-gym)** | Procedural tasks across logic, algorithms, graphs, arithmetic, and other domains; not all involve unfamiliar rules. | Scalable generators, adjustable difficulty, and programmatic verifiers. Generated question–answer pairs can train with next-token loss; RL is not required. | **Low–moderate for selected tasks; moderate–high for broad coverage.** Freeze generated rows and metadata, add task definitions, and wrap upstream verifiers where exact match is insufficient. Task selection, dependency pinning, and family-level holdouts add work. |

## Estimated evaluation time

**Planning estimates, not benchmark measurements:** per checkpoint, one decoding
condition, one response per question, after integration and data preparation.
Assume one H200-class GPU, a 2B–27B model that fits with its processor, native GCW
automatic batching, and **50–150 aggregate output tokens/second**. Hardware is a
planning assumption, not verified from the timing files below. Exclude queueing,
downloads, training, and implementation time; allow roughly 10–20 minutes for
loading and scoring. Actual response length matters more than the token cap.

| Resource | Evaluation workload | Assumed mean generated tokens per question | Approximate wall time |
|---|---|---|---|
| KOR-Bench | Full 1,250-question core benchmark; excludes extra self-correction/mixed-task experiments | 512–1,024 | **1.5–8 hours** |
| PrOntoQA / PrOntoQA-OOD | Proposed 2,000 generated questions total across selected depths/conditions | 64–256 | **0.5–3.5 hours** |
| ProofWriter | Proposed 2,000-question stratified subset of a specified test split; not the full release | 64–256 | **0.5–3.5 hours** |
| Reasoning Gym | Proposed 10-task selection × 200 examples; not the entire library | 256–1,024 | **1–12 hours** |

The rate assumption is informed by existing Paper7 `gcw_runtime` records:
[2B diffusion](../../results/qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24_final_paper7_update_metrics_thinking-off.json)
produced about 142 output tokens/second on GSM8K and 119 on TheoremQA;
[27B Longshot](../../results/qwen35_27b_instruct_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-off.json)
produced about 111 on GSM8K and 44 on SVAMP. These are task totals divided by
elapsed task time, including overhead, not isolated decode speeds or matched
hardware comparisons. The estimates are rounded extrapolations, not upper bounds.

Use `time ≈ question_count × mean_response_tokens / aggregate_tokens_per_second`
plus loading/scoring. Long prompts, fragmented batches, and processor settings
can lower throughput. If KOR-Bench responses average 2,048 tokens, budget roughly
**5–15 hours** instead. Label-only deduction scoring can be much faster than the
short-explanation assumptions above; full proofs or extended thinking can be slower.

Before budgeting a campaign, time a representative 50–100-question pilot per
benchmark and model/processor setting, covering long prompts and hard cases.
Scale by actual example count and response length. Multiply for checkpoints,
seeds, thinking modes, and processor ablations; independent GPUs can reduce
calendar time but not total GPU-hours. No pilot has been launched.

## Reuse in this codebase

Use `gcw eval tasks`: `src/gcw/evaluation/core.py` discovers YAMLs under
`configs/tasks/` and runs them through `GCWEvalLM` in
`src/gcw/evaluation/lm_eval_adapter.py`. Native checkpoint loading, processor bypass,
generation, and likelihood already exist; no new model backend is expected.
The specialised chat evaluator has a fixed task list and numeric scoring paths,
so the general task route is the simpler extension point.

Add benchmark-specific preparation, prompts, and scorers. Training examples can
use existing `prompt`/`answer` records, with rule IDs, proofs, generator settings,
and provenance retained separately. The shared evaluator now supports opt-in `--log-samples` for per-example results
and task configurations, including KOR-Bench answer extraction.
See the [evaluation guide](../guides/evaluate.md) and
[harness integration](../integrations/lm-evaluation-harness.md).

## Controls and interpretation

- **Separate novelty levels:** fresh instances, unseen compositions, and unseen
  rule/task families. New seeds or names alone do not establish structural
  generalisation. Public benchmarks may occur in backbone pretraining; supplement
  them with freshly generated, audited holdouts.
- **Match comparisons:** data exposure, prompts, decoding, generation caps, and
  tuning budgets. Record supervised tokens, response lengths, truncation, and
  compute. Use multiple training seeds where feasible; example-level uncertainty
  does not measure seed variability.
- **Attribute gains:** include the untouched backbone and processor-off control.
  Ablate cross-head communication and vary refinement steps, recording additional
  computation. Transfer gains do not prove that memorisation is absent.
- **Fix scoring and supervision:** validate answer extraction and label semantics,
  report parse failures, and use gold proof depth for difficulty analyses.
  Choose answer-only versus proof supervision explicitly and consistently across
  methods; longer prompts alone need not require deeper reasoning.

## Suggested sequence and open decisions

1. Evaluate existing checkpoints on **KOR-Bench**, keeping it outside training and
   checkpoint selection. This tests transfer from current maths adaptation.
2. Run a controlled **PrOntoQA** study: train on short deductions, test fresh
   instances, longer chains, held-out compositions, and premise-change pairs.
   ProofWriter offers a complementary deduction benchmark.
3. Expand to a selected **Reasoning Gym** training mixture with held-out task
   families, plus the FineWeb/maths comparison. Retain maths evaluation to measure
   transfer back to the original application.

Still to choose: backbone/processor variant, token/compute budget, training targets,
precise splits, primary metric, seed count, and integration priority. Update this
note with pinned dataset versions and results as these are settled.
