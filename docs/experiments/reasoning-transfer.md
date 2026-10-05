# Reasoning transfer and novel-rule evaluation

**Status:** Living experiment note, updated 2026-09-22. Maths SFT followed by
KOR-Bench evaluation has been reported as successful. Quantitative results and
run references remain to be recorded here. [KOR-Bench integration](../integrations/korbench.md)
and the [six-task Reasoning Gym preparation pipeline](../integrations/reasoning-gym.md)
are implemented; the next training studies below remain proposals.

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

## Next-step priority: Reasoning Gym before FineWeb

Prioritise **Reasoning Gym training followed by KOR-Bench and maths evaluation**
for full fine-tuning, parameter-matched LoRA, and frozen-backbone Longshot. This
directly tests whether a transfer advantage persists when all methods receive
more varied procedural supervision, reducing dependence on the maths training
domain. It complements the maths-to-KOR experiment rather than replacing it. The
initial scope is now six tasks with deterministic worked solutions, as assessed
in the [implementation review](reasoning-gym-supervised-traces.md). This is a
selected procedural mixture, not broad coverage of the Reasoning Gym library.

| Training and evaluation | Main question | Interpretation |
|---|---|---|
| Reasoning Gym → KOR-Bench and maths | Does Longshot transfer reasoning improvements better after diverse procedural reasoning training? | A consistent advantage across maths-only and broad reasoning training would strengthen the evidence beyond one source dataset. |
| FineWeb → maths and KOR-Bench | Can Longshot acquire transferable reasoning improvements from general text without explicit reasoning supervision? | Strong transfer would support a broader claim, but weak transfer could reflect sparse reasoning signal, insufficient training, or the text mixture. |

FineWeb remains a valuable complementary experiment. Its potentially broader
positive result comes with a less diagnostic negative result. Prioritise it
first only if the central claim becomes learning transferable reasoning without
reasoning-specific supervision. General-text provenance alone does not establish
absence of reasoning content or benchmark exposure.

Broader Reasoning Gym training may bring the training distribution closer to
KOR-Bench. The [task gallery](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md)
includes logic, ciphers, and puzzles. Describe the experiment initially as
transfer from diverse procedural reasoning to KOR-Bench; audit overlap in rules
and procedures before claiming transfer to unseen reasoning families. A dataset
boundary alone does not establish that distinction.

To isolate the effect of **training breadth**, add a narrow Reasoning Gym mixture
with the same supervision format and budget. Comparing broad final-answer
training with historical maths worked-solution training changes both breadth and
supervision, so that comparison alone cannot attribute gains to breadth.

## PrOntoQA as an alternative

PrOntoQA is a strong alternative for controlled deductive generalisation. It
does not replace the training-breadth objective of Reasoning Gym. The
[upstream generator](https://github.com/asaparov/prontoqa) supplies questions
with explicit reasoning chains. PrOntoQA-OOD extends the original restricted
deductions to multiple inference rules and controls proof depth, width, and
composition; this adds diversity within deduction rather than across procedural
domains. See the [PrOntoQA-OOD paper](https://arxiv.org/abs/2305.15269).

| Consideration | PrOntoQA / PrOntoQA-OOD | Selected Reasoning Gym mixture |
|---|---|---|
| Training coverage | Deduction in a controlled logical setting | Rewriting, state tracking, spatial transformations, and arithmetic |
| Supervised targets | Generated proofs and answers | Deterministic execution traces and answers, rendered per task |
| Generalisation controls | Proof depth, width, and held-out inference-rule compositions | Task-specific difficulty, operation compositions, and family holdouts |
| Main scientific contribution | Systematic deduction and its transfer to external tasks | Transfer after more varied procedural supervision |
| Implementation | Data conversion is straightforward; strict holdouts and proof checking need work | Six-task implementation estimated at 4–6 hours; broader coverage requires more work |

PrOntoQA would provide a cleaner test of a specific generalisation ability.
Evaluate on fresh instances, longer proofs, and unseen compositions as well as
KOR-Bench and maths. Improvement on these controlled tests alongside external
transfer would be more diagnostic than KOR-Bench accuracy alone. Gold proofs
enable analysis of generated reasoning, but do not by themselves establish that
the processor uses the same internal mechanism.

PrOntoQA → KOR-Bench would complement maths → KOR-Bench by testing a second
training domain. It would still be a relatively narrow-source transfer study,
with a particularly close relationship to KOR-Bench's logic category. Inspect
category-level results rather than assuming uniform distance from training.

**Current recommendation:** retain the six-task Reasoning Gym study as the next
experiment, followed by PrOntoQA-OOD for controlled deductive generalisation and
FineWeb for transfer from general text. The preference over PrOntoQA is modest:
two of the six tasks involve rewriting, and the mixture primarily
teaches execution rather than proof search. It is justified by the current
objective of varying the reasoning training domain, not by an expectation of
larger gains. Before scaling, verify that the prepared mixture contains enough
distinct, nontrivial, complete examples and that each method can learn it. If
that requires expanding beyond the implementation budget, prefer PrOntoQA-OOD.
Prioritise PrOntoQA immediately if systematic deduction becomes the central
claim; prioritise FineWeb if learning without explicit reasoning supervision
becomes central.

## Reasoning Gym training setup

Use Reasoning Gym initially as an **offline data generator** for the existing
supervised training pipeline. Its [upstream interface](https://github.com/open-thought/reasoning-gym)
supports configurable difficulty, weighted task mixtures, and records containing
`question`, `answer`, and `metadata`. GCW's `src/gcw/data/sft.py` already accepts
`question`/`answer` rows, and `src/gcw/data/training.py` loads JSONL. All three
methods can consume identical frozen examples without a new model backend or
an RL training loop.

### Supervision choice

- **Final answers:** the simplest initial dataset for tasks with valid targets.
  Short answers provide fewer supervised computation steps than worked solutions
  and may encourage shorter responses. A weak transfer result would therefore
  need to be interpreted in light of the supervision choice.
- **Worked solutions plus final answers:** closer to the maths-training setting,
  and the recommended initial route. Selected generators retain intermediate
  states or reusable operations, allowing deterministic traces without a teacher.
  There is no uniform trace interface across the library. See the
  [source audit and local checks](reasoning-gym-supervised-traces.md) for task
  coverage, limits, and the remaining validation work.

This choice concerns target content. GCW's `sft_loss_scope: answer_only` masks
prompt tokens and can supervise either a final answer or an entire worked solution.
Select and validate tasks explicitly: for example, the upstream
[propositional-logic generator](https://github.com/open-thought/reasoning-gym/blob/main/reasoning_gym/logic/propositional_logic.py)
returns `answer: None`, with an example conclusion in metadata and a verifier.
Such tasks need an audited target-construction step or exclusion from the initial
mixture.

### Initial experiment design

1. Start with the **six audited tasks** in the implementation review, with
   deterministic worked solutions, controlled difficulty, and fixed mixture
   weights. Freeze the generated JSONL and retain generator revision,
   configuration, sample identities, and checksums. Expansion to 10–20 tasks is
   a later option outside the initial implementation estimate.
2. Start all methods from the same backbone checkpoint and use identical prepared
   examples and comparable tuning budgets. If continuing from maths SFT, use the
   same maths checkpoint for all methods. Match data exposure and report
   supervised tokens and compute separately.
3. Prepare fresh-instance validation for checkpoint selection and separate
   held-out-family and harder-instance evaluations. Check duplicate prompts and
   overlapping generator seed ranges: generators using `seed + index` can produce
   overlapping samples even with different base seeds. Audit tokenised lengths
   so GCW's sequence truncation does not remove complete targets.
4. Add a small verifier-backed Reasoning Gym evaluation to establish that each
   method learns the training tasks. Preserve metadata needed by the scorers and
   distinguish fully correct solutions from partial reward.
5. Keep KOR-Bench outside checkpoint and mixture selection. Report overall and
   category gains over the starting backbone, with untouched-backbone and
   processor-off controls. Retain maths evaluation to assess transfer back to the
   original application, and use multiple training seeds where feasible.

**Revised engineering estimate:** 4–6 hours for the six-task preparation path,
correctness tests, split and length checks, three training configs, and
documentation. This excludes training, GPU qualification, and a comprehensive
new evaluation suite. The [implementation review](reasoning-gym-supervised-traces.md)
separates observed feasibility from remaining work. Broad worked-solution
coverage is not covered by this estimate.

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

The rate assumption is informed by local Paper7 `gcw_runtime` records, which
are not distributed with the documentation. The 2B diffusion record
`results/qwen35_2b_fair_campaign_v1_diffusion_adapter22m_seed24_final_paper7_update_metrics_thinking-off.json`
reported about 142 output tokens/second on GSM8K and 119 on TheoremQA. The 27B
Longshot record
`results/qwen35_27b_instruct_longshot_fair_gate_rank1_seed24_step2000_paper7_thinking-off.json`
reported about 111 on GSM8K and 44 on SVAMP. These are task totals divided by
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
calendar time but not total GPU-hours. The estimates above have not yet been
updated with timings from the completed maths-to-KOR evaluation.

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

1. **Maths adaptation → KOR-Bench:** reported successful. Record quantitative
   results, run references, and measured runtime before using them to size the
   next campaign.
2. **Reasoning Gym → KOR-Bench and maths:** the next priority for comparing transfer
   after varied procedural supervision. Begin with the six-task trace mixture;
   qualify data quality and learning before a large campaign, and include
   Reasoning Gym validation and controlled holdouts.
3. **PrOntoQA-OOD → controlled holdouts, KOR-Bench, and maths:** train on bounded
   proof complexity, then test fresh instances, longer proofs, held-out
   compositions, and premise-change pairs. ProofWriter offers a complementary
   deduction benchmark.
4. **FineWeb → maths and KOR-Bench:** a complementary test of transfer from
   general-text training. A fixed maths/general mixture can follow as a separate
   condition, but cannot isolate general-text transfer.

Still to choose: backbone/processor variant, token/compute budget, training targets,
task mixture, precise splits, primary metric, and seed count. Update this
note with pinned dataset versions and results as these are settled.
