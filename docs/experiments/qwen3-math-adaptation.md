# Qwen3 math adaptation experiment

**Question.** Given one Qwen3-0.6B Instruct checkpoint and one ordered math dataset, which adaptation architecture improves math reasoning without destroying general ability?

## Fixed contract

Every arm uses sequence length 4,096, global batch 64 rows, the same optimizer-step count, dataset bytes, row order, tokenizer, starting checkpoint, answer-only objective, and seed. Physical microbatch and gradient accumulation may differ. FLOPs are reported, not matched.

**4K decision.** The 2026-08 retention audit found that 4,096 retains 94,961/154,780 admissible OpenR1 rows and 31.8% of their target-token mass; 8,192 retains 135,253 rows and 66.3% of target-token mass. The experiment explicitly accepts the lost long-trace mass to keep the 11-arm screen tractable. This is a compute trade-off, not a claim that 4K retains most reasoning tokens.

`scripts/prepare_math_experiment.py` builds the natural admissible union of OpenMathInstruct-2 and verified OpenR1-Math. It excludes exact normalized-prompt overlap with the generative math benchmarks, rejects duplicate prompts, and admits only rows whose complete rendered prompt and response fit the context cap. Multiple-choice and language-model benchmarks are not mathematically prompt-equivalent to the SFT rows; their revisions remain pinned in evaluation evidence rather than the exact-prompt filter. The manifest records retention at 1,024, 2,048, 4,096, and 8,192 tokens.

`scripts/generate_math_experiment_configs.py` writes the fairness contract and named training arms. Training refuses a config whose dataset hash, path, tokenizer, backbone, sequence length, global row batch, or steps differ from that contract. Canonical runs validate at start and ten evenly spaced points using 64 held-out batches. They overwrite one exact-resume checkpoint only at completion; retained intermediate checkpoints require an explicit `GCW_RETAIN_CHECKPOINT_EVERY` override.

## Main arms

- full-backbone SFT;
- LoRA rank 32;
- BT;
- BGT;
- Icarus;
- Longshot;
- Longshot internal recurrence ×4;
- Longshot external recurrence ×4;
- Flow-BGT with four field evaluations;
- Diffusion-BGT with four refinements and noise 0.15;
- matched deterministic Diffusion-BGT control with noise zero.

All runs mirror to `huawei-noahs-ark/global-cache-workspace`. Local run artifacts remain canonical.

## Evaluation

Evaluation protocol V3 has two lanes. HellaSwag, PIQA, WinoGrande, ARC Easy, ARC Challenge, MMLU, LogiQA, and AGIEval Gaokao MathQA use their full LM Evaluation Harness datasets, plain prompts, no chat template, and benchmark-native option likelihood. These scores have no thinking dimension. Gaokao follows the official harness rule: its selected argmax option counts as correct when it belongs to the gold set, including the seven multi-answer rows. GSM8K OpenMath, SVAMP, and GSM8K Hard use the official Qwen chat template with thinking off/on, deterministic generation, a 2,048-token cap, and numeric-last-answer scoring. Reports retain processor invocation count, elapsed time, peak VRAM, parse and truncation rates, trainable parameters, generated tokens, throughput, and FLOPs.

Completed V2 multiple-choice chat-generation results remain supplementary discontinued diagnostics. They never enter V3 aggregate tables. Valid canonical-likelihood and numeric-generation artifacts may be reused byte-for-byte when their hashes and protocol contracts pass validation; reuse manifests record the source path and digest. LAMBADA remains excluded because next-word completion has no meaningful thinking-on analogue.

Processor-off results test the frozen-backbone invariant. Any material difference from the untouched instruct checkpoint indicates checkpoint or evaluation drift. High-throughput generation through separate batch controls, padded native decoding, continuous batching, or optimized kernels is deferred to a separately versioned protocol; see [Deferred extension: high-throughput generation](../guides/evaluate.md#deferred-extension-high-throughput-generation).

## Full single-seed series

`scripts/run_math_experiments_full_single_seed.py` declares the expanded 11-arm full-epoch series. It covers full SFT, LoRA rank 32, BT, BGT, Icarus, noisy and noiseless Diffusion-BGT, Longshot, Longshot internal recurrence ×4, Longshot external recurrence ×4, and Flow-BGT. The completed version-1 full-SFT checkpoint remains immutable and may be imported into the additive matrix by digest; historical version-1 configs and evaluation evidence are never rewritten.

The series uses Warmup–Stable–Decay (WSD): linear warmup for 3% of optimizer steps, a constant peak learning rate for 87%, then linear decay to zero for 10%. The imported full SFT arm used `5e-6` and LoRA uses `1e-4`. The expanded version-2 matrix uses the user-approved common `1e-3` peak for every frozen-backbone CacheProcessor arm. This rate is a new protocol rather than a continuation of the earlier `1e-4` selection, so production launches require fresh finite-loss and gradient gates.

All canonical launch configs use eager execution. Rejected adaptive grouping, regional compilation, CUDA Graph, pair-wave, and Triton candidates remain disabled; the deployable performance multiplier is `1.00x`.

The 697,744-row split leaves a 16-row remainder at global batch 64. The series uses standard drop-last epoch semantics: 697,728 rows and 10,902 optimizer steps. Every arm sees the same retained rows once, in the same order. The script defaults to a read-only JSON plan and requires `--execute` before it creates artifacts or launches work. It checkpoints exact-resume state, stops launching new cells at the declared deadline, and resumes completed training and evaluation cells on the next invocation.

Every non-processor target gets full canonical-likelihood coverage plus thinking off/on numeric-generation cells. Every processor target gets the same evidence with processor off and on. Multiple-choice chat generation is not scheduled. Training metrics and FLOP estimates remain in local run records and mirror to W&B. Evaluation results remain in local evaluation records and publish scalar metrics plus authenticated result artifacts to W&B.

### Completed H100 full-SFT numeric result

Protocol V3 numeric generation for the completed 10,902-step full-SFT checkpoint is complete on the `lrc-h100-v1` lineage. Thinking modes remain separate metrics:

| task | thinking off | thinking on | on − off | truncation off / on |
| --- | ---: | ---: | ---: | ---: |
| GSM8K OpenMath | 890/1,319 (67.48%) | 862/1,319 (65.35%) | −2.12 pp | 1.06% / 29.26% |
| SVAMP | 239/300 (79.67%) | 239/300 (79.67%) | 0.00 pp | 2.33% / 21.00% |
| GSM8K Hard | 427/1,319 (32.37%) | 303/1,319 (22.97%) | −9.40 pp | 6.14% / 72.78% |

Thinking-on generation does not improve this full-SFT checkpoint under the fixed 2,048-token protocol; GSM8K Hard is dominated by truncation. H100 and H200 lineages are not averaged. Result-file SHA-256 values (off/on) are: GSM8K OpenMath `2be8e334f5b9e6edc3ca211d2dd2cada7d74c411277d777d06a3647f492b076e` / `3867ee6cc4f1ace5de48fa7317663f2d9c8acc90df8bd5ae7f9f2e9ca7384302`; SVAMP `69fa3ed0e3f0aa07b29a55304d551d38c633f8bb7aedd4ab3ae26b7a6425fbb0` / `f8498c01d66978dbcf923c74f36e2b1ecd0f6af5c74f96955d2f78d50110e5b0`; GSM8K Hard `6de6257fb5e74bce4c4a91fc463af9a6e0ef8d4f2f7fef9f80fbaa05d304e725` / `ed9e3e41d95a151b9b606ec4fba891c0fb77a7add9b3d8d335e828c8d63afbf9`.

```bash
uv run --extra gpu --extra eval --extra convert python \
  scripts/run_math_experiments_full_single_seed.py \
  --model /path/to/Qwen3-0.6B \
  --backbone-checkpoint /path/to/native-backbone \
  --data-manifest /path/to/manifest.json \
  --experiment-root /path/to/experiment-root \
  --import-full-sft-bundle /path/to/full-sft/evaluation-bundle
```

The command above only prints the resolved plan. Add `--execute` after inspecting it.

## Dual-GPU training and evaluation lanes

The campaign can isolate training and evaluation on separate hosts. The training GPU owns optimizer state and publishes a minimal authenticated evaluation bundle only after an arm reaches its exact final step. The evaluation GPU verifies the bundle before loading it. Git synchronizes code at one pinned clean commit; bundle manifests synchronize checkpoint evidence.

```bash
# Training host: train one arm and atomically publish its evaluation bundle.
uv run --extra gpu --extra eval python scripts/run_math_training_lane.py \
  --arm bt \
  --config /path/to/experiment/configs/bt.yaml \
  --experiment-root /path/to/experiment \
  --bundle-outbox /path/to/evaluation-outbox \
  --campaign-plan /path/to/experiment/plan.json \
  --expected-gpu H100

# Evaluation host: verify and evaluate the transferred bundle.
uv run --extra gpu --extra eval python scripts/run_math_evaluation_lane.py \
  --arm bt \
  --checkpoint-bundle /path/to/inbox/bt \
  --tokenizer-model /path/to/Qwen3-0.6B \
  --experiment-root /path/to/experiment \
  --evaluation-lineage lrc-h100-v1
```

Both commands are dry-run by default and require `--execute`. `scripts/math_evaluation_bundle.py` can create or verify bundles independently. A bundle contains only `config.json`, `model.safetensors`, and `train_state.json`; its manifest records source commit, final step, training-config digest, original checkpoint-manifest digest, file sizes, and file hashes. Transfer into a temporary directory, verify it, then rename atomically. Never transfer optimizer state to the evaluator or evaluate a partial directory.

Before hardware cutover, run the untouched baseline and completed full-SFT checkpoint on both evaluator lineages. `scripts/compare_math_evaluation_lineages.py` enforces exact row identity and the declared aggregate, parsing, truncation, and paired-prediction parity bands. H100 evidence uses distinct paths, W&B IDs, artifacts, host metadata, and lineage tags. H200 evidence remains valid secondary evidence rather than being overwritten.

Allow at most one training arm ahead of accepted evaluation. The training host may train arm *N+1* while arm *N* is evaluated, but it must not start arm *N+2* after an evaluation or transfer failure. An evaluation OOM restarts the entire affected cell at a declared lower batch. Training recovery must preserve global batch 64, row order, and optimizer semantics.

## Staging gates

1. Publish the data retention and contamination manifest.
2. Record the explicit 4K/8K trade-off and approve one cap before execution. **Approved: 4K.**
3. Convert and hash the exact instruct checkpoint.
4. Run one-step CUDA canaries for every arm.
5. Screen bounded learning rates on the source-balanced held-out SFT split; select by final held-out answer-only loss, never benchmark test score.
6. Train the canonical one-seed matrix.
7. Replicate only the strongest mechanisms with three seeds.
