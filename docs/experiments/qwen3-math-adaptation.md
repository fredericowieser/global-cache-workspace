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

Evaluation has two lanes. Full standard general benchmarks use no chat template and option log-likelihood once per model/processor state. Separate chat-generation cells run thinking off/on: labeled-choice extraction for general multiple-choice tasks and numeric-last-answer scoring for math tasks, always with a 2,048-token generation cap. Large general tasks use authenticated deterministic subsets for the secondary thinking comparison; standard general scores remain full-dataset. LAMBADA is excluded because next-word completion has no meaningful thinking-on analogue. Reports retain processor invocation count, elapsed time, peak VRAM, parse and truncation rates, trainable parameters, generated tokens, throughput, and FLOPs.

Processor-off results test the frozen-backbone invariant. Any material difference from the untouched instruct checkpoint indicates checkpoint or evaluation drift.

## Full single-seed series

`scripts/run_math_experiments_full_single_seed.py` declares the first full-epoch series. It evaluates the untouched instruct model, then trains and evaluates full SFT, BT, BGT, noisy Diffusion-BGT, Longshot, Longshot internal recurrence ×4, Longshot external recurrence ×4, LoRA rank 32, and Flow-BGT in that order.

The series uses Warmup–Stable–Decay (WSD): linear warmup for 3% of optimizer steps, a constant peak learning rate for 87%, then linear decay to zero for 10%. Full SFT uses `5e-6`; LoRA and every frozen-backbone processor use `1e-4`. A unified processor rate prevents mechanism-specific optimizer tuning from becoming an architectural advantage.

The 697,744-row split leaves a 16-row remainder at global batch 64. The series uses standard drop-last epoch semantics: 697,728 rows and 10,902 optimizer steps. Every arm sees the same retained rows once, in the same order. The script defaults to a read-only JSON plan and requires `--execute` before it creates artifacts or launches work. It checkpoints exact-resume state, stops launching new cells at the declared deadline, and resumes completed training and evaluation cells on the next invocation.

Every non-processor target gets one full standard-general cell plus thinking off/on chat-generation cells. Every processor target gets those three cells with processor off and on. Training metrics and FLOP estimates remain in local run records and mirror to W&B. Evaluation results remain in local evaluation records and publish scalar metrics plus authenticated result artifacts to W&B.

```bash
uv run --extra gpu --extra eval --extra convert python \
  scripts/run_math_experiments_full_single_seed.py \
  --model /path/to/Qwen3-0.6B \
  --backbone-checkpoint /path/to/native-backbone \
  --data-manifest /path/to/manifest.json \
  --experiment-root /path/to/experiment-root
```

The command above only prints the resolved plan. Add `--execute` after inspecting it.

## Staging gates

1. Publish the data retention and contamination manifest.
2. Record the explicit 4K/8K trade-off and approve one cap before execution. **Approved: 4K.**
3. Convert and hash the exact instruct checkpoint.
4. Run one-step CUDA canaries for every arm.
5. Screen bounded learning rates on the source-balanced held-out SFT split; select by final held-out answer-only loss, never benchmark test score.
6. Train the canonical one-seed matrix.
7. Replicate only the strongest mechanisms with three seeds.
