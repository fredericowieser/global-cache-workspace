# Qwen3 math adaptation experiment

**Question.** Given one Qwen3-0.6B Instruct checkpoint and one ordered math dataset, which adaptation architecture improves math reasoning without destroying general ability?

## Fixed contract

Every arm uses sequence length 4,096, global batch 64 rows, the same optimizer-step count, dataset bytes, row order, tokenizer, starting checkpoint, answer-only objective, and seed. Physical microbatch and gradient accumulation may differ. FLOPs are reported, not matched.

**4K decision.** The 2026-08 retention audit found that 4,096 retains 94,961/154,780 admissible OpenR1 rows and 31.8% of their target-token mass; 8,192 retains 135,253 rows and 66.3% of target-token mass. The experiment explicitly accepts the lost long-trace mass to keep the 11-arm screen tractable. This is a compute trade-off, not a claim that 4K retains most reasoning tokens.

`scripts/prepare_math_experiment.py` builds the natural admissible union of OpenMathInstruct-2 and verified OpenR1-Math. It excludes exact normalized-prompt overlap with the generative math benchmarks, rejects duplicate prompts, and admits only rows whose complete rendered prompt and response fit the context cap. Multiple-choice and language-model benchmarks are not mathematically prompt-equivalent to the SFT rows; their revisions remain pinned in evaluation evidence rather than the exact-prompt filter. The manifest records retention at 1,024, 2,048, 4,096, and 8,192 tokens.

`scripts/generate_math_experiment_configs.py` writes the fairness contract and named training arms. Training refuses a config whose dataset hash, path, tokenizer, backbone, sequence length, global row batch, or steps differ from that contract.

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

Every processor checkpoint runs processor off/on × thinking off/on across the maintained general and math suites. Full-backbone and LoRA checkpoints run both thinking modes. Reports retain processor invocation count, elapsed time, peak VRAM, trainable parameters, generated tokens, throughput, and FLOPs.

Processor-off results test the frozen-backbone invariant. Any material difference from the untouched instruct checkpoint indicates checkpoint or evaluation drift.

## Staging gates

1. Publish the data retention and contamination manifest.
2. Record the explicit 4K/8K trade-off and approve one cap before execution. **Approved: 4K.**
3. Convert and hash the exact instruct checkpoint.
4. Run one-step CUDA canaries for every arm.
5. Screen learning rates on held-out SFT data.
6. Train the canonical one-seed matrix.
7. Replicate only the strongest mechanisms with three seeds.
