# Qwen3.5 ablation decisions

These experiments test answer-only adaptation of Qwen3.5-2B. The inventory contains **39 production training runs**, including partial and cancelled runs. Different inference depths are evaluations of a checkpoint, not new training runs. See the [complete comparison](qwen35-results/comparison.md), [CSV](qwen35-results/comparison.csv), and [JSON](qwen35-results/comparison.json).

## What the results support

Keep ordinary one-pass Longshot as the baseline. Across three training seeds, its eight-task likelihood accuracy was **55.24%**, compared with **54.42%** for time-conditioned diffusion with training perturbation. Thinking-on numeric accuracy was similar: **65.06% versus 65.23%**. These results do not establish diffusion superiority.

Keep scheduled diffusion and independent horizon/solver controls to test the mechanism. They do not establish test-time compute scaling: the same H50 checkpoint scored **67.18%, 65.95%, and 62.96%** on thinking-on numeric accuracy at K=1, 4, and 32. Additional numerical refinement did not improve that task score.

The common answer-NLL comparison used 1,024 development examples (322,414 answer targets) and 4,096 test examples (1,293,940 targets):

| Treatment | Trainable parameters | Test NLL ↓ | Difference from control |
|---|---:|---:|---:|
| Rank-one H50/K4 control | 22,131,840 | 0.344702 | — |
| KV-LoRA16 + Longshot | 24,982,656 | **0.308065** | −0.036638 |
| Untied four stages | 43,382,784 | **0.339152** | −0.005551 |
| Adaptive full-cache rereading | 22,576,128 | 0.344236 | −0.000467 |
| Persistent workspace | 22,131,841 | 0.345108 | +0.000406 |
| Read1/write4 | 22,427,520 | 0.345570 | +0.000868 |
| Read4/write1 | 22,476,672 | 0.348601 | +0.003898 |

**KV-LoRA produced the largest measured NLL improvement**, about 10.6%. It relaxes the frozen effective backbone: original weights stay frozen, but adapters change K/V projections. We have not separately trained a KV-LoRA-only control, so we cannot attribute the gain to combining LoRA with Longshot. Disabling the processor after joint training does not supply that control.

Untied stages improved NLL by about 1.6%, with almost twice the processor parameters. This supports a capacity control, not fixed-checkpoint depth scaling. Archive its implementation rather than maintain the workspace experiment family. Rereading's small single-seed gain does not justify its complexity; persistence and higher ranks showed no gain here.

## Retained capabilities and retired experiments

| Component | Decision |
|---|---|
| Ordinary Longshot and upstream routing | Preserve as supported functionality. Our local selector experiments do not justify replacing upstream routing. |
| Scheduled diffusion; horizon and solver depth; time/noise controls | Retain with explicit compute accounting and unchanged disabled defaults. |
| General LoRA | Preserve ordinary named-projection adaptation and joint processor training. |
| KV-only LoRA | Retain as a small targeting extension, not a separate adaptation framework. |
| Live rank-one initialization | Keep separate from solver and adapter changes. The experiments used zero value rows to preserve zero initial writes and live key rows to permit a first value gradient. |
| Fixed/refreshed local routing, higher-rank probes/writers, untied/readout/reread/persistent workspace variants | Retire maintained local implementations; preserve results and original source. |
| Sampled training depth and 256k-unique-data campaigns | Archive. The sampled-depth downstream pipeline failed; neither campaign establishes a benefit that warrants permanent machinery. |
| Allocation-specific launchers, monitors, smoke wrappers and campaign import chains | Retire. Keep numerical evaluation and useful regression tests in their normal library/test owners. |

Cancellation is not a final negative result. Direct readout reached step 574, retained checkpoint 500, and received no held-out evaluation. Some learned-routing evaluations also remained cancelled or partial. Missing results stay missing.

## Baselines and the gate contrast

General all-linear LoRA remains a necessary baseline. It scored **58.05%** on the eight-task likelihood aggregate; the historical joint all-linear LoRA/diffusion run scored **57.30%**. Their generation records lack explicit EOS/PAD metadata and report 100% truncation, so those generation scores are not clean stopping-matched comparisons.

Tuned full SFT at LR 1e-5 reached **73.77%** numeric accuracy with thinking off and **72.99%** with thinking on. Do not use the poorly tuned historical LR 1e-4 run to claim an advantage over SFT. Common-split NLL for tuned SFT and ordinary all-linear LoRA remains missing.

The fresh ordinary Longshot sigmoid-gate −2 run completed 2,000 steps. Its final printed training EMA was **0.3305**, and the mean of printed values at steps 1951–2000 was **0.335148**; the fixed-one seed24 control gave **0.3339** and **0.338752**. The BF16 scalar finished at **−2.0**, with FP32 sigmoid **0.1192029193**. Small updates rounding away are consistent with this observation. There is no held-out result and no evidence of learned gate adaptation. Keep the result; do not select a new default from training loss alone.

## Protocol limits

- Printed `loss=` is bias-corrected training-CE EMA with beta 0.9. The last50 measure averages printed EMA values, not raw minibatch losses. Held-out NLL and benchmark accuracy are separate metrics.
- Training generally used seed24, 2,000 steps, BF16, LR 1e-4, 16×8 accumulation at length512, and the same 128k rows twice. Tuned SFT used a different physical microbatch and learning rate. Matched steps/data does not mean matched compute.
- The common dev/test data excluded normalized prompts from the defined training ranges. These splits informed later design decisions; they are not untouched confirmatory holdouts. Most architecture contrasts have one training seed.
- Legacy generation caps, cadence, stopping rules, and batch sizes differ. The full table labels those differences and separates checkpoint overrides from separately trained models.
- Native processor-only training resume still lacks qualified omitted-backbone restoration. Upstream's configuration migration does not repair that path. Consolidation adds no resume qualification or new evaluation results.
- Historical checkpoints require their matching source/configuration. Upstream changed cache-state ordering and removed legacy workspace support. Do not silently translate retired experimental settings into current defaults; use their frozen source for reproduction.

## Current-runtime H50/K4 training

`configs/train/qwen35_2b_diffusion_longshot_h50_k4.yaml` is the retained shared-weight, frozen-backbone recipe: no LoRA, zero initial writes with live key factors, H=50/K=4, cadence64, BF16, and 2,000 answer-only SFT steps. It uses the raw native Qwen3.5-2B backbone, not an SFT checkpoint.

From the repository root, inside a GPU allocation with the GPU/Qwen3.5 dependencies already installed:

```bash
uv run --locked --no-sync --extra gpu --extra qwen35 gcw train \
  --config configs/train/qwen35_2b_diffusion_longshot_h50_k4.yaml
```

Check the cluster-specific `backbone_checkpoint` and `data.dataset` paths, W&B authentication, and select a fresh `output_dir` before each run. This recipe passes current model/experiment admission; it is not a new GPU qualification or an exact frozen-source reproduction of historical scores. Keep `resume: null`: processor-only training resume remains unqualified.

## Evidence and reproduction

The original report, configurations, campaign scripts, tests, pre-consolidation source, and gate completion receipts are retained in an authenticated archive:

- Remote: `/mnt/jfs-00/team-agent/home/f50056118/gcw-archives/qwen35-consolidation-20260908/research-evidence.tar.gz` on `lrc`.
- Local artifact location: `data/results/qwen35-consolidation-20260908/` (ignored run evidence, not library source).
- SHA256: `733e3e1c0c86987abf556f5c9fef5e84e8ebe6c6ee3e29ca3597e0caa986896d`.
- The adjacent `archive.json` lists member hashes. [Published table hashes](qwen35-results/evidence.json) bind the compact documentation files.

For historical reproduction, use the archived exact configuration and its recorded frozen cluster source, dataset/tokenizer identities, and checkpoint. Do not substitute a newly generated recipe and label it the same experiment. The archive is historical evidence, not a qualification of the consolidated runtime.

No new project-root report folder, remote evaluation, training run, or checkpoint mutation forms part of this cleanup.
