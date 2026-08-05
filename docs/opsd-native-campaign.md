# OPSD native full-trajectory campaign v2

## Question

Can a frozen Qwen3-0.6B cache processor learn the answer distributions induced by the same backbone's native hidden reasoning, then improve thinking-off benchmark performance?

The strict teacher response is:

```text
<think> hidden reasoning </think> visible answer EOT
```

Malformed, truncated, delimiter-unbalanced and non-terminal-EOT trajectories are rejected. The objective is the equal mean of exact shared-prefix forward KL on the student-visible and teacher-visible trajectories. Hidden thinking tokens are teacher context only.

## Training dose

The comparison reference is `configs/train/bottleneck_llm/qwen3_06b_bgt_from_sft_frozen.yaml`.

| Field | Reference SFT | Native distillation |
|---|---:|---:|
| LR | 1e-4 | 1e-4 |
| Optimiser steps | 1,000 | 1,000 |
| Per-rank examples | 16 | 16 effective examples by accumulation |
| SFT global examples on 8 GPUs | 128 | 16 on one GPU |
| Sequence length | 512 | up to 8,192 |

The native run matches the reference per-rank batch, optimiser-step count and LR. It does **not** claim the 8-GPU SFT global batch of 128. Native autoregressive trajectories are currently samplewise and single-GPU.

Configs:

- smoke: `qwen3_06b_opsd_native_full_smoke_v2.yaml`;
- 1k: `qwen3_06b_opsd_native_full_steps1000_bs16_v2.yaml`;
- 10k: `qwen3_06b_opsd_native_full_steps10000_bs16_v2.yaml`.

All runs use fresh no-overwrite outputs and online W&B group `opsd-native-full-v2`.

## Recorded SFT-backed smoke result

The reviewed SFT-backed smoke at source `d9615c837d96e626106e7f95e787a267e2c9fbd6` ran on an H100 as Slurm job `22457` for 9m47s. It attempted all 64 configured rows and produced:

- strict valid trajectories: **0**;
- rejected trajectories: **64**;
- rejection: all 64 violated the exactly-one thinking-close-delimiter requirement;
- W&B: `https://wandb.ai/huawei-noahs-ark/global-cache-workspace/runs/b9ucjryd`;
- terminal state: `FAILED`, exit `1:0`, with no `SUCCESS.json` or checkpoint.

This is a fail-closed scientific negative for the staged SFT checkpoint. The 1k, evaluation and 10k stages were not launched.

A distinct fallback may use the official native-thinking `Qwen/Qwen3-0.6B` base checkpoint. It is **not** a continuation of the failed SFT-backed protocol and uses `qwen3_06b_opsd_native_base_smoke_v1.yaml`, followed only on smoke promotion by `qwen3_06b_opsd_native_base_steps1000_bs16_v1.yaml`. Before either run, the HF→GCW conversion must pass the fixed policy in `scripts/verify_hf_conversion.py`: BF16 model parameters/internal compute, the native model's intentional FP32 output logits, explicit FP32 comparison, `max_abs <= 0.25`, `mean_abs <= 0.01`, final-token top-1 exact, and complete model/tokenizer/checkpoint manifests. Every base training allocation re-executes parity immediately before optimisation and revalidates that fresh evidence after training; a stored JSON claim alone cannot authorize success.

The reviewed base conversion at source `71aa72fe2df7a82bc2b416dd8a677b3ae81c6a32` completed as job `22464` in 1m35s with authenticated stable source hashes. The corrected schema-3 parity run at source `b31ff9636b62859c0ab944cbc2678b44bf8d2355` completed as job `22466` and failed the fixed numerical gate: max absolute error `0.234375` (limit `0.25`), mean absolute error `0.035964` (limit `0.01`), final-token top-1 exact, full-token top-1 fraction `1.0`. Its evidence SHA-256 is `1aca29e6db3099d652d8635d57f8f79a9acccfbec915a378c0477e7104b86216`. No base smoke, 1k run or evaluation was launched; do not relax the parity threshold.

The authenticated five-prompt diagnostic at source `fbb55d8f886021e14db0e3a8be3f54b1892e376e` completed as job `22525` in 51s. Evidence SHA-256: `a02e037e2cb666f6ca38dc7e4c07e193be8e4e70b0c89b6beb95213a7ef94821`. Vocabulary centring retained `77.1–88.3%` of raw mean absolute logit error (`82.6%` aggregate), so a softmax-invariant offset explains only a minority of the mismatch. Mean KL was `0.000750–0.001523` nats/token; mean total variation was `1.39–1.95%`; top-1 agreement remained `100%`. Embeddings were exact. The first measured divergence was the **output** of decoder `layer_00` for every prompt (`0.5427–0.6565%` relative L2). This does not localise an operation inside layer 0 or distinguish BF16 kernel/order effects from a parameter, architecture or implementation mismatch. The failed parity gate remains authoritative; no smoke or training is authorised.

## Promotion gates

### Smoke → 1k

All must hold:

1. at least one strict native trajectory is valid within 64 attempted rows;
2. one finite optimiser step completes on an H100;
3. backbone gradient count is zero;
4. processor state changes and checkpoint fidelity verifies;
5. teacher/student visible responses terminate with EOT;
6. projected 1k walltime fits one five-day allocation with 10% finalisation headroom.

A zero-yield smoke is evidence that the staged SFT checkpoint does not implement the proposed native teacher. Do not relabel EOT as a thinking-close delimiter or weaken the filter. Diagnose the real prompt/template/checkpoint before defining another protocol.

### 1k → full 2×2 evaluation

All must hold:

1. exactly 1,000 optimiser steps and effective batch 16;
2. finite loss/KL/gradient/gate metrics in local JSONL and W&B;
3. non-zero cache displacement and changed processor hash;
4. source, tokenizer, traces and checkpoint remain authenticated;
5. no unreported invalid-trace or walltime failure.

### Full evaluation → 10k

Evaluate all examples from GSM8K, SVAMP, GSM-Hard, LogiQA and Gaokao under:

LogiQA and Gaokao use source-controlled `generate_until` variants that require a final answer letter. This is necessary for a real CoT on/off intervention; their scores are **not** presented as canonical multiple-choice log-likelihood leaderboard results.

| Condition | CoT | CP |
|---|---:|---:|
| A | off | off |
| B | off | on |
| C | on | off |
| D | on | on |

Launch 10k only if:

1. W&B and local metrics show last-100-step mean unclipped KL at least 10% below first-100-step mean;
2. deployment effect `B-A` has positive five-task macro and is positive on at least three tasks;
3. no task loses more than 3 percentage points under `B-A`;
4. the full matrix has matching manifests, complete sample counts and valid processor-event evidence;
5. a continuation plan can survive the five-day Slurm walltime without discarding optimiser state.

If these gates fail, preserve the negative result and diagnose objective, teacher validity or systems throughput. Do not launch 10k merely to consume more compute.

## Unattended execution

- Training wrapper: `scripts/slurm/opsd_native_train.sbatch`.
- Base conversion wrapper: `scripts/slurm/convert_qwen3_base.sbatch`.
- Base conversion parity wrapper: `scripts/slurm/verify_qwen3_base_conversion.sbatch`.
- Full matrix wrapper: `scripts/slurm/opsd_full_matrix.sbatch`.
- Evaluation W&B publisher: `scripts/log_opsd_matrix_wandb.py`.

A run is complete only when Slurm exits zero and the output contains a validated `SUCCESS.json`. Submission, `RUNNING`, a W&B URL or a final checkpoint directory alone is not success.
