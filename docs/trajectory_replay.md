# GSM8K trajectory replay for a frozen Longshot backbone

## Objective

`trajectory_replay` trains only `model.cache_processor`. The backbone is frozen and remains in eval mode.
For each GSM8K prompt, the processor deliberates for a sampled depth `k ∈ {1, 2, 4}` and writes one
prompt-conditioned cache. Two teacher-forced continuations share that cache:

1. **Thinking on:** the tokenizer's exact thinking-on assistant trigger, the full canonical GSM8K rationale,
   `</think>`, then the boxed answer.
2. **Thinking off:** the tokenizer's exact thinking-off assistant trigger, then the boxed answer directly.

The loss is

```text
L = process_weight * CE(canonical rationale + thinking close)
  + answer_weight * 0.5 * (
        CE(answer | thinking-on rationale replay)
      + CE(answer | thinking-off direct response)
    )
```

Each CE is independently token-normalised. Long rationales therefore do not drown the two answer terms.
The rationale target is a full causal trajectory rather than a set of isolated "step N" branches. This matches
autoregressive generation: later rationale tokens condition on earlier canonical tokens. The route does not optimise
a matched-vs-shuffled hinge. Shuffled workspaces remain a held-out diagnostic only.

A KL to the unprocessed frozen backbone was deliberately omitted. That model is the student backbone, not a
stronger teacher; matching its baseline distribution would regularise the processor toward doing nothing. Canonical
rationale replay supplies the useful token-level teacher signal available in the upstream GSM8K file.

## Data

Prepare the immutable 7,473-row artifact:

```bash
.venv/bin/python scripts/prepare_gsm8k_trajectory_replay.py
```

Inputs and outputs:

- Source: `data/exploratory/gsm8k-upstream-train.jsonl`
- Records: `data/process_branch/gsm8k-trajectory-replay-v1.jsonl`
- Manifest: `data/process_branch/gsm8k-trajectory-replay-v1.manifest.json`

The converter preserves exact questions and canonical numeric final answers, removes GSM8K `<<calculator>>`
annotations from the visible rationale, and rejects malformed rows. Observed preparation accepted all 7,473 rows.
The frozen split is rows `0:6000` for training, `6000:6256` for held-out loss controls, and `6256:7473`
untouched.

## H200 training

The pinned config is:

```text
configs/train/process_branch_qwen3_06b/
  qwen3_06b_trajectory_replay_random_depth_6000_seed24.yaml
```

On a single H200 checkout:

```bash
REPO=/path/to/global-cache-workspace \
  bash scripts/run_trajectory_replay_h200.sh
```

Or invoke training directly after preparing and authenticating the data:

```bash
uv run --locked --extra gpu --extra convert python scripts/train.py \
  --config configs/train/process_branch_qwen3_06b/qwen3_06b_trajectory_replay_random_depth_6000_seed24.yaml
```

The full arm performs 1,500 AdamW steps at effective batch 4, exactly one pass over the 6,000 training rows.
The runner refuses overwrite, requires exactly one H200, verifies data identities, runs offline, and checks that the
backbone hash is unchanged, the processor hash changed, and the final checkpoint reload succeeded.

## Evaluation expectation

Use the same Qwen chat template and prompt suffix as training when comparing thinking on/off. The scientific signal
is exact-answer GSM8K generation under both assistant triggers, with processor OFF/INIT/K1/K2/K4 controls. The
objective explicitly balances the two modes; neither answer policy is a secondary diagnostic. Generation accuracy,
not held-out teacher-forced CE, selects the checkpoint and depth.
