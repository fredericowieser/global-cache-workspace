# SmolLM 360M Matched-Total 20B Runs

Goal: compare TF, BT, and BGT at the same 20B-token data budget, with BT/BGT total non-embedding params kept below the trained SmolLM TF baseline.

Baseline target is the native SmolLM TF config in this repo:

- sequence length: 2048
- training steps: 38,147
- local batch per GPU: 32
- GPUs: 8
- total tokens: 20,000,014,336
- FLOP convention: PPT-style analytical forward FLOPs per token, multiplied by 3 for forward+backward

| model | backbone layers | backbone params | processor params | total non-embed params | delta vs TF | processor/backbone | train FLOPs/token | total train FLOPs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF | 32 | 314,635,200 | 0 | 314,635,200 | 0 | 0.00% | 2.925527040e9 | 5.851058274e19 |
| BT matched | 30 | 294,970,560 | 12,844,830 | 307,815,390 | -6,819,810 | 4.35% | 2.876912640e9 | 5.753829404e19 |
| BGT matched | 30 | 294,970,560 | 13,967,041 | 308,937,601 | -5,697,599 | 4.74% | 2.881017600e9 | 5.762039330e19 |

BT and BGT share the exact same backbone: 30 layers, hidden size 960, intermediate size 2560, 15 attention heads, 5 KV heads, head dimension 64.

The processors are below the requested 5% cap. The totals are under the TF baseline because the next larger shared backbone, 31 layers, leaves too little room for a near-5% processor while staying below the baseline parameter count.

Training configs:

- `configs/train/smollm360m_tf_20b_wandb.yaml`
- `configs/train/smollm360m_matched_bt_20b.yaml`
- `configs/train/smollm360m_matched_bgt_20b.yaml`

W&B tracking:

- project: `global-cache-workspace`
- group: `smollm360m-matched-total-20b`
- train metrics every optimizer step: loss, perplexity, LR, LR scale, tokens, FLOPs, progress, grad norm, throughput
- validation metrics every 250 steps and at step 0: loss, perplexity, evaluated tokens
- validation window: 10 batches per rank, matching 5,242,880 tokens on 8 GPUs
