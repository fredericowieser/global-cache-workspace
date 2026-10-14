# Cache-processor benchmarks

`scripts.benchmarks.harness` measures maintained BT/BGT cache-processor operations and a complete optimizer step. The optimizer path loads a normal `TrainConfig`, initializes its configured checkpoint, and follows its trainable policy, AdamW schedule, gradient clipping, gradient accumulation, compile policy, and BF16 setting. It loads one real training batch through the configured stream and reuses it so data-loader and network time stay outside the measurement.

The harness honors `loss_backend`. Keep `reference` for baseline evidence; set `target_only` or `liger` in the benchmark's training config for an explicit backend comparison. Set `truncated_bptt_windows: 0` for full BPTT or a positive horizon for the optional truncated-BPTT path; the optimizer-step result records the selected training method. The output binds backend provenance.

```bash
uv run --extra gpu python -m scripts.benchmarks.harness \
  --train-config configs/train/smoke_bgt.yaml \
  --device cuda \
  --warmup 5 \
  --steps 30 \
  --output artifacts/benchmarks/smoke-bgt.json
```

Use `--suite processor` or `--suite train-step` to isolate one part. The default `all` suite records index selection, gather, processor transform, scatter, end-to-end processor forward, and the full training optimizer step.

The output is atomic integrity-enveloped JSON under the repository's existing SHA-256 convention. Its payload binds the resolved config and hash, authored config-file hash, source commit and source-tree hash, processor variant, device/Torch/CUDA identity, warmup count, every measured step, interpolated median/p10/p90, peak allocated CUDA memory, and actual non-padding and supervised token counts. `source_dirty` must be false before treating a run as canonical evidence. The embedded unkeyed digest detects corruption but is not evidence authentication.

Micro-operation timings are forward-only and intentionally exclude fixture construction. The optimizer-step timing includes zeroing gradients, all configured accumulation microsteps, forward, backward, clipping, schedule updates, and AdamW update. It excludes data loading, evaluation, checkpoint writing, distributed communication, and post-timing evidence hashing. After the final synchronized sample, the harness records exact trainable-parameter, post-clip gradient, and optimizer-tensor hashes so matched candidate runs can reject trajectory drift. `GCW_COMPILE` retains its normal effect; use warmup iterations to keep compilation outside measured steps. CPU reports zero peak memory because Torch exposes no equivalent CPU allocator peak.

Correctness parity runs in ordinary CPU pytest. GPU performance execution is opt-in through `GCW_RUN_GPU_PERF_TESTS=1` and carries both `gpu` and `performance` markers; no latency threshold runs in CPU CI.

## Triton cache movement

The initial handwritten-kernel candidates gather contiguous layer-major BF16 K/V directly into Longshot/Icarus per-cache-head, BT per-layer, or BGT global layout. They are lazy, default-off, and benchmark-only until complete H100/H200 trajectory gates pass. Forward and backward move raw 16-bit payloads without floating-point arithmetic. Select the microbenchmark boundary with `--layout per_head|layer|global`.

```bash
GCW_RUN_TRITON_CACHE_CUDA_TESTS=1 pytest -q tests/test_triton_cache_movement_cuda.py
python -m scripts.benchmarks.triton_cache_movement \
  --expected-accelerator h200 \
  --output artifacts/benchmarks/triton-cache-h200.json
```

The microbenchmark compares the complete eager and candidate gather/backward operation. Its timing does not establish optimizer-step or campaign admission. `triton_processor.py` reproduces the rejected cache-processor `silu(gate) * up` candidate and intentionally enforces the `5e-4` relative-L2 gate before timing; the observed forward drift was `0.00288389` on both GPU classes.

## Fused structured row loss

`scripts.benchmarks.fused_row_loss` reproduces the rejected production-shaped bounded-reference versus per-row Liger `row_sum` forward/backward boundary. It reports raw CUDA-event samples and peak allocator state. Run it only in an exact `liger-kernel==0.8.1` environment. The measured candidate was about 34–37x slower on H100/H200 and is not dispatched by model training.

```bash
python -m scripts.benchmarks.fused_row_loss --tokens 1024 --hidden 1024 --vocab 151936
```

## Exact pair-wave timing

`scripts.benchmarks.pair_wave_timing` reserves 32 distinct fixed two-row pairs, checks bitwise pair-loss and raw-gradient equality, then compares serial and width-2/4 concurrent-forward accumulation with ordinary source-ordered backward and one gradient clip. It rejects adaptive grouping and Diffusion RNG. The executor remains benchmark-only; any autograd stream-mismatch warning, OOM, parity failure, or speedup below `1.15x` rejects retention.

## Formal CUDA parity

`scripts.benchmarks.adaptive_cuda_parity` compares fixed batch-2, adaptive eager, and adaptive compiled execution from one model state and one pre-execution reservation of 32-pair optimizer steps. It is a correctness harness, not a timer.

```bash
uv run --extra gpu python -m scripts.benchmarks.adaptive_cuda_parity \
  --train-config /path/to/adaptive-bt.yaml \
  --device cuda \
  --steps 3 \
  --output artifacts/benchmarks/adaptive-bt-parity.json
```

Adaptive mode requires `performance.adaptive_batching`; fixed-regional mode does not. The CLI supports deterministic Qwen3 BT/BGT regional compiled backends. Before model allocation it rejects dirty source, resume configs, unavailable requested resume evidence, trajectories beyond `max_steps`, distributed execution, unsupported families, processor or LoRA dropout, and noisy Diffusion.

Use `--variant-set fixed_regional` to compare fixed batch-2 eager against fixed batch-2 regional `compile` and `compile_full_windows` execution. Use `--variant-set triton_cache` for fixed eager versus the benchmark-only Triton gather/writeback candidate. Both modes preserve each pair's physical batch shape and padded width; neither treats adaptive grouping as an oracle. The Triton variant records exact semantic selection/write events plus pair losses, raw gradients, Adam moments, parameter updates, and RNG under the standard tolerances.

Use `--variant-set checkpoint_threshold --candidate-checkpoint-min-padded-sequence-length N` to compare two fixed eager segmented-checkpoint thresholds. Each variant declares its own expected replay boundaries while retaining the same reserved pairs, source order, model state, optimizer, and semantic event gate. This remains reservation-only evidence: the harness explicitly does not claim production checkpoint/resume parity.

The integrity-enveloped JSON reports:

- one-time reservation identity, explicitly not per-variant loader-resume evidence;
- stable source pair/row and actual cache-boundary/window identities;
- semantic selected indices and write destinations after classifying and trimming grouped padding-only events;
- expected and actual checkpoint replay counts aligned by stable boundary identity, including reverse-order replay;
- all 32 pair losses in source order;
- global and per-tensor relative-L2/cosine gradient and Adam-moment metrics, with strict near-zero and structural nonfinite gates;
- parameter trajectories, optimizer step counters, and CPU/CUDA RNG digests;
- exact BT/BGT operation coverage and an overall fail-closed verdict.

The harness observes cache indices and write destinations only; it does **not** claim cache-value parity. Production checkpoint save/reload is not implemented here, so `--require-resume-evidence` deliberately fails rather than presenting reservation cursors as resume evidence.

Instrumentation is restoring and limited to standard BT/BGT selection and writeback boundaries. Every event copies indices from the device to CPU; that synchronization, Python hooks, and report construction make instrumented execution invalid for timing. Run performance measurements separately with `scripts.benchmarks.harness`.

Qwen3 compile acceptance uses `CompileAcceptanceRecorder` rather than hiding compilation in warmup. Adaptive 16/8/4/2 execution should produce four initial-window graphs and four later-window graphs: one per physical batch shape in each region. Each region therefore records three batch-shape recompiles; past-cache length remains dynamic within each later-window batch specialization. The recorder retains synchronized first-call cost per observed signature; Dynamo acceptance counters supply graph and recompile counts. Report cold costs separately from steady-state timings. The recorder attaches only to the tested backbone instance, so a runner cannot reuse bindings from another model.

## Gemma4 native cache storage

`scripts.benchmarks.gemma4_cache_storage` is a small synthetic diagnostic for native list/static storage, selected-history gather, score tiling, frontier writeback, and a processor-plus-one-token decode. It creates four tiny Gemma4 layers in BF16 at batch size two, with a 128-token sliding window, and measures both all-layer and full-attention-only selection at increasing history lengths. It loads no checkpoint or dataset. Score and gather microbenchmarks isolate one native layer shape; their reference packs or merges the full history. Both decode paths use the current native processor, so decode timing isolates storage behavior rather than comparing against an older implementation. Static decode timing includes restoring the synthetic frontier between samples.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python -m scripts.benchmarks.gemma4_cache_storage --device cpu --output artifacts/benchmarks/gemma4-cache-cpu.json
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=src python -m scripts.benchmarks.gemma4_cache_storage --device cuda --output artifacts/benchmarks/gemma4-cache-cuda.json
GCW_RUN_GEMMA_CACHE_CUDA_TESTS=1 pytest -q tests/test_gemma4_cache_storage_cuda.py
```

The JSON records synchronized wall-time samples, median latency, retained KV bytes, and extra peak allocated CUDA memory above each operation's starting allocation. Extra peak excludes already allocated fixture caches and model weights; it is not whole-process peak memory. CPU allocator peaks are not measured. These diagnostics have no latency gate and are not canonical campaign evidence.

A development run on 2026-10-14 used Torch 2.10.0+cu128, one idle B300, two warmups, five samples, and a 512 MiB PyTorch allocator limit. All six CUDA checks passed, covering BF16 weights, CUDA BF16 autocast, mixed shapes, layer subsets, generation parity, and checkpointed training gradients. CPU tests additionally cover ring wraps, chunks larger than the window, shared KV consumers, stable backing pointers, unread-capacity sentinels, retained old sliding history, and packed/list gradient parity. Full-attention-only synthetic results were:

| History | Dynamic retained KV (MiB) | Static retained KV (MiB) | Dynamic decode extra peak (MiB) | Static decode extra peak (MiB) | Dynamic/static GPU decode (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 2.000 | 1.126 | 3.022 | 1.266 | 3.447 / 3.618 |
| 4,096 | 8.000 | 4.126 | 12.046 | 5.062 | 3.527 / 3.621 |
| 16,384 | 32.000 | 16.126 | 48.141 | 20.250 | 3.460 / 3.608 |

At 16,384 tokens, GPU gather extra peak fell from 8.016 MiB to 0.023 MiB, static append used 0.062 MiB of temporary storage versus 32.004 MiB for dynamic append, and static frontier writeback added no allocator-visible payload versus 16 MiB for functional writeback. On CPU, median static writeback stayed near 0.029 ms across these lengths, and the 16,384-token processor/decode measurement was 26.556 ms versus 34.967 ms for dynamic storage. The tiny GPU decode was slightly slower with static storage despite the memory reduction; these results do not establish a production throughput improvement. The [cache-layout documentation](../../docs/concepts/cache-processing.md#cache-layout-and-layer-selection) gives separate 31B storage estimates. All-layer selection still retains every processor-visible archive, and KV-norm selection still scans those archives.
