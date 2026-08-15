# Cache-processor benchmarks

`scripts.benchmarks.harness` measures maintained BT/BGT cache-processor operations and a complete optimizer step. The optimizer path loads a normal `TrainConfig`, initializes its configured checkpoint, and follows its trainable policy, AdamW schedule, gradient clipping, gradient accumulation, compile policy, and BF16 setting. It loads one real training batch through the configured stream and reuses it so data-loader and network time stay outside the measurement.

The harness honors `loss_backend`. Keep `reference` for baseline evidence; set `target_only` or `liger` in the benchmark's training config for an explicit backend comparison. The output binds backend provenance.

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

Micro-operation timings are forward-only and intentionally exclude fixture construction. The optimizer-step timing includes zeroing gradients, all configured accumulation microsteps, forward, backward, clipping, schedule updates, and AdamW update. It excludes data loading, evaluation, checkpoint writing, and distributed communication. `GCW_COMPILE` retains its normal effect; use warmup iterations to keep compilation outside measured steps. CPU reports zero peak memory because Torch exposes no equivalent CPU allocator peak.

Correctness parity runs in ordinary CPU pytest. GPU performance execution is opt-in through `GCW_RUN_GPU_PERF_TESTS=1` and carries both `gpu` and `performance` markers; no latency threshold runs in CPU CI.

## Triton cache movement

The initial handwritten-kernel candidates gather contiguous layer-major BF16 K/V directly into Longshot/Icarus per-cache-head, BT per-layer, or BGT global layout. They are lazy, default-off, and benchmark-only until complete H100/H200 trajectory gates pass. Forward and backward move raw 16-bit payloads without floating-point arithmetic. Select the microbenchmark boundary with `--layout per_head|layer|global`.

```bash
GCW_RUN_TRITON_CACHE_CUDA_TESTS=1 pytest -q tests/test_triton_cache_movement_cuda.py
python -m scripts.benchmarks.triton_cache_movement \
  --expected-accelerator h200 \
  --output artifacts/benchmarks/triton-cache-h200.json
```

The microbenchmark compares the complete eager and candidate gather/backward operation. Its timing does not establish optimizer-step or campaign admission. `triton_processor.py` separately compares the production-shaped cache-processor `silu(gate) * up` forward/backward boundary and enforces the `5e-4` relative-L2 numerical gate before reporting speed.

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

Use `--variant-set fixed_regional` to compare fixed batch-2 eager against fixed batch-2 regional `compile` and `compile_full_windows` execution. This mode preserves each pair's physical batch shape and padded width; it does not treat adaptive grouping as an oracle.

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
