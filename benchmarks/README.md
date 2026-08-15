# Cache-processor benchmarks

`benchmarks.harness` measures maintained BT/BGT cache-processor operations and a complete optimizer step. The optimizer path loads a normal `TrainConfig`, initializes its configured checkpoint, and follows its trainable policy, AdamW schedule, gradient clipping, gradient accumulation, compile policy, and BF16 setting. It loads one real training batch through the configured stream and reuses it so data-loader and network time stay outside the measurement.

The harness honors `loss_backend`. Keep `reference` for baseline evidence; set `target_only` or `liger` in the benchmark's training config for an explicit backend comparison. The output binds backend provenance.

```bash
uv run --extra gpu python -m benchmarks.harness \
  --train-config configs/train/smoke_bgt.yaml \
  --device cuda \
  --warmup 5 \
  --steps 30 \
  --output artifacts/benchmarks/smoke-bgt.json
```

Use `--suite processor` or `--suite train-step` to isolate one part. The default `all` suite records index selection, gather, processor transform, scatter, end-to-end processor forward, and the full training optimizer step.

The output is atomic authenticated JSON under the repository's existing SHA-256 convention. Its payload binds the resolved config and hash, authored config-file hash, source commit and source-tree hash, processor variant, device/Torch/CUDA identity, warmup count, every measured step, interpolated median/p10/p90, peak allocated CUDA memory, and actual non-padding and supervised token counts. `source_dirty` must be false before treating a run as canonical evidence.

Micro-operation timings are forward-only and intentionally exclude fixture construction. The optimizer-step timing includes zeroing gradients, all configured accumulation microsteps, forward, backward, clipping, schedule updates, and AdamW update. It excludes data loading, evaluation, checkpoint writing, and distributed communication. `GCW_COMPILE` retains its normal effect; use warmup iterations to keep compilation outside measured steps. CPU reports zero peak memory because Torch exposes no equivalent CPU allocator peak.

Correctness parity runs in ordinary CPU pytest. GPU performance execution is opt-in through `GCW_RUN_GPU_PERF_TESTS=1` and carries both `gpu` and `performance` markers; no latency threshold runs in CPU CI.

## Formal adaptive CUDA parity

`benchmarks.adaptive_cuda_parity` compares fixed batch-2, adaptive eager, and adaptive compiled execution from one model state and one reserved sequence of 32-pair optimizer steps. It is a correctness harness, not a timer. Instrumented runs record no latency measurements.

```bash
uv run --extra gpu python -m benchmarks.adaptive_cuda_parity \
  --train-config /path/to/adaptive-bt.yaml \
  --device cuda \
  --steps 3 \
  --output artifacts/benchmarks/adaptive-bt-parity.json
```

The train config must enable `performance.adaptive_batching`. The executable compiled CLI currently supports deterministic Qwen3 BT/BGT because those are the maintained regional compiled backends. It rejects other families, distributed execution, processor or LoRA dropout, and noisy Diffusion before model allocation. It loads all variants from one state, restores one RNG snapshot, and emits authenticated JSON containing:

- stable step, pair, row, selection, and write-destination keys;
- primary cache events and checkpoint-recomputation linkage;
- all 32 pair losses in source order;
- named trainable gradients and parameter trajectory;
- named AdamW moments and optimizer step counters;
- loader cursor and CPU/CUDA RNG digests;
- operation-specific all-close statistics and an overall fail-closed verdict.

The restoring trace context covers standard BT/BGT selection and writeback plus maintained Icarus/Longshot latest-cache writes. New cache processor families must extend this instrumentation before their adaptive result can be admitted. Run performance timing separately with `benchmarks.harness`; instrumentation intentionally introduces Python hooks and device-to-host copies.
