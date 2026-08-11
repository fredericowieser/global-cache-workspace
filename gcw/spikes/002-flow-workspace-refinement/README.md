# Spike 002: flow workspace refinement

## Verdict

**Feasible.** A single tied conditional field can refine Longshot-shaped persistent workspace slots, preserve dimensions and end-to-end gradients, remain coordinate-wise bounded for 1/2/4/8 solver steps, and learn a tiny source-conditioned target with conditional flow matching. On the same endpoint-MSE toy objective, the residual flow reached `1.0945e-05` MSE versus `5.6695e-04` for tied direct-state recurrence (about 52x lower; ratio `0.0193`) with the same 3,970 parameters, initialization, optimizer updates, and four field evaluations per forward.

This warrants a model-integrated experiment. It does **not** establish downstream language-model quality or production stability.

## What was inspected

Production files were read but not modified.

- `gcw/models/flow.py`: `LatentFlow` already integrates a tied field with Euler steps and returns a displacement. Time enters as an additive sinusoidal embedding; a sigmoid gate bounds the total integration scale.
- `gcw/models/longshot.py`: `LongshotWorkspaceSource.merged` is shaped like `[batch, cache_heads, selected_time, input_dim]`. `initialise_workspace` creates persistent slots shaped `[batch, cache_heads, output_dim]`. `deliberate` replaces the latest source frontier value with the current workspace before running the cache processor.
- The existing Flow-Longshot boundary is inside `CacheProcessor.compute_update`: it integrates the encoded processor hidden state, while source/head selection is captured by a closure. This spike tests a different boundary: directly integrate the **persistent workspace** as `dW/dt = v_theta(W,t,source)` between initialization and readout.

## Prototype

`refinement.py` is standalone PyTorch code and imports no GCW production module.

### Contract

```text
workspace W: [B, S, D_workspace]
source:      [B, S, T_selected, D_source]
velocity:    [B, S, D_workspace]
```

`S` maps naturally to Longshot cache-head workspace slots. The field pools each slot's selected source history and a global cross-slot source summary. It concatenates those conditions with normalized workspace state and continuous-time features, then predicts

```text
v_theta(W, t, source) = v_max * tanh(MLP(...))
```

The same field object and parameters are reused at every step. Integration supports:

```text
Euler:    W[k+1] = W[k] + h * v(W[k], t[k], source)
Midpoint: W[mid] = W[k] + h/2 * v(W[k], t[k], source)
          W[k+1] = W[k] + h * v(W[mid], t[k] + h/2, source)
```

Because `h = 1 / steps` and every velocity coordinate is in `[-v_max, v_max]`, total coordinate displacement over `[0,1]` is bounded by `v_max` for either solver.

### Conditional flow-matching objective

The synthetic target is a bounded source-conditioned residual over the initial workspace. Training samples a straight bridge

```text
W_t = W_0 + t * (W_target - W_0)
target velocity = W_target - W_0
loss = MSE(v_theta(W_t, t, source), target velocity)
```

The direct-recurrence comparison is separate and fair: both direct recurrence and residual Euler flow train against the same full-batch terminal reconstruction MSE for 500 Adam updates. They use deep-copied identical initial fields, 3,970 parameters, and four field evaluations per forward. The direct baseline replaces state, `W[k+1] = F_theta(W[k],t,source)`; the flow keeps the identity path and integrates scaled residuals.

## Exact commands

Run from the repository root:

```bash
cd /home/fred/flow/code/global-cache-workspace

uv run --group dev python spikes/002-flow-workspace-refinement/run_experiment.py \
  --output spikes/002-flow-workspace-refinement/observed_results.json

uv run --group dev pytest -q \
  spikes/002-flow-workspace-refinement/test_refinement.py

uv run --group dev ruff check \
  spikes/002-flow-workspace-refinement
```

Environment observed: CPU, PyTorch `2.13.0+cpu`. The runner enables deterministic algorithms and fixes all random seeds.

## Observed output

Full machine-readable output is in `observed_results.json`.

### Checks

```text
all_pass: true
bounded_updates: true
conditional_flow_matching: true
dimensions_and_gradients: true
same_objective_comparison: true
tied_parameters: true
zero_field_identity: true
```

### Dimensions and gradient flow

```text
workspace: [3, 4, 6]
source:    [3, 4, 5, 8]
output:    [3, 4, 6]
workspace gradient norm: 9.657995223999023
source gradient norm:    0.1937549114227295
parameter tensors with finite nonzero gradients: 10 / 10
```

### Zero-field identity

Exact tensor equality passed for Euler and midpoint at 1, 2, 4, and 8 steps.

### Tied parameters

```text
steps:            1     2     4     8
parameter count: 3970  3970  3970  3970
same field object reused: true
```

### Bounded updates and step stability

Midpoint results use a 64-step midpoint endpoint as the numerical reference. The hard coordinate bound is `0.75`.

| Steps | Max absolute update | RMSE to 64-step endpoint | Within bound |
|---:|---:|---:|:---:|
| 1 | 0.27392623 | 4.94310e-04 | yes |
| 2 | 0.27358070 | 1.22252e-04 | yes |
| 4 | 0.27371517 | 2.85154e-05 | yes |
| 8 | 0.27375007 | 6.23169e-06 | yes |

The endpoint converged monotonically toward the 64-step reference in this run while the update magnitude stayed effectively constant.

### Tiny conditional flow-matching fit

```text
velocity MSE before training: 1.475026e-02
velocity MSE after 600 updates: 5.441817e-06
8-step midpoint endpoint MSE:  4.847750e-06
```

### Direct recurrence versus residual flow

Same full-batch endpoint MSE, initial parameter values, parameter count, optimizer (`Adam`, lr `0.008`), gradient clipping (`1.0`), 500 updates, and four field evaluations per forward:

| Model | Initial MSE | Final MSE |
|---|---:|---:|
| Direct tied state recurrence | 1.799393e-02 | 5.669547e-04 |
| Tied residual Euler flow | 1.427142e-02 | 1.094499e-05 |

The initial endpoint losses differ because replacement and residual integration apply identical field parameters with different state semantics. The residual target favors an identity path by construction; that is the intended workspace-refinement regime, but it also means this toy is not a neutral benchmark for arbitrary state replacement.

### Test and lint output

```text
..............                                                           [100%]
14 passed in 8.33s
```

```text
All checks passed!
```

## Recommendation for a model-integrated follow-up

Insert a `WorkspaceFlowRefiner` around Longshot's persistent `workspace` in the process-branch path: initialize once, condition every field call on the frozen `LongshotWorkspaceSource`, integrate for configurable Euler/midpoint steps, then pass the refined workspace to `write_for_readout`. Keep one tied processor field, a hard or gated velocity bound, and a zero-field/zero-projection identity initialization.

Measure one, two, four, and eight steps at fixed trained weights. Report task loss, gradient norms, update norms, solver disagreement, latency, and memory. Compare against existing external `deliberate` recurrence at matched field evaluations. Do not replace the current hidden-state `LatentFlow` until that test separates the two possible flow boundaries.

## Limitations

- Synthetic fixed-batch regression is only a feasibility check.
- Source conditioning uses mean pooling, not Longshot's selected-head attention.
- The bound is coordinate-wise, not an L2 or RMS norm bound.
- Midpoint doubles field evaluations per nominal step; matched-compute comparisons must account for this.
- No mixed-precision, distributed, checkpoint, or real-cache integration was exercised.
