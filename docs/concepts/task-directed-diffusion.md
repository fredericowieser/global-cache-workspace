# Task-directed cache diffusion

**Status:** Experimental

Diffusion-BGT and Flow-BGT are separate mechanisms.

- **Flow-BGT** deterministically integrates a time-conditioned latent vector field.
- **Diffusion-BGT** injects independent noise into a bottleneck state during training, repeatedly refines that state with shared weights, and learns the endpoint through future-token cross-entropy.

Inference is explicit: `deterministic` omits fresh noise for a stable endpoint; `stochastic` samples fresh noise and requires recorded seeds for scientific comparison. Training remains noisy whenever `noise_scale > 0` and `perturbation_enabled` is true.

## Mechanism

Let $X$ be a prefix, $K$ its ordinary KV cache, and $Z=B(K)$ the BGT bottleneck. Diffusion-BGT computes

$$
S_T=Z+\sigma\epsilon,
\qquad
S_{i-1}=S_i+\Delta t_i\,v_\theta(S_i,Z,t_i),
$$

then projects $S_0$ into a gated cache update. The frozen backbone consumes the rewritten cache. Gradients pass through the backbone and all refinement steps into the processor.

The endpoint is not the original cache. No cache reconstruction, noise-prediction, or cache-space MSE objective defines the primary model:

$$
\mathcal L_{
text{primary}}=
-\sum_{h=1}^{H}\log p_F(x_{n+h}\mid x_{n+1:n+h-1},\widetilde K).
$$

The processor may construct any cache representation that supports future prediction.

## Scheduled training and independent solver depth

`training_method: fixed` is the default and retains the original learned-gate
recurrence, state-dict keys, initialization/RNG order, and serialized identity
when the new controls are unset. `training_method: scheduled` removes the
refinement gate and, with the default `update_gate_type: static_scalar`, the
write scalar gate. This legacy scheduled write remains fixed-one even when
`output_scale_mode` is left at its `learned` default. Every scheduled treatment
requires `cache_processor.out_proj_init: zero` for an initial no-op.

Longshot can opt into a contextual scalar or vector **writeback** gate with
`update_gate_type: contextual_scalar` or `contextual_vector` and
`output_scale_mode: learned`. The existing fused output projections predict
update and gate channels from the final solver output, once per intervention,
not once per field call. Neither the Euler solver nor its fixed step scale gains
a gate. For example, `gate_mode: sigmoid` and `gate_init: -2.0` initialize these
write gates to approximately 0.1192. KV and linear-state readouts own their gates
separately; see [Longshot update gating](../development/longshot-update-gating.md).

Explicit `output_scale_mode: fixed_one` remains incompatible with contextual
write gates and is also available without scheduled diffusion for BGT and
Longshot, with the same zero-readout requirement.

```yaml
diffusion_processor:
  training_method: scheduled
  schedule_horizon: 50
  training_solver_steps: 4
  inference_solver_steps: 8  # defaults to training_solver_steps
  noise_scale: 0.15
  time_conditioning: true
  perturbation_enabled: true
cache_processor:
  out_proj_init: zero
```

Training samples one level `s` uniformly from `1..H`, sets `t=s/H` and
`sigma=noise_scale*t`, and perturbs the initial latent once. Exactly `K`
shared-field Euler calls integrate `t -> 0`, with update scale
`eta_max*t/K` and log-sigma conditioning at the positive right endpoints.
The decoded displacement excludes the initial perturbation. Deterministic
inference draws no level or noise, starts at `t=1`, and uses the inference K.
`num_steps` is not solver depth when explicit horizon/depth controls are set.
For older scheduled configs with all three controls unset, a sampled level `s`
still performs `s` calls, and evaluation performs `num_steps` calls.

`perturbation_enabled: false` suppresses the perturbation but preserves training
RNG consumption and the sampled schedule. Disabling `time_conditioning` requires
perturbation to be disabled in scheduled mode; positive `noise_scale` remains
required to define log-sigma. These are controls, not diffusion treatments.

Longshot retains the upstream initial head identities and routing gates across
all field calls, including soft contextual routing. It does not reselect heads
or multiply solver depth by ordinary internal iterations; internal iterations
above one are rejected for Diffusion-Longshot. External recurrence and generation
cadence remain separate. Scheduled recurrence requires the eager backend.

For `gcw eval all|math|general|tasks|chat`,
`--diffusion-eval-solver-steps K` overrides only runtime evaluation depth on a
scheduled, explicitly decoupled checkpoint. It rejects HF targets, processor
bypass, fixed/coupled diffusion, and invalid K, and never edits checkpoint
config/weights. Results record the override and actual scheduled field calls.
These counters are exact for inference without checkpoint replay; they do not
measure checkpointed backward computation. Replay may repeat a field or stop
inside it before the counter increments.
Training FLOPs use configured K (or mean sampled depth for coupled schedules);
telemetry reports sampled level/time/sigma and the runtime write-gate mean
(one for legacy scheduled writes). Contextual linear-state gates have a separate
mean. Gate projections and activations are counted once per invocation, outside
the K field evaluations.

## Information and computation

If every refinement step receives only $Z$, its current state, time, and noise independent of the context, then

$$
X\rightarrow K\rightarrow Z\rightarrow S_T\rightarrow\cdots\rightarrow S_0.
$$

The data processing inequality gives

$$
I(X;S_0)\le I(X;Z)\le I(X;K).
$$

Dimensional expansion does not restore discarded context information. The bottleneck controls information; diffusion allocates computation over what remains.

This bound does **not** apply if clean raw KV enters the denoiser after $Z$, if an original-cache residual bypasses the bottleneck, or if raw-cache-dependent routing carries an unaccounted side channel. Residual BGT writeback also leaves untouched cache state available to the backbone. Claims must name the bounded path rather than treating the complete transformer state as bottlenecked.

## Difference from conventional diffusion

Conventional diffusion learns a natural-data endpoint through local denoising targets. Task-directed cache diffusion receives an indirect functional gradient through a frozen language model. Its endpoint is an equivalence class of predictively useful caches, not one observed cache.

Expected consequences:

| Property | Expected effect |
|---|---|
| Training cost | Higher: several processor evaluations plus frozen-backbone differentiation. |
| Optimisation | Harder credit assignment and greater noise-seed variance. |
| Natural-cache fidelity | Lower by design; fidelity is not the target. |
| Task performance | Can improve because irrelevant cache detail need not be reconstructed. |
| Generalisation | Can improve under a useful bottleneck, but activation shortcuts are a central risk. |
| Stochasticity | May collapse, leaving a deterministic recurrent processor. |
| Extra-step reasoning | Not automatic; later steps must show measured improvement. |

Long-horizon teacher-forced loss, free-running generation, unseen context lengths, repeated processing, and seed sensitivity matter more than one-token perplexity.

## Required controls

1. One step, zero noise: one-shot control.
2. Several steps, zero noise: deterministic iteration control.
3. Several steps, non-zero noise: Diffusion-BGT treatment.
4. Processor bypass: exact loaded-backbone control.
5. Multiple noise seeds: distinguish useful stochasticity from uncontrolled variance.
6. Intermediate readouts: test whether later steps improve future-token loss.
7. No-bypass intervention: hold $Z$ fixed while perturbing unavailable raw KV; the generated update must remain fixed.

A diffusion claim requires non-zero training noise, time/noise conditioning, shared iterative refinement, and an authenticated stochastic treatment. Repeated deterministic computation alone is Flow-BGT or recurrence, not diffusion.

## Per-example update diagnostics

Native evaluation can record how each Longshot diffusion step changes the latent
and the proposed cache update. Enable this explicitly:

```bash
gcw eval paper7 --checkpoint /path/to/checkpoint --out results/paper7.json \
  --cache-update-metrics
```

This writes `results/paper7.cache-update-metrics.jsonl`. Pass
`--cache-update-metrics /path/to/updates.jsonl` to choose another destination;
an explicit path is required when evaluation has no results destination.
The Python equivalent is `EvalConfig(cache_update_metrics_path="updates.jsonl")`.
Answer NLL also accepts `cache_update_metrics_path` directly, preserving its
automatic inference of chat settings from the checkpoint.

The option supports fixed and scheduled Longshot diffusion in Paper7,
Paper7-legacy, other native lm-eval tasks, chat/GSM8K benchmarks, answer NLL, and
FineWeb. It requires eager diffusion recurrence. HF-only targets, processor
bypass, and other model variants reject it.
It adds no measurements to training. With the option disabled, no extra
readouts or tensor reductions run.

Each JSONL record identifies one task and example and contains a `steps` list.
Each entry identifies `external_iteration` and `solver_step` (both start at 1)
and provides the following groups under `metrics`:

Let $p_s$ be the cumulative ungated projected proposal at solver step $s$,
with $p_0 = 0$, and let $d_s = p_s - p_{s-1}$ be that step's contribution.
Keys and values are measured separately.

| Group | Measurement for one head |
| --- | --- |
| `latent_kv` | L2 norm of that solver step's actual change to the KV latent vector |
| `latent_linear` | The same for a linear-state latent vector |
| `key`, `value` | L2 norm of $d_s$ for attention keys or values |
| `linear_key`, `linear_value` | L2 norm of $d_s$ for projected linear keys or values, before normalization and writer transforms |
| `linear_state` | Frobenius norm of the change in the proposed linear-state matrix update since the preceding solver step |
| `key_direction_change`, `value_direction_change` | $1 - \cos(d_s, d_{s-1})$, starting at step 2 |
| `linear_key_direction_change`, `linear_value_direction_change` | The same for projected linear keys or values |
| `key_direction_to_original`, `value_direction_to_original` | $1 - \cos(d_s, \mathrm{original})$ |
| `key_cache_direction_from_original`, `value_cache_direction_from_original` | $1 - \cos(\mathrm{original} + p_s, \mathrm{original})$ |

Cosine distance is unitless: 0 means aligned, 1 orthogonal, and 2 opposite.
Distances use FP32 arithmetic with similarity clamped to $[-1, 1]$.
Comparisons with either vector norm at most $10^{-12}$ are undefined and
excluded. Original attention vectors come from the already gathered frontier,
in the proposal's coordinate system; when `unrope_keys` is enabled, original
keys are unrotated too. Each processor application, including each external
iteration, starts a new contribution history and uses its own incoming frontier.

The solver's step-size multiplier is included. Final writeback gates are excluded.
For `linear_state`, key normalization and the linear writer's own dynamics
are included. Linear key/value magnitudes and directions use only raw projected
keys and values, excluding gated-delta beta logits and decay inputs. In
`gated_delta` mode, configuration disables the outer linear writeback gate;
the writer's internal sigmoid and decay dynamics remain included in `linear_state`.
Value-only writeback emits only the attention `value` measurements, alongside
latent and any linear measurements. All destination heads across the relevant
layers are included; attention selection does not restrict this population.
Absent linear states produce no linear metric groups.

For cache contributions, the first step is compared against a zero update;
subsequent steps are compared against the preceding decoded proposal. Any
projection bias therefore belongs to the first step. For the linear writer,
each proposal is decoded against the same incoming recurrent state. Intermediate
proposals are only measured; the real cache is still written once at the end.
The measured contributions add up to the final ungated proposal. Step direction
change compares consecutive contributions, not consecutive cumulative proposals.

Every processor application produces a min, max, mean, and population standard
deviation across per-head measurements. Each example's record then averages
**each statistic separately** across contributing applications. For example, if
two applications have head magnitudes `[1, 3]` and `[5, 9]`, the reported values
are `min=3`, `max=6`, `mean=4.5`, and `std=1.5`. The maximum is the average of
the two maxima, not the largest magnitude encountered. Single-head standard
deviation is zero.

For directions, each application's statistics use only defined comparisons.
Applications with no defined comparisons do not contribute; entirely undefined
groups are omitted. `heads` remains the full head population, while `applications`
counts contributing applications. Direction groups additionally record
`valid_head_observations`, the total number of defined head comparisons across
those applications and all requests for the example. Thus applications retain
equal weight even when their counts of defined heads differ. Magnitude
aggregation is unchanged and includes zero magnitudes.

Every metric is reported for prefill, decoding, and their combination. Prefill
includes processing supplied tokens and any update on the completed prompt
cache before generated tokens enter it. In generation, the update at
`generated_idx == 0` therefore belongs to prefill even though it runs inside
the decoding loop. Subsequent updates belong to decoding. Teacher-forced
option likelihood, Answer NLL, and bits-per-byte evaluations contribute only
to prefill, including their supplied answer tokens.

Phase-specific sums and application counts are accumulated separately; combined
statistics add those sums and counts before dividing. They do not average the
two phase means equally. The same direction-validity exclusions apply within
each phase. Measurements and head reductions run only once per application;
only the small accumulators are separated, without retaining token histories.

Finished generation rows and padded input tails do not contribute.
Multiple-choice examples combine applications from all their answer-option
requests, weighted
by application count, and record the number of requests. A FineWeb example is
one evaluated token-sequence row. An example with no processor applications has
an empty `steps` list. Batch retries discard unsuccessful observations;
distributed harness padding requests are deduplicated.

The sidecar schema is `gcw.cache_update_metrics.v3`. Each example retains
`steps` as its combined view and adds `phase_steps.prefill` and
`phase_steps.decoding`, each using the same step and metric format. Both phase
lists are always present; a phase with no contributing applications has an
empty list. For example, a teacher-forced record has this structure:

```json
{
  "schema": "gcw.cache_update_metrics.v3",
  "task": "example",
  "example_id": "0",
  "request_count": 1,
  "steps": [{"external_iteration": 1, "solver_step": 1, "metrics": {
    "value": {"min": 1, "max": 3, "mean": 2, "std": 1,
              "applications": 1, "heads": 2}
  }}],
  "phase_steps": {
    "prefill": [{"external_iteration": 1, "solver_step": 1, "metrics": {
      "value": {"min": 1, "max": 3, "mean": 2, "std": 1,
                "applications": 1, "heads": 2}
    }}],
    "decoding": []
  }
}
```

Evaluation results contain `cache_update_metrics` metadata (under each task's
`gcw_runtime` for lm-eval suites) with the path, aggregation definition,
completion flag, example count, and per-task content hashes. Resume checks
these hashes before reusing completed tasks and rejects v1/v2 sidecars or
metadata because their phase breakdown cannot be recovered. Use a fresh
evaluation destination to collect v3 measurements; evaluations without
diagnostics also cannot resume into one that requires them. Historical v1/v2
sidecars remain readable by the plotting script for combined plots; v1 retains
its recorded combined key/value groups without inventing separate measurements
or directions. Phase-specific plots require v3.

Answer NLL publishes the sidecar before its exclusive result receipt. If the
receipt write fails before publication, the newly created sidecar is removed
so evaluation can be retried. If the receipt was published but a subsequent
filesystem synchronization fails, both files are retained. An existing receipt
is rejected before publishing a sidecar. If another writer publishes a receipt
concurrently, the new sidecar is removed only when that receipt can be read and
does not reference it; an unreadable receipt leaves the sidecar intact.
A hard process interruption cannot run cleanup: after confirming that the
evaluation process has stopped and its receipt is absent, remove that run's orphan sidecar before
retrying. An existing receipt and its referenced sidecar must be kept together.

Diagnostics add one KV/linear readout pair for each intermediate solver step and
reuse the existing final readout. Four steps therefore add three readout pairs.
Direction diagnostics reuse those projections, retaining the previous contribution
alongside the previous proposal. Original attention KVs use a detached view of
the gathered frontier; no full-cache copies are made. Additional vector storage
is independent of sequence length.
The outer-product writer's matrix-difference norms are calculated from factors
without allocating full matrices. The gated-delta writer additionally evaluates
the proposed state transitions, one layer at a time. Summaries accumulate on the
device and transfer at batch boundaries. Wall time and memory overhead depend
on the writer, head dimensions, and solver depth; these metrics describe update
magnitude and direction, not whether the updates improve predictions.

## References

- [Diffusion-QL](https://arxiv.org/abs/2208.06193): diffusion outputs shifted by downstream utility while retaining a behavioural prior.
- [DRaFT](https://arxiv.org/abs/2309.17400): direct reward backpropagation through a diffusion sampling trajectory.
- [AlignProp](https://arxiv.org/abs/2310.03739): end-to-end differentiable reward alignment through denoising.
- [IRED](https://arxiv.org/abs/2406.11179): diffusion-like annealed computation as iterative reasoning.
- [Palette](https://arxiv.org/abs/2111.05826): conditional inputs and generated outputs need not be the same object.

None is an exact precedent for future-token-trained KV-cache generation through an explicit information bottleneck.
