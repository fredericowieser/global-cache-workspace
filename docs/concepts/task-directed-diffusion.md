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
when the new controls are unset. `training_method: scheduled` removes both the
refinement and write scalar gates; it requires `cache_processor.out_proj_init:
zero` for an initial no-op. Scheduled writes remain fixed-one even when
`output_scale_mode` is left at its `learned` default. Explicit `output_scale_mode:
fixed_one` is also available without scheduled diffusion for BGT and Longshot,
with the same zero-readout requirement.

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
telemetry reports sampled level/time/sigma and the fixed-one write scale.

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

## References

- [Diffusion-QL](https://arxiv.org/abs/2208.06193): diffusion outputs shifted by downstream utility while retaining a behavioural prior.
- [DRaFT](https://arxiv.org/abs/2309.17400): direct reward backpropagation through a diffusion sampling trajectory.
- [AlignProp](https://arxiv.org/abs/2310.03739): end-to-end differentiable reward alignment through denoising.
- [IRED](https://arxiv.org/abs/2406.11179): diffusion-like annealed computation as iterative reasoning.
- [Palette](https://arxiv.org/abs/2111.05826): conditional inputs and generated outputs need not be the same object.

None is an exact precedent for future-token-trained KV-cache generation through an explicit information bottleneck.
