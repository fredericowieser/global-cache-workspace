# Task-directed cache diffusion

**Status:** Experimental

Diffusion-BGT and Flow-BGT are separate mechanisms.

- **Flow-BGT** deterministically integrates a time-conditioned latent vector field.
- **Diffusion-BGT** injects independent noise into a bottleneck state during training, repeatedly refines that state with shared weights, and learns the endpoint through future-token cross-entropy.

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
