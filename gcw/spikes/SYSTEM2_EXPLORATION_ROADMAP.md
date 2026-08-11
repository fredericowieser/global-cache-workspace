# Frozen-LLM global-workspace System-2 exploration

## Target

Build a sidecar for a frozen causal LLM whose answer quality improves with optional internal compute. The sidecar must support state carry, new latent sequence positions, persistent workspace memory, and eventually branch selection. Visible chain-of-thought is a teacher/control channel, not required at inference.

## State contract

| State | Shape | Role | Mutability |
|---|---|---|---|
| Source cache $S$ | frozen per-layer KV | Prompt evidence | Immutable |
| Workspace $W_k$ | $B\times M\times d_w$ | Persistent plan/evidence/hypotheses | Recurrently writable |
| Thought tape $Z_k$ | $B\times Q\times d_{model}$ | Temporary Coconut-like latent positions | Appended then discardable |
| Latent KV $C_k$ | frozen-backbone KV for $Z_k$ | Lets the decoder consume emitted thoughts as genuine causal positions | Append-only within a branch |
| Visible output $Y$ | normal vocabulary tokens | Answer/actions/checkpoints | Autoregressive |

One cycle is

$$
W_{k+1/2}=\Phi_\theta(W_k,S,k),\qquad
Z_k=E_\theta(W_{k+1/2},S),
$$

$$
C_k,H_k=\mathrm{FrozenLM}(Z_k\mid S,C_{<k}),\qquad
W_{k+1}=\mathrm{Commit}_\theta(W_{k+1/2},H_k,S).
$$

The terminal answer decoder consumes $(S,C_{1:K},W_K)$. Extra $K$, $Q$, solver steps or particles are separately reported compute budgets.

## Architecture families

### A. Existing-cache recurrence

Current Longshot baseline. Improve optimisation with direct/strong gates, bounded residual updates, per-head gates, depth-weighted schedules and no-carry controls.

### B. Continuous thought tape

The sidecar emits continuous embeddings that create new causal positions in the frozen decoder. Unlike current cache rewriting, each cycle can add representational bandwidth. Compare:

1. Coconut feedback: project the last frozen hidden state into the next latent embedding.
2. Workspace emitter: cross-attend $Q$ learned queries to $W_k$ and $S$.
3. Delta emitter: emit only a residual over a fixed learned latent token.
4. Multiscale tape: one global thought plus several local/evidence thoughts per cycle.

### C. Persistent workspace + temporary thoughts

Temporary thought positions perform local computation; a fixed-size workspace commits only durable information. Use a fixed schedule first:

`source -> q latent thoughts -> workspace commit -> repeat -> answer`.

This avoids unbounded cache growth while preserving a true recurrent state.

### D. Flow/ODE workspace refinement

Use a tied conditional velocity field

$$
\frac{dW(t)}{dt}=v_\theta(W(t),t,S)
$$

with Euler or midpoint integration. Randomise solver steps during training and evaluate 1/2/4/8 unseen steps. Two objectives:

1. end-to-end answer CE through the solver;
2. conditional flow matching toward a same-backbone visible-rationale endpoint.

Cap update RMS relative to workspace RMS. Compare against the same tied block applied as ordinary residual recurrence; solver depth alone is not evidence of flow benefits.

### E. Diffusion/denoising deliberation

Corrupt a teacher or successful workspace endpoint $W^*$ at noise level $t$ and train a denoiser/velocity field conditioned on $S$. At inference, refine one or more workspace particles from noise or from a source-conditioned initialiser. Guide trajectories with a frozen-answer energy or small verifier. This creates a compute/quality axis through denoising steps and particle count.

### F. KaVa-style same-backbone distillation

Use public GSM8K rationale text aligned exactly to the answer-only rows. The teacher path is the same frozen backbone consuming visible rationale. The student sees only the prompt and latent mechanism. Prefer:

$$
L=L_{answer}+\lambda_{KL}T^2\mathrm{KL}(p_{visible}^T\|p_{latent}^T)
$$

at aligned retained answer tokens. Add value-summary or compressed-cache alignment only as an ablation; unrelated-teacher raw KV is invalid.

### G. Global-workspace particles

Maintain $P$ workspace particles. A shared transition proposes updates; a small energy/value head scores particles; selection, resampling or soft mixing chooses commits. Scale test-time compute through cycles and particles separately. Controls: one particle, equal-FLOP serial recurrence, random selector, and source-only/no-carry particles.

### H. Out-there variants

| Variant | Mechanism | Sharp test |
|---|---|---|
| Reversible scratchpad | Add invertible workspace updates and rollback | Can later cycles repair an injected wrong state? |
| Predictive-coding workspace | Alternate latent-state inference and prediction-error correction | Does iterative error reduction predict answer improvement? |
| Hypernetwork compiler | Workspace emits low-rank per-example value writers or adapters | Does dynamic computation beat same-parameter static sidecar? |
| Multiscale workspace | Fast token slots + slow persistent slots | Do slow slots preserve plans while fast slots revise details? |
| Latent beam search | Branch workspace/thought tapes, score and merge | Does particle compute beat serial compute at matched FLOPs? |
| Equilibrium workspace | Solve $W=F(W,S)$ with Anderson acceleration | Does convergence correlate with correctness and generalise beyond trained depth? |
| Event-driven halting | Predict marginal value of another latent cycle | Does halting improve the accuracy/latency frontier without collapsing to fixed depth? |
| Tool-conditioned workspace | Treat observations/actions as visible commits around latent cycles | Does persistent state improve multi-turn executable tasks? |

## Training objectives

Introduce one at a time:

1. terminal authoritative-answer CE;
2. matched-vs-shuffled cache contrastive margin;
3. same-backbone visible-rationale forward KL;
4. endpoint/flow-matching loss;
5. depth consistency/anytime loss after causal recurrence works;
6. verifier or energy guidance for particles;
7. preservation loss on easy direct-answer rows.

No student CoT-token loss is required. Visible rationale remains a discarded teacher branch.

## Experiment ladder

1. Finish current 500-step config matrix and promote strong-gate/identity candidates.
2. Validate continuous latent positions, flow workspace and particle workspace with executable spikes.
3. Integrate the smallest latent-token emitter and run 32/128/500-step K1 screens.
4. Add persistent workspace commits; compare cache-only, token-only and hybrid at matched train examples and measured latency.
5. Add answer-only neural-ODE refinement; sweep 1/2/4/8 solver steps.
6. Build same-backbone rationale teacher endpoints and train KL/flow-distilled variants.
7. Combine the best emitter, workspace and flow mechanisms.
8. Add particles/selection only after one trajectory exhibits input-specific matched-vs-shuffled separation.
9. Replicate winners and evaluate exact-match scaling over cycles, latent tokens, solver steps and particles.

## Promotion metrics

Primary:

- held-out exact match;
- held-out answer NLL;
- accuracy versus measured FLOPs and latency;
- improvement with unseen extra internal compute.

Mechanism:

- matched workspace better than in-distribution shuffled/transplanted workspace;
- cutting carry removes the depth gain;
- ordered state transitions beat no-carry/source-only calls;
- extra latent positions beat equal-FLOP static soft prompts;
- later cycles correct controlled wrong states rather than merely perturbing logits.

A low answer loss with matched≈shuffled is latent compilation or a learned answer prior, not System-2 reasoning.

## External anchors

- [Coconut: Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769)
- [CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation](https://arxiv.org/abs/2502.21074)
- [SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs](https://arxiv.org/abs/2502.12134)
- [SoftCoT++: Test-Time Scaling with Soft Chain-of-Thought Reasoning](https://arxiv.org/abs/2505.11484)
- [Diffusion of Thoughts](https://arxiv.org/abs/2402.07754)
- [DiffCoT](https://arxiv.org/abs/2601.03559)
- [KaVa: Latent Reasoning via Compressed KV-Cache Distillation](https://arxiv.org/abs/2510.02312)
