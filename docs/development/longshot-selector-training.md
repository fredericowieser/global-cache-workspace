# Longshot contextual-selector training

This note proposes a training strategy for the learned contextual head selector.
It covers differentiable top-k routing, the transition to hard selection, and the
additional controls needed to avoid selector collapse or starvation.

## Contextual-selector architecture

Longshot first selects historical token positions and encodes each KV cache head
into the processor hidden space. When recurrent linear-state heads are active,
their adapter encodes them into the same space and appends them to the KV heads.
The contextual selector therefore ranks one combined sequence of active heads;
it does not participate in the earlier token-position selection.

For encoded heads $h_1,\ldots,h_H$, the selector independently RMS-normalizes
each head, averages the normalized heads into one example-level context, and
scores every candidate using separate bias-free projections:

$$
z_i = \operatorname{RMSNorm}(h_i), \qquad
c = \frac{1}{H}\sum_i z_i, \qquad
s_i = \frac{(W_c c)^\top(W_h z_i)}{\sqrt{d_s}} + b_i.
$$

The projected dot product is accumulated in FP32. The optional $b_i$ is a
learned per-head identity bias and is disabled by default. Hard routing takes
the highest-scoring heads under the unified KV-plus-linear budget and returns
their local positions in the combined tensor.

One selector instance is shared between the two independently configurable use
sites. Initial contextual selection runs before processor execution; contextual
iterative reselection runs after all processor layers at each non-final internal
iteration. Legacy magnitude-based initial selection and the existing
attention- or norm-based iterative scores remain separate alternatives.

The selector's mean-pooled context follows the permutation-invariant aggregation
principle from
[Deep Sets](https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html).
Its candidate-wise normalization uses
[RMSNorm](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html),
and its scaled context--candidate dot product follows the attention construction
in [Attention Is All You Need](https://arxiv.org/abs/1706.03762). The resulting
selector is a Longshot-specific combination of these components.

## Recommended routing relaxation

Let the contextual selector produce one FP32 score per active head:

$$
s_i \in \mathbb{R}, \qquad i \in \{1, \ldots, H\}.
$$

During soft training, replace the discrete top-k mask with an exact-budget
sigmoid mask:

$$
p_i = \sigma\!\left(\frac{s_i - \lambda}{T}\right),
$$

where the per-example threshold $\lambda$ is chosen so that

$$
\sum_{i=1}^{H} p_i = k.
$$

Here, $T$ is the routing temperature and $k$ is the unified active-head budget.
This produces a differentiable expected selection count of exactly $k$. As $T$
approaches zero, the mask approaches hard top-k selection, assuming the scores
are not tied.

This addresses the same discrete-gradient problem studied by
[Differentiable Top-k with Optimal Transport](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html).
[NeuralSort](https://arxiv.org/abs/1903.08850) and
[Fast Differentiable Sorting and Ranking](https://proceedings.mlr.press/v119/blondel20a.html)
provide broader relaxations when a complete ordering is required. Longshot only
needs membership in an unordered size-$k$ subset, so it does not construct a
relaxed permutation or use the optimal-transport operator.

The exact-budget sigmoid is instead the solution of the entropy-regularized
cardinality problem

$$
\max_{0 < p_i < 1}
\left[
\sum_i s_i p_i
+ T\sum_i\left(-p_i\log p_i-(1-p_i)\log(1-p_i)\right)
\right]
$$

subject to

$$
\sum_i p_i = k.
$$

The first-order optimality condition for the cardinality multiplier $\lambda$
gives the sigmoid mask above. This is a simpler Longshot-specific relaxation
informed by entropy-smoothed top-k work, not the SOFT optimal-transport operator.

The closest published system-level analogue is
[SPARSEK Attention](https://arxiv.org/abs/2406.16747), which pairs a scoring
network with a differentiable top-k mask operator so that a constant-size
selection over key-value pairs remains trainable by gradient descent. SparseK
uses the clipped-linear projection $p_i = \operatorname{clip}(s_i-\lambda,0,1)$,
rather than the sigmoid relaxation proposed here. It nevertheless addresses the
same cardinality-constrained routing problem. SparseK
selects a separate subset for each query over a token sequence. Longshot
computes one shared subset per example over cache heads, where every candidate
is also a query, so the mask is smaller and no per-query solve is required.

This formulation is preferable to scaling a softmax by $k$. A softmax describes
one categorical choice, whereas the sigmoid mask represents $H$ coupled binary
selection decisions with an expected cardinality of $k$.

When $k \geq H$, routing should be bypassed and every gate should be one:

$$
p_i = 1.
$$

Those examples should not contribute to selector regularization because no
selection decision exists.

### Solving for the threshold

Define

$$
f(\lambda) = \sum_i \sigma\!\left(\frac{s_i-\lambda}{T}\right)-k.
$$

$f$ is continuous and strictly decreasing in $\lambda$, so its root is unique
when $0 < k < H$. Start with a conservative FP32 bracket
$[\min_i s_i-cT,\max_i s_i+cT]$, verify that its endpoint residuals have
opposite signs, and expand it by a fixed number of steps if necessary. The
constant $c$ must be large enough for the active $H$ and $k$; merely choosing a
small constant does not guarantee a bracket. Solve the bracketed root with a
fixed number of bisection steps under `torch.no_grad()`. Fixed iteration counts
keep both expansion and bisection traceable under `torch.compile` and avoid the
device synchronization required by a tolerance-driven `while` loop.

The threshold must still carry gradient. Rather than unrolling the solver and
differentiating through it, attach the exact derivative given by the implicit
function theorem. Differentiating $f(\lambda^*(s), s) = 0$ gives

$$
\frac{\partial \lambda^*}{\partial s_j}
= \frac{p_j(1-p_j)}{\sum_i p_i(1-p_i)}.
$$

This derivative is exact at the true root and costs $O(H)$. It avoids
backpropagating through the solver iterations, but the finite bisection result
is still an approximate root, so numerical accuracy continues to depend on the
bracket and iteration count. Clamp the denominator to a small positive value:
it collapses when every $p_i$ saturates, which happens at low $T$ once the
ranking is sharp. The solve, sigmoid, and routing logits all remain in FP32.

Newton's method converges faster than bisection, and its update is

$$
\lambda \leftarrow \lambda
+ T\frac{\sum_i p_i-k}{\sum_i p_i(1-p_i)},
$$

but that step can overshoot when $\sum_i p_i(1-p_i)$ is small. This occurs in
the low-temperature, sharply saturated regime or at extreme $k/H$, not simply
because early scores are tightly clustered. Bisection is therefore the safer
default at this problem size. Differentiable sorting methods such as
[Differentiable Top-k Classification Learning](https://proceedings.mlr.press/v162/petersen22a.html)
and convex sparse operators such as
[Fast, Differentiable and Sparse Top-k](https://arxiv.org/abs/2302.01425) are
related alternatives, but neither is evidence for this particular
threshold-solved sigmoid or its bisection implementation.

## Applying the routing gates

The selector gate should act as an attention prior. For attention logits
$a_{ji}$ from query head $j$ to candidate head $i$, compute

$$
\widetilde{a}_{ji} = a_{ji} + \log(p_i + \epsilon),
$$

followed by the normal attention softmax:

$$
\widetilde{w}_{ji}
= \operatorname{softmax}_i\!\left(\widetilde{a}_{ji}\right).
$$

As $p_i$ approaches zero, that candidate receives no attention mass. As $p_i$
approaches one, its attention logit is unchanged.

This is the continuous counterpart of the additive logit masking used by
scaled dot-product attention in
[Attention Is All You Need](https://arxiv.org/abs/1706.03762). Algebraically,

$$
\operatorname{softmax}(a_i + \log p_i)
= \frac{p_i e^{a_i}}{\sum_j p_j e^{a_j}},
$$

so $p_i$ acts as a multiplicative prior on normalized attention mass.

Multiplying only the value vector by $p_i$ is not recommended. A head with a
zero-valued contribution would still occupy probability mass in the attention
softmax denominator, shrinking the entire processor output. Adding $\log p_i$
to the attention logits performs the intended renormalized gating. The value
should not also be multiplied by $p_i$, because that would apply the gate twice.

The current `selected_heads -> gather` path is nondifferentiable. During fully
soft routing, all candidate heads must be presented as keys and values and the
soft gates must reach `ProcessorAttention`. Computing soft scores and then using
only hard indices does not provide a task-loss gradient to the selector.

## Transition to hard top-k

Let $m_i$ be the detached hard top-k mask. A smooth transition between soft and
hard routing is

$$
g_i = (1-\beta)p_i + \beta m_i,
$$

where $\beta$ increases from zero to one. Use $g_i$ rather than $p_i$ in the
attention-logit bias during this transition.

[The Concrete Distribution](https://arxiv.org/abs/1611.00712) and
[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
establish the broader use of temperature-controlled continuous relaxations that
approach discrete variables. The deterministic mask here does not sample
Concrete or Gumbel noise. The explicit hard blend and phase boundaries are
Longshot-specific engineering choices that must be validated empirically.

The gradient through the selector is multiplied by $1-\beta$. At $\beta=1$,
hard top-k is used and the selector receives no gradient unless a
[straight-through estimator](https://arxiv.org/abs/1308.3432) is deliberately
introduced.

### Initial schedule

The percentages below refer to the portion of training for which the cache
processor is active.

Schedule the blend through the suppression it actually applies rather than
through $\beta$ directly. For a non-selected head $m_i = 0$, so
$g_i = (1-\beta)p_i$ and the attention-logit bias is

$$
\log(g_i + \epsilon) \approx \log(1-\beta) + \log p_i .
$$

The blend therefore contributes $\log(1-\beta)$ nats of suppression, not
$\beta$. That function is nearly flat over most of $[0, 1]$: $\beta = 0.7$
gives only $1.2$ nats and $\beta = 0.98$ only $3.9$, so interpolating $\beta$
uniformly leaves almost all of the real hardening in its last few hundredths.
Define

$$
S = -\log(1-\beta), \qquad \beta = 1 - e^{-S},
$$

interpolate $S$ linearly, and derive $\beta$ from it. Cosine interpolation
remains suitable for $T$.

| Progress | Temperature $T$ | Suppression $S$ | Hard blend $\beta$ | Behaviour |
|---|---:|---:|---:|---|
| 0--10% | 2.0 | 0 | 0 | Fully soft exploration |
| 10--60% | 2.0 to 0.35 | 0 to 2 | 0 to 0.865 | Rankings gradually sharpen |
| 60--90% | 0.35 to 0.1 | 2 to 5 | 0.865 to 0.993 | Nearly hard routing |
| 90--100% | 0.1 | -- | 1 | Exact hard top-k |

The final phase is an exact switch rather than a continuation of the blend, so
whatever soft mass survives at $S = 5$ is discarded at that boundary. That is
the point at which the soft-versus-hard output difference in the monitoring list
matters most.

The final phase uses the existing hard gather path. The processor therefore
adapts to the exact computation used for evaluation and generation and regains
the selected-head FLOP reduction. The selector receives no task-loss gradient
during this phase because the hard mask is detached.

Exact hard routing does not inherently make
`DistributedDataParallel` fail. If the selector remains connected to the loss,
for example through a zero-multiplied differentiable term, its parameters are
used by autograd and receive zero-valued gradients. If the implementation
branches around the selector in the hard phase, its parameters are genuinely
unused; the opt-in selector-training configuration must then construct DDP with
`find_unused_parameters=True`. Dynamically changing `requires_grad` is not a
substitute because DDP fixes its registered parameter set at construction.

If continued selector learning is required during the hard phase, use a
straight-through estimator explicitly and treat its gradient as a biased
optimization heuristic. It is not part of the initial implementation.

The transition should be driven by metrics rather than percentages alone. Delay
annealing when selector gradients remain very small, the score margin is not
growing, or hard and soft processor outputs still differ substantially.

## Sparse-training alternatives

Fully soft routing processes all active candidate heads, so it does not provide
the inference-time top-k FLOP reduction during training. There is no method that
both omits an unselected head from the forward computation and obtains the exact
task-loss gradient through that omitted computation.

If dense training is too expensive, use a candidate pool consisting of:

1. The current hard top-k heads.
2. A randomly sampled set of additional heads.
3. Optionally, heads near the top-k boundary.

Apply the soft routing relaxation within this pool. Random candidates ensure
that heads outside the current top-k receive gradients over time. Sampling must
be reproducible across gradient-checkpoint recomputation.

Other possible relaxations include
[Gumbel-Softmax](https://arxiv.org/abs/1611.01144),
[NeuralSort](https://arxiv.org/abs/1903.08850), and
[optimal-transport top-k](https://proceedings.neurips.cc/paper/2020/hash/ec24a54d62ce57ba93a531b460fa8d18-Abstract.html).
[Sparsemax](https://proceedings.mlr.press/v48/martins16.html) and
[hard-concrete gates](https://arxiv.org/abs/1712.01312) provide related ways to
obtain exact zeros. These alternatives add stochasticity, solve a more general
ranking problem, or do not enforce the required fixed combined budget. The
exact-budget sigmoid mask is therefore the recommended first implementation.

## Processor warmup and gradient starvation

The selector receives gradients only through the cache processor's effect on the
model loss. With sigmoid gating and the current default

$$
\operatorname{sigmoid}(-5) \approx 0.0067,
$$

the processor update begins almost closed. If the output projection is also
zero-initialized, the selector initially receives no useful gradient through
that projection.

Two reasonable approaches are:

1. Run the processor with all heads active for roughly 2--5% of training, then
   enable fully soft routing once the processor has acquired a useful signal.
2. For selector experiments, initialize the processor gate between $-3$ and
   $-2.2$, corresponding to sigmoid scales of approximately $0.047$ and $0.100$.

The first approach is safer when adding a new processor to a pretrained
backbone. Temperature annealing should not begin until the processor itself is
learning.

## Initial and iterative selection

Initial routing and iterative reselection use the same routing schedule. During
processor warmup, both select all active heads. If both are configured, they
then enter soft routing together and share the same temperature and hard blend
until the final hard phase. This keeps schedule state with the shared selector
and avoids separate controls for its two call sites.

Metrics are recorded separately for initial selection and every iteration
boundary because the shared selector is applied to different hidden-state
distributions. A future experiment could introduce a separate iterative start
point or temperature if simultaneous routing proves unstable, but the initial
implementation does not provide independently scheduled routing phases.

## Regularization

The main task loss should remain the primary selector objective. Small temporary
regularizers can prevent early collapse.

### Entropy

If monitoring shows premature collapse, a later experiment can add a weak
entropy bonus during the fully soft phase:

$$
\mathcal{H}(p)
= -\frac{1}{H}\sum_i
\left[p_i\log(p_i+\epsilon)
+ (1-p_i)\log(1-p_i+\epsilon)\right].
$$

Its coefficient should decay to zero as routing becomes hard. Permanent entropy
maximization would oppose confident specialization. The initial implementation
defers this auxiliary loss so the task-loss-only baseline remains interpretable.

### Batch utilization

Let

$$
u_i = \frac{1}{B}\sum_b p_{bi}.
$$

A weak warmup-only balance penalty is

$$
\mathcal{L}_{\mathrm{balance}}
= \frac{1}{H}\sum_i\left(u_i-\frac{k}{H}\right)^2.
$$

This should only prevent heads from becoming permanently starved early in
training. It should not force uniform utilization throughout training: stable
global heads may genuinely differ in usefulness.

Noisy top-k routing and auxiliary utilization losses were introduced for sparse
conditional computation in
[Outrageously Large Neural Networks](https://arxiv.org/abs/1701.06538).
[Switch Transformers](https://www.jmlr.org/papers/v23/21-0998.html) and
[ST-MoE](https://arxiv.org/abs/2202.08906) further document stability problems
in learned sparse routers. Those systems route to largely interchangeable
experts; Longshot routes stable state identities, so their balancing objectives
motivate starvation monitoring but not a permanent assumption of uniform use.

[On the Representation Collapse of Sparse Mixture of Experts](https://arxiv.org/abs/2204.09179)
documents the complementary failure, in which a learned router drives candidate
representations toward a small number of centroids. One proposed mitigation is
a low-dimensional hyperspherical routing space. Longshot's small selector
projection is related, but RMS-normalizing its inputs does not normalize the
projected context and candidate vectors onto a hypersphere, so it should not be
treated as implementing that mitigation. The practical consequence is that
routing diversity must be monitored across the batch and not only across
iterations: high
iteration-to-iteration overlap is expected, whereas few distinct heads selected
across a batch indicates the selector has become context-independent.

Identity biases deserve particular attention because they can create permanent
global favourites. They are disabled by default
(`head_selector_use_identity_bias = False`) so that the first experiments
measure contextual selection on its own.

The two straightforward mitigations, mild L2 regularization and a reduced
learning rate, require optimizer machinery this repository does not have.
`_build_optimizer()` in
`src/gcw/training/optimizer.py` builds exactly two groups, split on
`p.ndim >= 2` and sharing one learning rate, so a one-dimensional bias lands
unconditionally in the `weight_decay = 0.0` group. Mild L2 on identity biases
therefore requires excluding them from that rule, and a reduced identity-bias
learning rate requires a third parameter group. Both are deferred with the bias
itself, and neither is needed while it is disabled.

A per-head identity bias is related to the static importance gates studied by
[Analyzing Multi-Head Self-Attention](https://aclanthology.org/P19-1580/) and
[Are Sixteen Heads Really Better Than One?](https://proceedings.neurips.cc/paper/2019/hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html).
Both find that most attention heads can be removed with little loss once such
gates are trained, which supports head-level selection as a meaningful
operation. They also bound what the identity bias alone can demonstrate: those
works gate a model's own attention heads statically, so if the bias comes to
dominate the contextual score, the selector has reproduced a static gating
result rather than a contextual one. Recording the ratio of identity-bias
variance to context-score variance distinguishes the two.

A brief auxiliary ranking target based on the existing magnitude or attention
scores can warm-start the selector, but it should decay quickly so the learned
router is ultimately optimized by the task loss rather than constrained to
reproduce the legacy heuristic.

## Stable head identities

`head_selector_use_identity_bias` defaults to `False`, so the first experiments
train a purely contextual selector. With the bias disabled there is no consumer
of global head identities at all: `selected_heads` holds positional row indices
into `hidden`, and every other use of head position is likewise positional.

The initial implementation therefore threads no `head_indices`, and
`GlobalCacheProcessor._linear_state_inputs()` keeps the `RuntimeError` it raises
when a linear-state group has no recurrent state. That raise is what makes
positional packing safe: a missing group is unreachable rather than silently
compacted, so no head can inherit another head's slot.

Identities become necessary only when `head_selector_use_identity_bias` is
enabled, because a per-head bias is the one parameter indexed by head rather
than by position. At that point:

- Restore an explicit `head_indices` input and gather the active biases:

$$
s_i \leftarrow s_i + b_{\text{head\_indices}_i}.
$$

- Assign KV heads the first global range, and give linear-state heads their
  positions from the complete configured specification order.
- Decide deliberately whether an absent linear group should raise or be skipped.
  Skipping requires identities; raising does not. The two cannot be mixed,
  because a skipped group under positional packing lets a later head inherit the
  wrong learned bias.

## Optimizer considerations

- Give selector projections the same learning rate as other newly initialized
  processor parameters. Raising it requires a third parameter group, because
  `_build_optimizer()` in `src/gcw/training/optimizer.py` currently builds only a
  decay and a no-decay group sharing one learning rate.
- Identity biases would want a lower learning rate still, but they are disabled
  by default and the required group does not exist. Deferred with the bias.
- Apply the existing global gradient clipping and report selector gradient norms
  separately.
- Do not apply routing regularizers on examples where every active head is
  selected.
- Preserve FP32 score accumulation and gate calculation under mixed precision.
- If routing noise or candidate sampling is used, generate it outside
  checkpointed regions or preserve the RNG state across recomputation.

## Required monitoring

At minimum, record:

- Selector projection and identity-bias gradient norms.
- Score mean and standard deviation.
- Soft-mask entropy and total soft mass.
- Selection rate for every head position, and the number of distinct heads
  selected anywhere in a batch.
- KV-versus-linear selection proportions.
- The score gap between the $k$-th and $(k+1)$-th heads.
- The difference between soft-routed and hard-routed processor outputs.
- Initial-to-iterative and iteration-to-iteration selection overlap.
- The processor output-gate value.
- The fraction of heads that have received no meaningful routing mass over a
  recent training window.

Hard-routing evaluations should run periodically before the final hard phase.
Validation quality under soft routing alone can conceal a large deployment-time
discontinuity.

## Recommended first experiment

The smallest informative experiment is:

1. Confirm `head_selector_use_identity_bias` is disabled, so the first result
   measures contextual selection alone.
2. Warm up with all heads active for 2--5% of training.
3. Use dense exact-budget sigmoid routing with attention-logit gates.
4. Apply the temperature and hard-blend schedule above.
5. When iterative contextual reselection is configured, apply the same routing
   schedule at the initial and iterative call sites.
6. Use no auxiliary routing loss initially; add a decaying entropy bonus only
   if monitoring shows premature collapse.
7. Run periodic hard-routing validation.
8. Use exact hard gather for the final phase and account explicitly for its
   unused selector parameters under DDP.

Only add stochastic routing, straight-through gradients, or load-balancing loss
if the corresponding monitoring shows that the simpler formulation is
insufficient.

## Initial Implementation plan

### Summary

Implement the core training recipe described above: processor warmup,
differentiable exact-budget routing, scheduled hardening, hard evaluation, and
routing telemetry. Legacy configurations remain hard-routed and preserve their
current execution and state-dict behaviour.

### 1. Training configuration and schedule

- In `src/gcw/config/training.py`, add
  `LongshotSelectorTrainingConfig` and optional
  `TrainConfig.longshot_selector_training`.
- Expose the new configuration from `src/gcw/config/__init__.py`.
- Use these validated defaults:
  - Head warmup ends at `0.05`.
  - The fully soft phase ends at `0.15`.
  - The second hardening phase begins at `0.60`.
  - Hard routing begins at `0.90`.
  - Temperatures are `2.0`, `0.35`, and `0.1`.
  - Suppression values are `0.0`, `2.0`, and `5.0` nats. Configure these rather
    than the blend, and derive `beta = 1 - exp(-S)` from them; the final phase
    uses exact hard routing rather than a blend value.
- Interpolate $T$ with cosine and $S$ linearly between phase boundaries.
- Reject this configuration unless Longshot uses `learned_contextual` for
  initial selection, iterative selection, or both.
- Reject it when `flow` is configured. `_iterate_layers()` dispatches to
  `LatentFlow` before any routing phase is resolved, and `_run_flow_field()`
  calls `_run_layer()` through a closure that carries no gates, so the flow
  field would run hard-routed while the rest of the processor ran soft. That
  combination is silent rather than wrong-looking, so it is rejected at
  configuration time until the gates are threaded through the flow closure.
- Reject it when soft routing would coexist with non-learned internal
  reselection, that is when `iteration_mode` is `internal`, `iterations` exceeds
  one, and `internal_iteration_score` is not `learned_contextual`. Only those
  combinations set `return_attn_scores`, which routes through
  `ProcessorAttention.forward_with_iteration_scores()`; its
  `cache_head_attention` branch reads normalized attention weights that SDPA
  does not return. Requiring either a single internal iteration or learned
  iterative reselection keeps the gated path confined to
  `forward_selected_head_kv()` and `forward_with_self_head_kv()`. Gating the
  scoring methods is possible and is deferred, not ruled out.
- Enforce that restriction in the training configuration only. All three
  iterative scoring modes stay valid for hard-routed training, evaluation, and
  generation, where no gates exist and the current code paths are unchanged.
- Mark the configuration as requiring unused-parameter discovery during its
  exact hard phase.
- Add the routing configuration to `_training_fingerprint()` in
  `src/gcw/training/audit.py` so resume cannot silently change schedules.

### 2. Differentiable routing

- In `src/gcw/models/cache_processors/longshot_selection.py`:
  - Keep `ContextualHeadSelector.forward(hidden)` as it stands. Identity bias is
    disabled by default, so no head-identity input is threaded.
  - Add `_soft_topk_mask(scores, topk_heads, temperature)`.
  - Establish and verify an FP32 root bracket, expand it with a fixed bounded
    loop when needed, solve with fixed-step bisection, and attach the analytical
    implicit gradient for the threshold.
  - Return all-one gates when the budget covers every active head.
  - Resolve warmup, soft, transition, and hard routing states from training
    progress.
  - Compute the context--candidate dot product inside
    `torch.autocast(..., enabled=False)`. Casting both operands with `.float()`
    is not sufficient: `torch.einsum` is on autocast's lower-precision list, so
    under `bf16_mixed` it downcasts them again and the scores come back in
    bf16. Ranking 160 near-tied scores at bf16 resolution makes top-k ties, and
    therefore arbitrary tie-breaking, likely at initialization. A dtype
    assertion alone does not catch the regression when the identity bias is
    enabled, because adding the FP32 bias promotes an already-quantized bf16
    score back to FP32.
  - Give `ContextualHeadSelector` ownership of everything downstream of its own
    scores: a `set_training_progress()` method holding the resolved mode,
    temperature, and hard blend in non-persistent buffers, a `route()` method
    returning hard indices plus optional log-space gates, and `pop_metrics()`
    for its own telemetry. Routing state stays off the processor so schedules,
    relaxations, and selector metrics are maintained in one module.
- Leave `LinearStateInputs` and `GlobalCacheProcessor._linear_state_inputs()`
  unchanged, including the `RuntimeError` raised for a missing recurrent state.
  Positional packing is sufficient while no parameter is indexed by head
  identity.

### 3. Processor routing

- In `src/gcw/models/cache_processors/longshot_processor.py`:
  - Keep `select_heads()` as the hard-selection operation.
  - Add one private `_route_heads()` method, reused by initial selection and
    iterative reselection, that guards the optional selector and delegates to
    `ContextualHeadSelector.route()`. The processor decides *when* to route; the
    selector decides *how*.
  - Extend `forward()`, `_iterate_layers()`, and `_run_layer()` only with the
    selection-trace and routing-gate data required by the shared path.
  - During warmup, use every active head without calling the selector.
  - During soft phases, retain hard top-k indices for metrics while supplying
    all active heads and their blended gates to attention.
  - During the final phase and in evaluation mode, use the existing hard top-k
    path.
  - Leave magnitude selection, non-learned iterative scoring, flow execution,
    projections, residual gates, and writeback unchanged.
- Add a model-wide `set_longshot_selector_training_progress()` operation in
  `longshot_selection.py` with wrapper traversal and progress validation. It
  locates `ContextualHeadSelector` modules directly rather than routing the call
  through their processors.
- In `run_training_loop()` in `src/gcw/training/core.py`, set progress once per
  optimizer step using `step / max(1, max_steps - 1)`. Resume therefore derives
  the same routing state from the restored step without checkpointing mutable
  schedule state.
- Set it at the top of the step body, before the microbatch loop at
  `core.py:183`. It must not be placed beside the learning-rate and weight-decay
  schedule at `core.py:277-289`, which runs after every forward and after
  `backward()`: every forward in step $n$ would then use step $n-1$'s progress,
  and step $0$ would use the initial value.
- In `src/gcw/training/assembly.py`, construct DDP with
  `find_unused_parameters=True` when differentiable selector training is
  enabled, because the final exact-hard branch does not use selector parameters.
  Preserve the existing DDP setting for every other configuration.

### 4. Attention gating

- In `src/gcw/models/cache_processors/longshot_layers.py`, extend
  `ProcessorLayer.forward()` and `ProcessorAttention.forward()` with an optional
  routing-gate tensor.
- Keep the single existing `flash_sdpa()` call in `forward_selected_head_kv()`
  and pass `attn_mask=gate_bias` with `try_flash=gate_bias is None`. Cast the
  bias to the attention compute dtype, as the call site already does for `q` and
  `k`: an FP32 mask against bf16 queries disqualifies the memory-efficient
  backend and silently falls through to the math backend. The bias is a
  suppression term rather than a ranking signal, so it does not need the FP32
  treatment the selector scores do. No new
  attention forward and no separate branch are required: when the gate tensor is
  absent, `attn_mask=None` is already the default and `try_flash` is already
  `True`, so the call is identical to the current one and the Flash path is
  preserved.
- The fallback is therefore selected by the presence of gates, not by training
  state. The attention layer never learns about the schedule: §3 resolves
  warmup, soft, transition, and hard routing in the processor, and gates are
  non-`None` only during the soft and transition phases. Warmup, the hard phase,
  evaluation, and generation all pass `None` and keep Flash.
- `try_flash=False` is required whenever the mask is present because the Flash
  backend rejects arbitrary additive masks, and
  `src/gcw/models/backbones/transformer/layers.py:171` re-raises rather than
  falling back whenever `GCW_STRICT_FLASH=1`, which
  `scripts/cluster/slurm/train.sbatch`, `scripts/cluster/slurm/eval.sbatch`, and
  `scripts/cluster/submit_array.sh` all export by default. Requesting the
  non-Flash backend at the call site reaches it without raising.
- Do not clear `GCW_STRICT_FLASH` globally to obtain the same fallback. It
  resolves to the same backend, but it is the backend-evidence guard for the
  backbone's own attention, so disabling it would hide an unrelated Flash
  regression for the whole run, and it would make behaviour depend on whether
  the job was launched through the cluster scripts.
- `torch.compile` guards on `gate_bias is None`, so it recompiles once at each
  phase boundary that changes gate presence. Three recompiles per run.
- When gates are present:
  - Project all active heads as keys and values.
  - Add `log(gate + 1e-6)` as an FP32 attention-logit bias, shaped
    `(batch, 1, 1, H)` so it broadcasts over queries and grouped-query heads.
  - Do not multiply values by the gate again.
  - Support normal attention and `include_self_head_kv`. The latter needs care:
    in the hard path the query's own head is appended as an extra key beyond the
    selected set, whereas in the dense gated path it is already one of the `H`
    candidates and carries whatever gate it was assigned. Force its gate to one
    along the diagonal to preserve the current semantics.
  - Leave `forward_with_iteration_scores()` and
    `forward_with_cache_head_attention_scores()` untouched. §1 rejects the
    configurations that would reach them while gates are present.
  - Raise in `ProcessorLayer.forward()` when gates accompany
    `return_attn_scores`. `ProcessorLayer` dispatches straight to
    `forward_with_iteration_scores()`, which takes no gates, so without this
    guard the §1 rejection is the only thing standing between that combination
    and a selector that silently receives no gradient at all. The equivalent
    guard on `ProcessorAttention.forward()` is unreachable from the layer.
- Recompute learned soft gates after each applicable internal iteration using
  the updated hidden states.

### 5. Telemetry and FLOPs

- Accumulate temperature, hard blend, score spread, mask entropy, top-k
  boundary margin, mask mass, head-utilization aggregates, and the
  KV-versus-linear routing fraction inside the selector, and merge
  `ContextualHeadSelector.pop_metrics()` into
  `CacheProcessor.pop_iteration_metrics()`. Report aggregates rather than
  per-head series: at 160 cache heads the per-head form dominates the routing
  cost it measures.
- Defer direct soft-versus-hard processor-output comparison, processor
  output-gate telemetry, and rolling meaningful-mass starvation tracking to a
  follow-up monitoring implementation. The initial implementation reports
  current-batch hard-selection starvation only.
- Extend `cache_processor_metrics()` in `src/gcw/performance/telemetry.py` with
  selector projection and identity-bias gradient norms, and with the processor
  output-gate value for every variant that has one.
- Record the gate inside `_gate_scale()` rather than reading `block.gate` from
  telemetry. Under FSDP the cache processor is not a wrapped unit, so its
  parameters live in the sharded root flat parameter and rank 0 may hold none of
  a one-element gate; recording during the forward captures the all-gathered
  value on every rank. Logging is rank-0 only
  (`src/gcw/training/core.py:402`), so a telemetry-time read could drop the
  metric silently.
- In `src/gcw/flops.py`, distinguish:
  - Warmup: dense all-head attention without selector scoring.
  - Soft and transition phases: dense all-head attention plus selector and gate
    costs.
  - Hard phase: selected-head attention plus selector costs.
- Preserve the existing public FLOP defaults as hard-routing estimates.
- Update `TrainingPlan`, `assemble_training_execution()`, and
  `run_training_loop()` to use phase-aware estimates for accumulated and
  per-step training FLOPs.

### 6. Documentation and example configuration

- Update `docs/concepts/cache-processing.md` to distinguish hard inference from
  opt-in differentiable training.
- Add a small Longshot training configuration demonstrating
  `LongshotSelectorTrainingConfig`.
- Regenerate configuration reference documentation with
  `scripts/docs/generate_reference.py`; do not edit generated files manually.

### Tests

- Extend `tests/test_longshot_contextual_selection.py` for exact mask mass,
  FP32 execution, schedule boundaries, analytical gradients, warmup, hard
  evaluation, and initial and iterative routing combinations. The existing
  missing-linear-state raise and the positional identity-bias path are already
  covered and should keep passing unchanged.
- Extend `tests/test_longshot_attention.py` for all-one equivalence, near-zero
  exclusion, selector gradients, `include_self_head_kv`, dropout, and mixed
  precision.
- Extend training and checkpoint tests for deterministic resume, gradient
  accumulation, DDP/FSDP traversal, and checkpoint recomputation.
- Extend `tests/test_flops.py` for warmup, soft, transition, and hard counts.
- Run Ruff, focused Longshot/config/training/FLOP tests, the full CPU suite, and
  the warnings-as-errors documentation build.

### Assumptions and deferred work

- Soft routing is opt-in through `TrainConfig`; omitting it preserves current
  hard routing.
- Token-position selection is unchanged. All-head warmup does not select all
  historical tokens.
- Evaluation and generation always use hard top-k.
- The final phase uses exact hard gather and gives the selector no task-loss
  gradient. Opt-in DDP runs use unused-parameter discovery; optimizer groups are
  never changed dynamically.
- Gating the iterative scoring methods is deferred, so soft routing requires
  either a single internal iteration or learned iterative reselection.
- Identity bias stays disabled, so stable global identity threading,
  `head_indices`, and the optimizer groups those would need are all deferred.
- Entropy and balance losses, selector-specific learning rates,
  straight-through estimators, Gumbel noise, sampled candidates, and ranking
  distillation are deferred.
- Soft-versus-hard output comparison, processor output-gate telemetry, and
  rolling meaningful-mass starvation tracking are deferred to a later
  monitoring implementation.
- Soft phases intentionally pay dense processor-attention FLOPs so every active
  candidate can receive meaningful gradients.
