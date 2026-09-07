# Cache processing

**Status:** Maintained

## When processing occurs

`process_every = P` divides a training sequence into windows. The model consumes the first window normally. Before each later window, it processes the cache left by prior windows. For length $T$, the number of calls is

$$
N_{calls}=\left\lceil\frac{T}{P}\right\rceil-1.
$$

This off-by-one matters: the processor cannot transform memory before any memory exists. By default, prefill and decoding use the same absolute cache cadence.

Set `generation_process_every` to use a separate, decoding-relative invocation cadence while leaving prefill unchanged. For example, `process_every: 64` with `generation_process_every: 1` retains 64-token prefill windows and invokes the processor before consuming every generated token. A value of `G` invokes it before decoded tokens 1, `G+1`, `2G+1`, and so on. The override changes only when the processor runs: cache selection still retains the newest `process_every` positions. `null` preserves the default behavior.

## Which positions survive

The selection rule combines recency with salience:

1. Always retain the newest `process_every` cached positions.
2. From older positions, retain the highest-scoring `topk` entries.
3. Sort the chosen indices back into chronological order.

The result preserves a local recent window and a sparse long-range memory. A score can come from the norm of the layer's attention output (`attn_l2`) or from learned key/value norm mixtures (`kv_norm`). Selection is discrete, so gradients train the representations and scoring parameters around selected states, not the top-k indices themselves.

`attn_l2` measures each layer's attention-output magnitude per token and averages those signals across depth. `kv_norm` computes RMS key and value magnitude, blends them through two learned logits, and takes the strongest layer/head response as token salience. The latter lets training decide whether key addressability or value content should dominate selection, although the subsequent top-k decision remains nondifferentiable.

Head-selective processors apply a second ranking after token selection. These controls are independent: `score` chooses historical token positions, `head_selection_method` chooses cache heads before processor execution, and `internal_iteration_score` optionally reselects heads between internal processor iterations. The default `head_selection_method: magnitude` preserves the existing raw KV-magnitude ranking and separate linear-state ranking and quotas.

Longshot can instead use `head_selection_method: learned_contextual`. After time encoding, it normalizes every active KV and linear-state head, averages all active heads into one context, and scores each candidate with learned context and candidate projections. The optional identity bias assigns one scalar to each configured head position. KV and active linear heads then compete in one ranking whose budget is the sum of `topk_heads` and the resolved `topk_linear_heads` budget. Setting `internal_iteration_score: learned_contextual` reuses the same selector after each non-final internal iteration; initial and iterative methods may be configured independently. The two roles deliberately share one selector: initial selection scores the encoded input and iterative reselection scores the post-layer hidden state, so a single set of weights ranks both.

For a parameter-free ablation, `head_selection_method: random` samples a uniform subset without replacement for every example and processor invocation. KV and linear-state heads are sampled independently: `topk_heads` is capped by the active KV-head count, while `topk_linear_heads` (or its resolved default) is capped by the active linear-head count. Unused capacity in either quota does not spill into the other. Random routing is always hard, has no learned selector or selector-training schedule, and uses the normal seeded PyTorch RNG state, so restoring the same RNG state reproduces a selection while successive invocations resample it. `select_all` remains deterministic and does not sample.

Setting `internal_iteration_score: random` resamples after every non-final internal iteration without asking processor layers to compute score tensors. With external iteration, `head_selection_method: random` likewise resamples on every processor invocation. Initial and iterative methods remain independently configurable.

For legacy paper7 evaluation, `--head-selection-method random` overrides both `head_selection_method` and `internal_iteration_score` at runtime. Learned-selector parameters from the checkpoint remain loaded but are not consulted, so every reachable initial or iterative routing decision uses parameter-free random sampling.

Hard selection remains the default for training, evaluation, and generation. An opt-in `longshot_selector_training` block on the training configuration makes learned contextual routing differentiable. It first warms the processor up with all heads, then uses an exact-budget sigmoid gate as an additive attention-logit bias while gradually lowering its temperature and blending toward hard top-k. Soft phases process all active heads so task-loss gradients reach every selector score. At the configured `hard_start`, training switches back to the same gathered top-k path used by evaluation and generation. Flow and non-contextual reselection across multiple internal iterations are rejected with this training mode because those paths do not accept routing gates.

Icarus ranks flattened `(layer, KV head)` streams by gathered KV norm. Workspace broadcast computes head salience over selected states, admits a configured writer fraction, derives token weights through a temperature-controlled softmax, and forms one writer-weighted stream. The processor's delta then reaches every selected destination head. `workspace_broadcast` changes values only; `head_mask_broadcast` can change keys and values.

## Shared processor block

The basic processor follows a residual bottleneck:

$$
H=\operatorname{ProcessorLayers}(W_{in}\operatorname{RMSNorm}(X)),
\qquad
Y=X+\gamma\odot W_{out}\operatorname{RMSNorm}(H).
$$

`ProcessorLayers` contains one or more small transformer blocks. The gate $\gamma$ either passes through a sigmoid or acts directly. Zero-initializing the output projection makes the module begin as an exact identity when the residual path and gate permit it. That initialization is useful when adding a processor to a capable pretrained model because the first optimization step does not destroy its cache.

Processor self-attention is bidirectional across the selected cache positions. This does not leak future tokens: the model invokes the processor only on memory from windows it has already consumed, before predicting tokens in the next window.

The meaning of the feature axis depends on the variant. BT mixes within one backbone layer. BGT concatenates all backbone layers. Icarus and Longshot treat every `(layer, KV head)` pair as a cache head.

## What receives gradients

No auxiliary cache loss exists. Later token cross-entropy backpropagates through the rewritten cache into the processor. With a frozen backbone, only processor parameters update, but gradients still traverse frozen operations to tell the processor how its rewrite affected predictions. With an unfrozen backbone, the optimizer updates both systems.

## Future work: prompt-boundary processing

Current SFT is windowed, not prompt-boundary-only. The processor runs every `process_every` tokens across the concatenated prompt and answer. Answer-only masking changes which tokens contribute cross-entropy; it does not change processor cadence or add an invocation at the exact prompt boundary.

A separate experiment should test one exact processor invocation after prompt prefill, before the first answer prediction. A second ablation may continue at a cadence relative to that boundary. Both require explicit per-row prompt lengths, ragged-batch handling, matching training and generation semantics, and separate provenance. They change the model's causal intervention schedule and therefore cannot replace or be aggregated with windowed results.

Flow and diffusion are distinct. Flow deterministically integrates a latent field. [Task-directed cache diffusion](task-directed-diffusion.md) injects training noise and refines a bottleneck state toward a functionally useful cache endpoint. Neither mechanism reconstructs the original cache.
