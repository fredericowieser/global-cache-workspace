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

## Cache layout and layer selection

The backbone determines which layers expose KV caches. Ordinary transformers expose one cache per attention layer. Hybrid backbones can also carry other state: Qwen3.5 exposes KV caches from full-attention layers and recurrent state from linear-attention layers. Longshot can additionally process those recurrent states with `linear_state_cache: true`; they are not additional KV shape groups. Backbones with sliding and full attention can expose KV caches with different head counts or dimensions.

`processor_layer_indices` identifies the backbone layers exposed to KV processing and must reference eligible cache-bearing layers. Arbitrary subsets currently work with Gemma4; Qwen3.5 requires all full-attention layers in its KV layer set. Excluding a layer from processing does not remove the state needed by its attention computation. Training and ordinary `generate` retain unselected attention histories in `attention_states`, including across cache conversion, detachment, and activation-checkpoint boundaries. Gemma4 static generation instead bounds unselected sliding-layer storage to the attention window; unselected full-attention layers still retain full history.

BT, BGT, and Icarus require the selected KV layers to have matching shapes. Longshot also supports mixed shapes: selected features are grouped by KV head count and head dimension, with shape-specific normalization, time encoding, and input/output projections around one shared processor. Head identities remain ordered by selected backbone layer, then KV head. `per_cache_head_projections: false` shares projections within each shape group; `true` gives each head its own projections. Output gates belong to the corresponding shape adapter.

Processor-visible KV archives retain their native dimensions and full accumulated history, including for selected sliding-attention layers. Gemma4 Longshot training uses per-layer list storage by default. History selection scores native views in bounded FP32 tiles, gathers selected positions before packing features, and avoids stacking full histories by shape. The explicit `WindowedCache` and `GroupedWindowedCache` formats remain available, including for execution policies that require packed storage. `select_all=True` intentionally gathers the complete history. Training appends and writeback remain functional operations so gradients can cross recurrent windows; list storage does not eliminate those per-layer copies.

Gemma4 Longshot `generate_fast` allocates a static cache for the truncated prompt length plus the requested number of new tokens, supporting both mixed shapes and layer subsets. Selected layers and unselected full-attention layers have full request capacity. Unselected sliding layers have ring buffers of capacity `min(sliding_window, request_capacity)`. Absolute sequence length is separate from ring occupancy, appends fail if they exceed capacity, and processor updates preserve the backing allocations. Static writes are inference-only. Ordinary `generate` remains the dynamic full-history storage reference.

Attention reads are bounded independently of processor archives. For a chunk starting at absolute position $s$ with $P$ tokens and sliding window $W$, sliding attention reads positions $[\max(0,s-W+1),s+P)$ and masks using absolute positions. This retains the history required by the earliest query in a multi-token chunk. Shared-KV consumers receive the same view and absolute offset. Ring attention may assemble a temporary of up to $W-1+P$ positions, even though persistent ring storage is bounded by $W$.

For batch size $B$, retained length $T$, and $b$ bytes per element, full-history KV storage alone occupies $2bBT\sum_\ell H_\ell D_\ell$ bytes, where $H_\ell$ and $D_\ell$ are each cache-bearing layer's KV head count and dimension. The 50 sliding layers with 16 heads of dimension 256 and 10 full-attention layers with 4 heads of dimension 512 in Gemma4 31B require 819,200 and 81,920 bytes per token respectively in BF16. At 32,768 tokens, training or dynamic generation retains 27.5 GiB per sequence, including when only full-attention layers are processed. Static generation with all layers selected also needs 27.5 GiB at that request capacity. With only full-attention layers selected and a 1,024-token sliding window, static generation retains approximately 3.28 GiB: 2.5 GiB for full-attention archives and 0.78125 GiB for sliding rings. These figures describe persistent KV storage, not peak memory; weights, scores, processor state, activations, and attention/scoring temporaries are additional. Budget context length and batch size together.

Longshot's `head_score_normalization` defaults to `none`, preserving existing scoring and checkpoint identities. Set it to `rms` when comparing head magnitudes across widths: squared KV scores are divided by twice the head dimension, and L2 scores by the square root of that factor, before the existing temporal aggregation. This does not change per-head history ranking, learned/random routing, or scores computed in the shared latent space. It compensates for feature width without imposing a quota for either attention type.

Execution options must support the selected cache layout. Mixed-shape caches use reference cache movement and reject the optional contiguous-cache full-history, KV-score-sidecar, and Triton movement optimizations. Inverse RoPE also depends on the backbone: Gemma4 Longshot requires `unrope_keys: false`, so keys remain in their layer-specific rotated frame. See [training configuration](../guides/configure-training.md#native-activation-checkpointing) for window activation-checkpointing support and restrictions.

## What receives gradients

No auxiliary cache loss exists. Later token cross-entropy backpropagates through the rewritten cache into the processor. With a frozen backbone, only processor parameters update, but gradients still traverse frozen operations to tell the processor how its rewrite affected predictions. With an unfrozen backbone, the optimizer updates both systems.

## Future work: prompt-boundary processing

Current SFT is windowed, not prompt-boundary-only. The processor runs every `process_every` tokens across the concatenated prompt and answer. Answer-only masking changes which tokens contribute cross-entropy; it does not change processor cadence or add an invocation at the exact prompt boundary.

A separate experiment should test one exact processor invocation after prompt prefill, before the first answer prediction. A second ablation may continue at a cadence relative to that boundary. Both require explicit per-row prompt lengths, ragged-batch handling, matching training and generation semantics, and separate provenance. They change the model's causal intervention schedule and therefore cannot replace or be aggregated with windowed results.

Flow and diffusion are distinct. Flow deterministically integrates a latent field. [Task-directed cache diffusion](task-directed-diffusion.md) injects training noise and refines a bottleneck state toward a functionally useful cache endpoint. Neither mechanism reconstructs the original cache.
