# Model variants

**Status:** Maintained

## TF: transformer baseline

`TF` runs the decoder without a processor. It defines the numerical baseline, owns the common training and generation implementation, and exposes the cache interfaces that processed variants reuse.

## BT: layer-local bottleneck transformer

BT creates one independent processor for each backbone layer. At one selected time position, layer $\ell$ exposes either

$$
[K^{(\ell)};V^{(\ell)}]\in\mathbb R^{2H_{kv}d_h}
$$

or a value-only vector in $\mathbb R^{H_{kv}d_h}$. Each processor mixes selected positions through its own small transformer, then writes the result only to that layer's cache.

BT's inductive bias is local in depth: it can reorganize time within a layer but cannot directly exchange cache features across layers. Its separate processors increase parameter count with backbone depth but make layer ownership explicit.

## BGT: global bottleneck transformer

BGT concatenates cache features from all layers at each selected time position, applies one shared processor, and splits the output back into layers. It gives the processor a global depth-wise view:

$$
X_t=\operatorname{concat}_{\ell=1}^{L}[K_t^{(\ell)};V_t^{(\ell)}].
$$

The processor can therefore route information across layers as well as across selected time positions. This increases feature width, so configurations tune its internal width and head count to control cost.

Two BGT-derived workspace modes narrow the write pattern:

- `workspace_broadcast` selects writer heads, builds a salience-weighted global value stream, processes it, and broadcasts the value delta back to selected states.
- `head_mask_broadcast` builds a similar workspace but writes both key and value deltas under explicit head masks.

These are architectural interventions, not analysis-time head ablations.

## Icarus: cache-head cross-attention

Icarus flattens layer and KV-head axes into $C=L H_{kv}$ cache heads. It gathers selected time positions, optionally removes rotary position encoding from keys, scores cache heads by gathered key/value magnitude, and lets every cache head query the selected heads. In schematic form:

$$
\widehat X_{c,:}=\operatorname{CrossHeadProcessor}
\left(X_{c,:},\operatorname{TopHeads}(X)\right).
$$

The module writes only the latest selected cache position, not every gathered position. That choice turns selected history into an update for the current memory frontier. Per-cache-head projections give each head independent parameters; shared projections trade expressivity for a smaller and more regular processor. When `unrope` is active, the module processes position-neutral keys and reapplies RoPE before storage.

## Longshot: time compression before head mixing

Longshot also treats `(layer, KV head)` as the cache-head axis, but it first compresses each head's selected time sequence into one vector. It supports:

- fixed-length flattening followed by a learned linear map;
- query attention with one learned query;
- query attention whose query comes from the latest selected token.

A cross-head processor then mixes those summaries. The residual anchors at the latest selected state, and the module writes one updated frontier position. Analysis can reapply this processor $k$ times to the same prefix cache. Those iterations probe recurrent refinement of memory; they do not generate $k$ additional tokens.

Longshot computes cache-selection magnitudes with the same eager FP32 expressions in bounded outer-row chunks. Each FP32 cast or square allocation is capped at 32 MiB, replacing a full-cache temporary; BF16 forward autograd saves the existing BF16 source rather than a full FP32 conversion. Chunking never splits a reduction row, so the feature-axis accumulation order and top-k decisions match the eager oracle. A single reduction row larger than 32 MiB exceeds the cap by design: feature chunking would change reduction order and can flip adversarial near ties.

For Qwen3.5 linear-attention states, `live_rank1_linear_writer: true` initializes the outer-product writer with random key rows and zero value rows. This preserves an exact zero initial correction while allowing the value factor to receive gradients immediately; zeroing both factors blocks that gradient. The option requires `linear_state_cache: true`, `linear_state_update_mode: outer_product`, and `out_proj_init: zero`. It preserves parameter shapes and subsequent RNG state. The disabled default preserves historical initialization where applicable. Config validation rejects the combination of `linear_state_cache: true`, `linear_state_update_mode: outer_product`, and `out_proj_init: zero` when the live writer is disabled, because both factors would stay at zero. Normal output initialization and gated-delta updates are unaffected.

## Variant comparison

| Variant | Reads across time | Mixes layers directly | Mixes cache heads directly | Write target | Principal control |
|---|---:|---:|---:|---|---|
| TF | normal attention | through backbone only | through attention/output projections | append-only cache | causal-transformer baseline |
| BT | yes | no | inside one layer vector | selected states in each layer | layer-local processor |
| BGT | yes | yes | through concatenated feature | selected states across layers | global depth workspace |
| Icarus | selected positions | yes | yes | latest selected position | cross-head memory routing |
| Longshot | compressed selected history | yes | yes | latest selected position | recurrent compressed memory |
