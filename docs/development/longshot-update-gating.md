# Longshot data-dependent update gating

This note describes scalar and channel-wise data-dependent gates for Longshot
cache updates and records the architecture and implementation details.

## Current architecture

Longshot compresses the selected history for each cache head, mixes the
resulting head representations through its processor layers, and projects each
processed representation into a cache update. The update is added to the latest
selected cache state.

By default, one learned scalar controls every update in every example and cache
head. In simplified notation, for processed cache head `c`:

$$
\Delta_c = W_{out}(z_c), \qquad
x'_c = x_c + \operatorname{sigmoid}(a)\Delta_c.
$$

Here, $z_c$ is the final normalized processor representation, $W_{out}$ is the
output projection, and $a$ is the shared learned scalar. `gate_mode: direct`
uses $a$ directly instead of applying the sigmoid.

## Architecture

The implementation keeps the static gate as the default and provides two
opt-in alternatives. For the contextual alternatives, the existing output
projection is expanded so one matrix multiplication produces both the proposed
update and its gate logits:

1. A contextual scalar for each example and cache head.
2. A contextual vector for each example, cache head, and output channel.

The three forms are:

$$
\begin{aligned}
\text{static scalar:}\quad
  &\Delta_c = W_{out}z_c + b_{out},
  &&\gamma = \operatorname{sigmoid}(a), \\
\text{contextual scalar:}\quad
  &[\Delta_c;g_c] = W^+_{out}z_c + b^+_{out},
  &&\gamma_c = \operatorname{sigmoid}(g_c), \\
\text{contextual vector:}\quad
  &[\Delta_c;\boldsymbol{g}_c] = W^+_{out}z_c + b^+_{out},
  &&\boldsymbol{\gamma}_c = \operatorname{sigmoid}(\boldsymbol{g}_c).
\end{aligned}
$$

The gated residual update remains:

$$
x'_c = x_c + \gamma_c \odot \Delta_c.
$$

For the contextual scalar, the expanded projection adds one gate-logit output.
For the contextual vector, it adds one gate logit per cache-update channel:
$d_h$ for a value-only update and $2d_h$ for a combined key/value update. The
complete projection widths are therefore `output_dim + 1` and
`2 * output_dim`, respectively. This makes the vector option channel-wise over
the value actually written to the cache rather than over Longshot's internal
hidden dimension.

The gate input should be the final `processed` representation after the
existing output RMSNorm. At that point it contains the compressed history and
the result of cross-head mixing, so it is the most informative local signal for
deciding how much of the proposed update to admit.

Select the gate with `cache_processor.update_gate_type`:

```yaml
model:
  variant: longshot
  cache_processor:
    update_gate_type: contextual_scalar  # or contextual_vector
    gate_mode: sigmoid
    gate_init: -5.0
```

The default is `static_scalar`, which preserves the original architecture and
checkpoint layout. Contextual gate types are accepted only for Longshot.

Although a shared output projection may be applied to every cache-head row, the
resulting gates remain cache-head-dependent because each row has different
data. When `per_cache_head_projections` is enabled, expanding the existing
`PerCacheHeadLinear` output projection naturally gives each stable cache head
separate delta and gate parameters.

## Initialization and compatibility

The delta and gate slices of an expanded output projection require different
initialization:

- Initialize the delta slice according to `out_proj_init`.
- Initialize the gate-logit weight slice to zero.
- Initialize the gate-logit bias slice to `gate_init`.

Consequently, every contextual gate initially equals the existing static gate:

$$
\operatorname{sigmoid}(g_c)
= \operatorname{sigmoid}(\mathrm{gate\_init}).
$$

The gate slice can then learn data dependence without changing Longshot's
initial update scale. Fusing the two affine maps has the same expressivity and
parameter count as separate delta and gate projections from $z_c$, but uses one
projection call. The current static scalar must remain the default so existing
configurations and checkpoints retain their architecture and state dictionary
exactly.

`gate_mode` should remain independent from gate granularity. Sigmoid is the
recommended mode for contextual gates because it bounds update admission
between zero and one. Direct contextual gating can remain available for
controlled experiments and equivalence tests, but it permits negative and
amplifying updates and should not be the default.

## Relation to gated attention

[Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708)
studies input-dependent multiplicative gates at several positions in ordinary
softmax attention. Its strongest variant applies a query-dependent sigmoid gate
to each attention-head output. Elementwise gating performs slightly better than
one scalar per head, while both outperform the ungated baseline. The paper also
finds that distinct gates between heads matter more than using the finest
possible granularity.

This supports testing both contextual scalar and contextual vector Longshot
gates. In Longshot, each processed cache head plays the role of an independently
modulated stream, and its processed representation supplies the context from
which to predict the gate.

The analogy has limits. Longshot gates a residual cache rewrite rather than an
ordinary attention output, so the paper's explanations involving attention
sinks and the value/output projection pair do not transfer directly. The
relevant architectural evidence is narrower: multiplicative sigmoid gates are
effective when they depend on the current data and can differ between heads.

There is not yet a strong reason to introduce a gate MLP, gate-specific
normalization, sparsity loss, or additional head embeddings. Expanding the
existing affine output projection provides the cleanest first comparison and
adds little computation.

## Linear-state updates

When Qwen3.5 linear-state caching is active, Longshot can also project processed
linear-state heads into updates whose dimensions differ from KV-cache updates.
The current scalar gate broadcasts to both streams, but a KV-sized contextual
vector cannot safely be reused for a linear-state update.

The linear-state adapter should therefore expand its own `output_proj` whenever
its output is contextually gated:

- A contextual scalar produces one value for each processed linear-state head.
- A contextual vector matches the linear-state adapter's output dimension.
- Its delta and gate-logit slices follow the same initialization rule as the KV
  output projection.

The adapter splits its expanded output into its proposed update and gate logits.
This is preferable to slicing, padding, or broadcasting the KV gate because the
two output projections propose different kinds of state update. The existing
`gate_linear_state_delta` control should continue to determine whether the
linear-state update receives this outer Longshot gate.

## Recommended first comparison

The initial experiment should compare only:

1. The existing static scalar gate.
2. A contextual scalar gate per cache head.
3. A contextual vector gate per cache head and output channel.

All three should use the same `gate_init`, gate activation, output-projection
initialization, training setup, and Longshot configuration. This isolates the
effect of data dependence and granularity before considering more elaborate
gate networks or regularization.

## Implementation plan

### Configuration and compatibility

- Add `CacheUpdateGateType` to `gcw.config.types` with the values
  `static_scalar`, `contextual_scalar`, and `contextual_vector`, and export it
  from `gcw.config`.
- Add `CacheProcessorConfig.update_gate_type`, defaulting to `static_scalar`,
  and validate it in `CacheProcessorConfig.__post_init__`.
- Keep `gate_mode` as the independent activation choice. `sigmoid` applies the
  sigmoid in FP32 before casting to the update dtype; `direct` uses the
  projected logits directly.
- In `gcw.architecture.validation._model_issues`, reject contextual gate types
  unless `ModelConfig.variant == "longshot"`. Add the corresponding validation
  code to `_MODEL_CONFIG_VALIDATION_CODES`.
- In `gcw.architecture.contracts._config_payload`, omit `update_gate_type` when
  it is `static_scalar`, preserving the canonical architecture payload for
  existing configurations.
- Regenerate the configuration reference with
  `scripts/docs/generate_reference.py`.

### Expanded output projection

Do not add a separate gate module. For contextual gate types, extend the output
width of the existing `CacheProcessor.out_proj` so it returns the proposed
update followed by its gate logits in one projection call.

Add two shared operations in
`gcw.models.cache_processors.longshot_layers`:

- `_initialize_contextual_output_projection(projection, delta_dim, config, std)`
  handles the different output-axis layouts of `nn.Linear` and
  `PerCacheHeadLinear`. It initializes the delta slice according to
  `out_proj_init`, zeros the gate-logit weight slice, and fills the gate-logit
  bias slice with `gate_init`. The caller supplies its existing initialization
  standard deviation, which defaults to `0.02` in both processor
  `init_weights` methods; it is not a configuration field.
- `_activate_update_gate(gate_logits, gate_mode, output_dtype)` applies sigmoid
  in FP32 or preserves direct logits, then casts to `output_dtype`. Keeping this
  shared prevents the primary and linear-state paths from developing different
  activation semantics.

The primary contextual gate has the following shapes:

- Scalar: `(batch, kv_cache_heads, 1)`.
- Vector with `update_stream: v`: `(batch, kv_cache_heads, head_dim)`.
- Vector with `update_stream: kv`: `(batch, kv_cache_heads, 2 * head_dim)`.

The static mode retains the existing output-projection width,
`CacheProcessor.gate` parameter, and state-dictionary keys. Contextual modes
retain the `out_proj.weight` and `out_proj.bias` names with their expanded
shapes and do not create a separate gate parameter or submodule.

### Longshot processor integration

Update `gcw.models.cache_processors.longshot_processor.CacheProcessor` as
follows:

- In `__init__`, derive `gate_dim` as zero for `static_scalar`, one for
  `contextual_scalar`, or `output_dim` for `contextual_vector`. Construct
  `self.out_proj` with width `output_dim + gate_dim`. Construct `self.gate` only
  for `static_scalar`; contextual modes register no static gate parameter.
- In `init_weights`, preserve the existing static initialization. For a
  contextual type, call `_initialize_contextual_output_projection` after the
  generic linear-module initialization so the gate slice is not overwritten.
- Change `_gate_scale` to accept optional gate logits and the output dtype. It
  uses the static parameter when the logits are absent, otherwise delegates to
  `_activate_update_gate`, and continues to populate `self._gate_value` for
  telemetry.
- Add `_project_primary_update(hidden)`, which calls `self.out_proj` once. In
  static mode it returns the projection and the static scale. In contextual
  modes it splits the last axis into `output_dim` delta channels and `gate_dim`
  gate-logit channels, activates the gate, and returns both tensors.
- In `forward`, assign `primary_hidden = processed[:, :x.shape[1]]`, call
  `_project_primary_update(primary_hidden)`, and multiply the returned delta by
  the returned gate before adding it to the latest cache state.

Flow, processor recurrence, head routing, token selection, and cache writeback
remain unchanged. The update gate is evaluated once from the final processor
output of each Longshot invocation.

### Linear-state integration

Update `gcw.models.cache_processors.longshot_linear_state.LinearStateAdapter`
as follows:

- In `__init__`, derive a linear-state `gate_dim` only when
  `update_gate_type` is contextual and `gate_linear_state_delta` is true. It is
  one for `contextual_scalar` and `self.output_dim` for `contextual_vector`.
  Expand the existing shared `output_proj` to `self.output_dim + gate_dim`.
- Preserve the adapter's current shared projection semantics even when the
  primary KV path uses `PerCacheHeadLinear`. Inputs still produce distinct
  data-dependent gates per linear-state head, without also changing the
  adapter's established parameter sharing.
- Add `project_update(hidden)`, which calls the expanded `output_proj` once,
  splits the delta and gate-logit slices, applies `_activate_update_gate`,
  records the detached gate in `self._gate_value`, and returns the gated delta.
- In `init_weights`, call `_initialize_contextual_output_projection` after the
  adapter's generic linear initialization when `gate_dim` is nonzero.

The linear-state contextual scalar has shape
`(batch, linear_state_heads, 1)`. Its contextual vector matches
`LinearStateAdapter.output_dim`. In
`longshot_processor.CacheProcessor.forward`, call
`linear_adapter.project_update(processed_linear)`. Apply the primary static
scalar afterward only for `static_scalar`; the adapter already owns contextual
linear-state gating.

The existing configuration resolution sets `gate_linear_state_delta` to false
for `linear_state_update_mode: gated_delta`. Do not expand `output_proj` or
produce unused gate logits in that case.

### Parameter, FLOP, and telemetry accounting

- Update `gcw.params._longshot_processor_params` so the Longshot block counts
  one parameter only for `static_scalar`. For contextual modes, increase the
  output-projection width by the gate dimension before applying the existing
  shared or per-head projection multiplicity.
- Update `gcw.params._longshot_linear_state_params` to use the expanded adapter
  output width only when contextual linear-state delta gating is enabled.
- Update `gcw.flops._longshot_processor_block_flops_per_call` to include the
  contextual projection, activation, and multiplication for the KV stream and,
  when enabled, the linear-state stream. Count each update gate once per
  processor invocation, outside processor-layer recurrence.
- Keep `cache_processor/gate` as the mean of the primary runtime gate tensor so
  existing telemetry consumers remain compatible.
- Extend `gcw.performance.telemetry.cache_processor_metrics` with
  `cache_processor/linear_state_gate`, reporting the mean when the adapter's
  contextual gate ran.

### Tests

Extend the configuration and architecture tests to cover:

- The `static_scalar` default, all accepted values, invalid-value rejection,
  and rejection of contextual types for BT, BGT, and Icarus.
- Omission of the default from canonical architecture payloads and inclusion of
  non-default contextual types.

Extend the Longshot model tests to cover:

- Numerical equivalence of static, contextual-scalar, and contextual-vector
  gates immediately after initialization for both `sigmoid` and `direct`.
- Data-dependent variation between examples and cache heads, scalar
  broadcasting across output channels, and independent vector channels.
- Shared and `PerCacheHeadLinear` projections with both KV and value-only
  update streams.
- Gradients to gate weights, gate biases, processor representations, and output
  projections.
- Linear-state scalar and vector shapes, initialization equivalence, gradients,
  and the absence of an outer gate for `gated_delta` updates.

Extend parameter, FLOP, checkpoint, and telemetry tests to cover:

- Exact agreement between estimated and instantiated parameter counts for all
  gate types, projection-sharing modes, update streams, and linear-state
  combinations.
- Contextual gate FLOP increments for primary and linear-state updates.
- Preservation of the legacy `cache_processor.block.gate` key and legacy
  checkpoint round trips for the default architecture.
- Contextual checkpoint round trips with expanded output-projection shapes, no
  additional gate-projection keys, and no static gate parameter.
- Primary and linear-state telemetry matching the mean of the runtime gate
  tensors.

Run Ruff, the focused Longshot, configuration, architecture, parameter, FLOP,
checkpoint, and telemetry tests, followed by the full CPU suite and the
warnings-as-errors documentation build.

### Deferred work

Do not add a gate MLP, extra gate normalization, sparsity loss, head-identity
embedding, or a delta-derived gate source in this implementation. These remain
separate experiments after the scalar-versus-vector comparison.
