# Research boundaries and proposals

**Status:** Mixed: sections label implemented behavior and proposals

## Diffusion cache processor proposal

This section outlines a prospective research direction; no current GCW variant implements a diffusion objective, noise schedule, or time-conditioned reverse process. Keep the proposal separate from descriptions of executable behavior until code and tests establish those contracts.

### Research question

BGT and Longshot perform deterministic cache rewrites. BGT jointly transforms every selected position through a global layer-concatenated representation. Longshot compresses selected history for each `(layer, KV head)` pair, mixes those cache-head summaries, and writes one frontier position. Both models learn solely through later next-token cross-entropy.

A diffusion cache processor would instead learn a sequence of coarse-to-fine cache refinements. One invocation would contain several shared-weight denoising steps. Each step would know the current noise level, update a small active cache state, and optionally revise which source tokens support that state. Training would combine future-token prediction with reconstruction of deliberately corrupted cache representations.

Noise injection alone does not make a processor a diffusion model. A genuine diffusion-style variant needs:

- a defined clean cache representation;
- a family of corruption levels;
- explicit noise-level conditioning;
- shared denoiser weights across refinement steps;
- a trained reverse schedule at inference.

Without these elements, the model is a recurrent cache processor with noise augmentation. That remains a valid baseline, but it tests a different mechanism.

### Three candidate architectures

| Candidate | Active state | Read pattern | Write pattern | Role |
|---|---|---|---|---|
| Diffusion-BGT | selected raw cache positions | BGT global feature at each selected time | all selected positions | literal and easy diffusion control |
| Diffusion-Longshot | one compressed vector per cache head | Longshot time encoder and cross-head attention | latest selected position | cheapest direct competitor |
| Sticky Diffusion Cache Processor | fixed concept slots routed to cache tokens | cheap global scout followed by sparse refinement | attention-routed residual broadcast | intended scalable research model |

Diffusion-BGT changes the fewest assumptions. It can establish whether scheduled denoising improves BGT before selection and broadcast change. Its cost remains high because the diffusion state contains every selected K/V coordinate across all layers.

Diffusion-Longshot preserves Longshot's time compression. It corrupts the compressed per-cache-head state, applies several time-conditioned cross-head updates, and writes the final prediction at the latest selected position. This design gives the cleanest comparison between one-shot Longshot, repeated deterministic Longshot, and a trained denoising trajectory.

The Sticky Diffusion Cache Processor, abbreviated SDCP below, separates global routing from expensive refinement. A small scout reads the full cache through compressed descriptors. Fixed concept slots gather information from selected positions, undergo iterative denoising, and broadcast their residual through the same routing weights. Fixed slots keep the diffusion state small even when their token support changes.

### Proposed SDCP data flow

```{mermaid}
flowchart TD
    A[full KV cache] --> B[small token descriptors]
    B --> C[global scout queries]
    C --> D[attention routing matrix]
    D --> E[wide candidate selection]
    E --> F[cheap candidate processing]
    F --> G[final sticky selection]
    G --> H[fixed concept slots]
    H --> I[initialise active state]
    I --> J[time-conditioned denoising step]
    J --> K{more refinement steps?}
    K -- yes --> J
    K -- no --> L[residual broadcast through accumulated routing]
    L --> M[small gated cache update]
    M --> N[backbone predicts later tokens]
```

The diagram describes inference and the optional unrolled CE branch. The ordinary denoising branch trains from one randomly sampled noise level without traversing the whole loop. The design follows one operating rule: read globally, refine a small state, and write sparsely. The scout supplies broad context once. The denoiser spends most of its compute on the active concept slots rather than the complete cache.

### Global scout and token descriptors

Full self-attention over every cache position would add quadratic context cost before denoising begins. SDCP should instead construct a narrow descriptor for each token by pooling or projecting:

- values across layers and KV heads;
- optional key magnitude or position-neutral key features;
- backbone salience scores;
- recency and absolute position.

A small set of learned scout queries then cross-attends to all descriptors. With $M$ scouts and $T$ cached positions, routing scales with $MT$ rather than $T^2$. Eight to sixteen scouts provide a plausible first regime. Descriptor width should scale with head dimension and processor budget; widths from 64 to 128 are starting points rather than a capacity claim. Hierarchical layer or head groups offer a larger alternative if flat pooling loses essential structure.

The scout produces an attention matrix $A$ from concept slots to cache positions. Column mass supplies a learned token score. Each row records which tokens support one concept slot. The system may accumulate these scores across scout rounds or denoising steps, but it should treat attention as a routing signal rather than a semantic explanation.

Useful diagnostics include attention entropy, coverage, overlap between scouts, and the fraction of selected positions replaced at each round. If every scout selects the same token, the routing bottleneck has collapsed. A mild diversity or coverage term may help, but experiments should first determine whether collapse occurs.

### Multiple selection rounds and stickiness

Selection should occur more than once inside one processor invocation. Two designs offer different risk profiles.

The recommended first design uses cascaded selection:

1. The global scout chooses a wide candidate pool.
2. A shallow processor reads those candidates.
3. The scout recomputes scores from the updated state.
4. A narrower final set becomes sticky.
5. The denoiser keeps that final set fixed for the remaining steps.

For example, the scout might reduce the full context to 128 candidates and then to 32 active positions. The exact sizes require compute-matched sweeps. Locking the final set prevents membership changes from breaking the denoising trajectory.

A later dynamic design may revise selection after every denoising step. It should retain most previous positions and replace only the weakest fraction. New members must enter at the current noise level; otherwise they have skipped earlier diffusion states. Hysteresis should require a new token to beat an existing sticky token by a margin. A reserved recent-token quota protects the memory frontier.

Fixed concept slots offer a cleaner dynamic alternative. Their attention support can move between tokens while the denoising tensor keeps the same shape. The model changes routing rather than allocating a new diffusion state whenever selection changes.

### Shared selection and broadcast

SDCP should derive selection and broadcast from the same routing matrix. Let $D\in\mathbb R^{T\times d_s}$ contain narrow token descriptors, let $A\in\mathbb R^{M\times T}$ route positions into concept slots, and let $U$ expand a slot residual into the cache-writer width $d_w$. The forward route gathers descriptors. A normalised transpose returns expanded residuals to the contributing positions:

$$
Z=A D,
\qquad
\Delta C=\operatorname{norm}(A^\top)\,U(\Delta Z).
$$

The writer $U$ may use factorised layer-specific or cache-head-specific projections instead of one dense map. The concept slots carry an update, not a compressed replacement for the complete cache; the original cache remains available to the backbone. This constraint gives each update a traceable support: the tokens that created a concept also receive its refined message. It removes a second independent broadcaster that could learn an unrelated route.

The implementation should still gate the final write. A zero-initialised output projection or a small direct gate keeps the initial model near the backbone's unmodified behavior. Broadcasting to every token would erase the intended compute saving and increase the risk of prompt corruption. The first version should write only the final sticky support or a small frontier set.

### What to corrupt

Value-only diffusion is the safest starting point. Values carry retrieved content, while keys control subsequent attention routing. Stored keys also contain RoPE position transforms. Uncalibrated key noise can make later queries attend to unrelated tokens even when values remain coherent.

A staged programme should test:

1. values only;
2. values with key-derived selection features;
3. keys after inverse RoPE;
4. joint K/V diffusion.

The processor must scale noise by layer, cache head, and stream. Raw K and V coordinates do not share one global variance. Per-group RMS normalisation or whitening should define the corruption space. Isotropic Gaussian noise is rotation-invariant, so inverse RoPE is unnecessary for Gaussian corruption alone. It remains useful when the denoiser mixes keys from different positions in a shared content frame or applies structured, non-isotropic corruptions. Those paths must restore RoPE before writing keys.

Corruptions may include Gaussian noise, head dropout, token masking, quantisation error, and low-rank perturbations. Gaussian noise supplies a standard diffusion path. Structured corruptions test whether the processor learns robust cache restoration rather than one narrow noise inversion.

### Training objective

Let $X_0$ denote the clean active cache state after stop-gradient normalisation. Training samples a corruption level $t$ and constructs

$$
X_t=\alpha(t)X_0+\sigma(t)\epsilon.
$$

The denoiser receives $X_t$, the noise level, and clean conditioning from the selected cache descriptors. It may predict $X_0$, the added noise, or a velocity field. The first implementation should choose one parameterisation and keep the solver consistent with it.

The minimal objective follows the meeting proposal:

$$
\mathcal L
=
\mathcal L_{\mathrm{CE}}
+
\lambda_{\mathrm{denoise}}\mathcal L_{\mathrm{denoise}}.
$$

The denoising term anchors the output near valid cache states. Future-token cross-entropy rewards deviations that improve prediction. Cross-entropy must cover tokens after the processor invocation; loss on earlier tokens cannot measure the rewrite's effect.

These terms compete. A large denoising weight recovers the original cache and learns an identity map. A small weight permits useful but potentially unstable states. A practical curriculum should:

1. warm up the processor on cache denoising with a frozen backbone;
2. introduce future-token cross-entropy;
3. reduce the denoising weight while retaining a nonzero cache prior;
4. keep the final write gate small until joint training stabilises.

The clean target must remain fixed within each training example. A frozen backbone provides the simplest target. If the backbone also trains, detach the clean cache or use a slowly updated reference model; otherwise the denoiser chases a moving target.

The first training design should use two explicit branches. The denoising branch samples one random noise level, predicts the clean active state in one network evaluation, and applies only the denoising loss. The CE branch initially broadcasts that one-step predicted-clean state and measures future-token loss. A later rollout phase may unroll the same two to eight reverse steps used at inference and backpropagate CE through that trajectory, using activation checkpointing or truncated gradients if memory requires it. This progression separates cheap denoiser pretraining from exposure to self-generated multi-step states. Report the chosen branch mix and every unrolled processor evaluation in FLOP counts.

### Inference contract

SDCP performs conditional refinement rather than generation from an empty cache. At one processor invocation:

1. The scout selects and routes the current cache state.
2. The processor copies the active state instead of corrupting the stored cache in place.
3. A configured moderate noise level perturbs the copy.
4. Two to eight shared denoising steps reduce the noise.
5. The model converts the final prediction into a gated residual.
6. The routing matrix broadcasts that residual to the allowed write positions.

Starting from complete Gaussian noise would discard the cache information the processor should preserve. A moderate image-to-image-style corruption gives the denoiser room to move while retaining the original representation. Deterministic evaluation should fix the initial noise seed or use a deterministic flow solver. Stochastic evaluation should report sample count and selection strategy.

One reverse trajectory is refinement, not search. Multiple seeds, a stochastic reverse process, or explicit particle selection would be required to claim exploration over alternative cache interpretations.

### Speculative ambiguity experiment

The meeting's duck-rabbit image suggests a speculative test of cache bistability. One cache may support two coherent interpretations. Different noise seeds could lead the scout toward different sticky token groups and produce different valid continuations. Output-space sampling already provides a strong and cheaper ambiguity baseline, so cache perturbation must improve quality or reveal distinct routing behavior to justify itself.

This claim requires multimodal data and measurements of routing, output diversity, and quality. A single greedy trajectory cannot establish it. A useful figure would show two seeds, their evolving sticky supports, and two coherent outcomes for one ambiguous prompt. The figure should report an observed behavior rather than act as evidence for the architecture.

### Required baselines and ablations

The first experiments should isolate mechanisms before combining them:

| Comparison | Question |
|---|---|
| Static bottleneck projection versus iterative processors | Does any iteration justify its added compute? |
| Longshot versus repeated shared-weight Longshot | Does extra recurrent compute help without diffusion? |
| Repeated Longshot versus Diffusion-Longshot | Does noise-level training improve matched-step refinement? |
| Fixed selection versus cascaded selection | Does reselection identify better support? |
| Cascaded selection versus dynamic sticky selection | Does selection during refinement justify its instability and cost? |
| Independent broadcast versus tied routing | Does one attention route improve accuracy or consistency? |
| Value-only versus unroped K/V | Do key updates add benefit beyond their routing risk? |
| Denoising only, CE only, and joint loss | Which term supplies robustness and which supplies prediction gain? |

Every comparison must match processor parameters, number of processor evaluations, and as closely as possible total FLOPs. Wall-clock latency and peak memory remain first-class outcomes because iterative refinement can lose any accuracy gain through repeated cache reads.

Evidence for a useful diffusion mechanism would include:

- recovery from held-out cache corruptions;
- lower future-token loss than BGT and Longshot at matched compute;
- improvement as reverse steps increase within the trained range;
- better results than a deterministic recurrent refiner with the same evaluations;
- stable routing with bounded selection churn;
- preserved performance on ordinary prompts that do not benefit from rewriting.

If deterministic recurrence matches diffusion, GCW should keep the recurrent model. If additional denoising steps do not improve results, the model has not learned coarse-to-fine refinement. If shared routing collapses to one token or churns at every step, selection needs repair before broader evaluation.

### Recommended first complete configuration

The strongest first SDCP configuration uses:

- value-only cache descriptors and updates;
- eight to sixteen scout slots;
- one global cross-attention scout pass;
- a wide candidate selection followed by one narrower reselection;
- a fixed final support during four denoising steps;
- one shared time-conditioned denoiser;
- attention scores accumulated across refinement;
- a normalised transpose route for sparse residual broadcast;
- a zero-initialised or very small output gate;
- future-token CE plus cache denoising loss.

This design differs cleanly from BGT and Longshot. It does not rewrite every selected global vector like BGT, and it does not compress once and make one frontier update like Longshot. It introduces a trained reverse trajectory while keeping global routing cheaper than full cache processing.

### Relation to latent reasoning

The initial SDCP experiment tests cache robustness and iterative refinement. It does not train a chain-of-thought replacement. Reconstructing a corrupted cache teaches restoration; future-token CE supplies any task-useful change.

A later latent-reasoning project would need a different target. One defensible route would keep the prompt cache immutable, compress verified reasoning traces into student-native auxiliary slots, and train a cache-conditioned flow to generate those slots. Raw KVs from independently fine-tuned checkpoints are not directly interchangeable because their query, key, value, and output projections define different coordinate systems. That extension should remain separate until the simpler diffusion processor beats deterministic cache refinement.

## JEPA from first principles

### The central idea

A Joint-Embedding Predictive Architecture learns to predict one representation from another instead of reconstructing every input detail. A memorable distinction is:

> A generative model tries to repaint the missing landscape. A JEPA tries to predict the missing region on a useful map.

Let $x$ denote observed context, $y$ a target, and $z$ information about the prediction task such as a mask, relative position, transformation, or latent code. A broad joint-embedding formulation computes

$$
s_x=f_\theta(x),\qquad
s_y=f_{\bar\theta}(y),\qquad
\widehat s_y=g_\phi(s_x,z),
$$

and minimizes an energy or distance in representation space:

$$
\mathcal L_{\mathrm{JEPA}}
=D\!\left(s_y,\widehat s_y\right).
$$

The encoder decides what information the representation should retain. The predictor models what remains predictable about the target from context. Because the target lives in feature space, the system can ignore pixel texture, exact wording, or other details that do not help the learned semantics. I-JEPA and V-JEPA apply stop-gradient to $s_y$ and update the target encoder $f_{\bar\theta}$ by EMA rather than direct backpropagation; other JEPA formulations need not use that exact optimization scheme.

### A five-step view

1. Split one observation into context and target views. In an image, these can be visible and masked blocks. In video, they can be visible and masked spatiotemporal regions.
2. Encode context and target with related encoders. The target path supplies a learning target, not a decoded output.
3. Tell the predictor where or what to predict through mask or positional information.
4. Predict the target embedding from the context embedding.
5. Reduce representation error and update the online encoder and predictor. In I-JEPA and V-JEPA, an exponential moving average of the online encoder updates the target encoder.

### Why representation prediction can help

Input-space prediction spends capacity on all observable uncertainty. If many visually plausible textures fit a hidden image block, a pixel decoder must model that multimodality. A representation target can preserve category, geometry, motion, or interaction while discarding incidental texture. The system concentrates its budget on features the encoder judges stable and useful.

This is not free semantic knowledge. The masking policy, encoder architecture, predictor capacity, training data, and anti-collapse mechanism determine what the representation learns. A weak target encoder or an easy mask can produce a useless shortcut.

### Collapse and asymmetric targets

Naive self-prediction admits a trivial solution: both encoders output the same constant vector for every input. The predictor then achieves zero distance without learning structure. This is representation collapse.

I-JEPA and V-JEPA counter it with an asymmetric learning path:

- gradients update the context encoder and predictor;
- the target representation receives stop-gradient;
- an exponential moving average of the context encoder updates the target encoder;
- the predictor must use visible context plus target-location information to match nontrivial target features.

The moving target changes slowly enough to stabilize learning but follows the online encoder over time. Mask design also matters. I-JEPA uses one informative context block and multiple sufficiently large target blocks; V-JEPA extends the idea to masked spatiotemporal regions.

### I-JEPA

[I-JEPA](https://arxiv.org/abs/2301.08243) applies joint-embedding prediction to images. A context encoder sees visible image patches. A predictor combines those context features with positional mask tokens and predicts the target encoder's patch representations for several masked target blocks. Training uses average patch-level $L_2$ distance. It does not reconstruct RGB pixels.

Large target blocks encourage semantic prediction. A spatially distributed context supplies enough information to make that prediction possible. If targets are tiny, local texture shortcuts dominate; if context is too narrow, the task becomes underdetermined.

### V-JEPA

[V-JEPA](https://arxiv.org/abs/2404.08471) extends feature prediction to video. Multi-block masks remove spatiotemporal tube-like regions. The context encoder processes visible tokens, and the predictor estimates target-encoder features for masked regions. The objective favors motion and scene structure without requiring a pixel decoder.

### The broader energy view

The JEPA proposal in [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) frames compatibility through an energy

$$
E_w(x,y,z)=D\!\left(f_y(y),g(f_x(x),z)\right),
$$

possibly minimizing over a latent variable $z$ when the future has several compatible outcomes. Low energy marks compatible context-target pairs; high energy marks incompatible pairs. This view generalizes beyond image masking, but an implementation still needs a concrete target, predictor, collapse defense, and sampling scheme.

## Why GCW is not a JEPA

GCW and JEPA both operate on learned representations, but that resemblance stops before the objective. The current repository has no target encoder, EMA teacher, masked representation target, JEPA predictor loss, or energy-minimization training path.

| Question | GCW | I-JEPA / V-JEPA |
|---|---|---|
| What receives prediction loss? | next token identity | target-region representation |
| Main factorization | left-to-right causal likelihood | masked context-to-target compatibility |
| Target network | none | stop-gradient EMA encoder |
| Decoder | vocabulary projection generates tokens | predictor estimates features, no pixel decoder required |
| Internal intervention | rewrite selected KV-cache states | encode visible context and predict hidden embeddings |
| Collapse defense | token labels prevent constant-output optimum | asymmetric EMA target plus masking/predictor design |
| Inference purpose | language generation and likelihood | learned representation, often followed by downstream probes |

A cache processor can learn abstract internal transformations, but next-token cross-entropy supervises those transformations. Calling it a JEPA would erase the most important scientific distinction.

To build a genuine JEPA experiment here, one would need at least:

1. a defined context-target split over tokens, spans, cache states, or multimodal observations;
2. online and target encoders with an explicit stop-gradient and EMA update contract;
3. a predictor conditioned on target location or transformation;
4. a representation-space loss and anti-collapse validation;
5. controls against the existing autoregressive objective, including equal compute and parameter budgets.

A hybrid could predict future cache representations while also optimizing token loss. That would still require careful ablations because the language-model target might dominate the representation objective. No such hybrid currently exists in this tree.

## Primary references

The JEPA account above relies on primary papers rather than analogy alone:

- Yann LeCun, [*A Path Towards Autonomous Machine Intelligence*](https://openreview.net/pdf?id=BZ5a1r-kVsf), 2022. This position paper introduces the broader energy-based JEPA framework and latent-variable view.
- Mahmoud Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243), CVPR 2023. This paper specifies I-JEPA's context blocks, target blocks, predictor, EMA target encoder, and feature-space loss.
- Adrien Bardes et al., [*V-JEPA: Latent Video Prediction for Visual Representation Learning*](https://arxiv.org/abs/2404.08471), 2024. This paper extends masked feature prediction to spatiotemporal video regions.

The diffusion-cache proposal draws on primary work that defines denoising, generalised corruption, continuous flows, and latent cache augmentation:

- Jonathan Ho, Ajay Jain, and Pieter Abbeel, [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), NeurIPS 2020. This paper establishes the standard noise-and-reverse-process formulation.
- Yaron Lipman et al., [*Flow Matching for Generative Modeling*](https://arxiv.org/abs/2210.02747), 2022. Flow matching trains a continuous vector field along fixed probability paths and offers a simpler alternative to a discrete DDPM reverse process.
- Arpit Bansal et al., [*Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise*](https://openreview.net/forum?id=XH3ArccntI), NeurIPS 2023. This paper shows that iterative restoration can invert structured degradations such as masking and blur, motivating non-Gaussian cache corruptions.
- Luyang Liu et al., [*Deliberation in Latent Space via Differentiable Cache Augmentation*](https://openreview.net/forum?id=IaUJl5RCOu), ICML 2025. A trainable coprocessor reads a frozen language model's KV cache and adds latent representations that improve later decoding.
- Haoqiang Kang et al., [*LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning*](https://arxiv.org/abs/2510.04573), ICLR 2026. LaDiR diffuses compressed reasoning blocks rather than raw cache tensors and separates iterative latent refinement from final autoregressive answering.

For repository behavior, source code and tests remain the primary evidence. In particular, `src/gcw/models/backbones/transformer.py`, `src/gcw/training/core.py`, `src/gcw/training/checkpoint.py`, and `tests/test_models.py` define the contracts most likely to invalidate a high-level description when they change.
