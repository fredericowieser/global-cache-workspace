# Transformer and cache foundations

**Status:** Maintained

## Objective

For tokens $x_1,\ldots,x_T$, the model factorizes the sequence probability from left to right:

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

Training minimizes negative log-likelihood:

$$
\mathcal L_{\mathrm{AR}}(\theta)
=-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

In pretraining, every valid next token contributes. In supervised fine-tuning, the loader assigns `-100` to prompt and padding labels, so PyTorch ignores them and only answer tokens contribute. The model remains causal in both cases.

## Transformer layer

Let $X\in\mathbb R^{B\times T\times d}$. Each pre-norm decoder block computes

$$
U = X + \operatorname{Attention}(\operatorname{RMSNorm}(X)),
\qquad
X' = U + \operatorname{SwiGLU}(\operatorname{RMSNorm}(U)).
$$

Attention projects queries, keys, and values, applies rotary position embeddings to queries and keys, enforces a causal mask, and uses scaled dot-product attention. Grouped-query attention permits $H_q>H_{kv}$: several query heads share one key-value head. The implementation prefers PyTorch's FlashAttention-compatible kernel and falls back when the shape or device cannot use it.

The model pads the vocabulary dimension to a multiple of 128 for efficient matrix shapes, then slices logits back to the true vocabulary size. It may tie token embeddings to the output projection. Chunked output-loss computation limits peak memory without changing the mathematical loss.

## Cache semantics

At layer $\ell$, autoregressive decoding stores

$$
K^{(\ell)},V^{(\ell)}\in
\mathbb R^{B\times H_{kv}\times T_{cache}\times d_h}.
$$

The reference cache grows Python lists of tensors and favors clarity. The static cache preallocates the maximum storage and writes in place, avoiding repeated concatenation during fast generation. Both represent the same causal history and carry per-token selection scores.

Generation uses greedy decoding. It supports token stop sequences, prompt truncation, and both reference and fast cache paths. Fast/reference parity tests protect their semantic equivalence.
