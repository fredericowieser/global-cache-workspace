from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BottleneckSpec:
    family: str
    source: str
    vocab_size: int
    legacy_vocab_size: int | None
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    norm_eps: float
    tie_embeddings: bool
    qk_norm: bool
    bt_hidden_size: int
    bt_intermediate_size: int
    bgt_hidden_size: int
    bgt_intermediate_size: int
    sft_path: str
    artifact_slug: str
    display_name: str


BOTTLENECK_SPECS: dict[str, BottleneckSpec] = {
    "llama-3.2-1b": BottleneckSpec(
        family="llama",
        source="meta-llama/Llama-3.2-1B",
        vocab_size=128256,
        legacy_vocab_size=None,
        hidden_size=2048,
        intermediate_size=8192,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        head_dim=64,
        rope_theta=500000.0,
        norm_eps=1e-5,
        tie_embeddings=False,
        qk_norm=False,
        bt_hidden_size=512,
        bt_intermediate_size=2240,
        bgt_hidden_size=1536,
        bgt_intermediate_size=6272,
        sft_path="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-1b/final_model",
        artifact_slug="llama32_1b",
        display_name="Llama 3.2 1B",
    ),
    "llama-3.2-3b": BottleneckSpec(
        family="llama",
        source="meta-llama/Llama-3.2-3B",
        vocab_size=128256,
        legacy_vocab_size=None,
        hidden_size=3072,
        intermediate_size=8192,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        head_dim=128,
        rope_theta=500000.0,
        norm_eps=1e-5,
        tie_embeddings=False,
        qk_norm=False,
        bt_hidden_size=512,
        bt_intermediate_size=2240,
        bgt_hidden_size=1344,
        bgt_intermediate_size=5632,
        sft_path="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.2-3b/final_model",
        artifact_slug="llama32_3b",
        display_name="Llama 3.2 3B",
    ),
    "llama-3.1-8b": BottleneckSpec(
        family="llama",
        source="meta-llama/Meta-Llama-3.1-8B",
        vocab_size=128256,
        legacy_vocab_size=None,
        hidden_size=4096,
        intermediate_size=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        rope_theta=500000.0,
        norm_eps=1e-5,
        tie_embeddings=False,
        qk_norm=False,
        bt_hidden_size=512,
        bt_intermediate_size=2240,
        bgt_hidden_size=1408,
        bgt_intermediate_size=4352,
        sft_path="/mnt/jfs-01/shared/bottleneck_llm/sft/llama-3.1-8b/final_model",
        artifact_slug="llama31_8b",
        display_name="Llama 3.1 8B",
    ),
    "qwen3-0.6b": BottleneckSpec(
        family="qwen3",
        source="Qwen/Qwen3-0.6B",
        vocab_size=151936,
        legacy_vocab_size=151669,
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        rope_theta=1000000.0,
        norm_eps=1e-6,
        tie_embeddings=True,
        qk_norm=True,
        bt_hidden_size=512,
        bt_intermediate_size=2240,
        bgt_hidden_size=1344,
        bgt_intermediate_size=5632,
        sft_path="/mnt/jfs-01/shared/bottleneck_llm/sft/qwen3-0.6b/final_model",
        artifact_slug="qwen3_06b",
        display_name="Qwen3 0.6B",
    ),
}


def bottleneck_spec(backbone: str) -> BottleneckSpec:
    key = backbone.lower()
    if key not in BOTTLENECK_SPECS:
        raise ValueError(f"unsupported bottleneck backbone {backbone!r}; expected one of {sorted(BOTTLENECK_SPECS)}")
    return BOTTLENECK_SPECS[key]
