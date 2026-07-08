from __future__ import annotations

from gcw.config import CacheProcessorConfig, ModelConfig


PAD_VOCAB_TO = 128


def _padded_vocab(vocab_size: int) -> int:
    return ((vocab_size + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO) * PAD_VOCAB_TO


def _backbone_matmul_params(config: ModelConfig) -> int:
    head_dim = int(config.head_dim)
    q_dim = config.num_heads * head_dim
    kv_dim = config.num_kv_heads * head_dim
    per_layer = (
        config.hidden_size * q_dim
        + config.hidden_size * kv_dim
        + config.hidden_size * kv_dim
        + q_dim * config.hidden_size
        + 3 * config.hidden_size * config.intermediate_size
    )
    lm_head = config.hidden_size * _padded_vocab(config.vocab_size)
    return config.num_layers * per_layer + lm_head


def _backbone_attention_flops_per_token(config: ModelConfig, seq_len: int) -> int:
    return config.num_layers * 4 * config.num_heads * int(config.head_dim) * seq_len


def backbone_flops_per_token(config: ModelConfig, seq_len: int) -> float:
    return float(2 * _backbone_matmul_params(config) + _backbone_attention_flops_per_token(config, seq_len))


def _processor_layer_matmul_params(config: CacheProcessorConfig) -> int:
    head_dim = config.hidden_size // config.num_heads
    q_dim = config.num_heads * head_dim
    kv_dim = config.num_kv_heads * head_dim
    return (
        config.hidden_size * q_dim
        + config.hidden_size * kv_dim
        + config.hidden_size * kv_dim
        + q_dim * config.hidden_size
        + 3 * config.hidden_size * config.intermediate_size
    )


def _processor_block_flops_per_call(input_dim: int, config: CacheProcessorConfig, selected: int) -> float:
    in_out = input_dim * config.hidden_size + config.hidden_size * input_dim
    matmul = 2 * selected * (in_out + config.num_layers * _processor_layer_matmul_params(config))
    head_dim = config.hidden_size // config.num_heads
    attn = config.num_layers * 4 * config.num_heads * head_dim * selected * selected
    return float(matmul + attn)


def _selected_count(valid_tokens: int, process_every: int, topk: int) -> int:
    recent = min(valid_tokens, process_every)
    earlier = max(0, valid_tokens - recent)
    return recent + min(topk, earlier)


def processor_forward_flops_per_sequence(config: ModelConfig, seq_len: int) -> float:
    cp = config.cache_processor
    if config.variant == "tf" or cp is None:
        return 0.0
    per_layer_dim = cp.stream_factor * config.num_kv_heads * int(config.head_dim)
    total = 0.0
    for start in range(cp.process_every, seq_len, cp.process_every):
        selected = _selected_count(start, cp.process_every, cp.topk)
        if config.variant == "bt":
            total += config.num_layers * _processor_block_flops_per_call(per_layer_dim, cp, selected)
        elif config.variant == "bgt":
            total += _processor_block_flops_per_call(config.num_layers * per_layer_dim, cp, selected)
        elif config.variant == "icarus":
            cache_heads = config.num_layers * config.num_kv_heads
            selected_heads = min(max(cp.topk_heads, 0), cache_heads)
            total += _icarus_processor_block_flops_per_call(2 * int(config.head_dim), cp, selected, cache_heads, selected_heads)
    return total


def _icarus_processor_block_flops_per_call(
    input_dim: int,
    config: CacheProcessorConfig,
    selected: int,
    cache_heads: int,
    selected_heads: int,
) -> float:
    in_out = input_dim * config.hidden_size + config.hidden_size * input_dim
    matmul = 2 * selected * cache_heads * (in_out + config.num_layers * _processor_layer_matmul_params(config))
    head_dim = config.hidden_size // config.num_heads
    attn = config.num_layers * 4 * config.num_heads * head_dim * (cache_heads * selected) * (selected_heads * selected)
    return float(matmul + attn)


def flops_per_token(config: ModelConfig, seq_len: int) -> float:
    return backbone_flops_per_token(config, seq_len) + processor_forward_flops_per_sequence(config, seq_len) / seq_len


def training_flops_per_token(config: ModelConfig, seq_len: int | None = None) -> float:
    return 3.0 * flops_per_token(config, seq_len or config.sequence_len)


def total_training_flops(config: ModelConfig, tokens: int, seq_len: int | None = None) -> float:
    return training_flops_per_token(config, seq_len) * tokens
