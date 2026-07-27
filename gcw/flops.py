from __future__ import annotations

from gcw.config import CacheProcessorConfig, ModelConfig


PAD_VOCAB_TO = 128


def _padded_vocab(vocab_size: int) -> int:
    return ((vocab_size + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO) * PAD_VOCAB_TO


def _head_dim(config: ModelConfig) -> int:
    if config.head_dim is None:
        raise ValueError("model head_dim must be resolved before FLOP accounting")
    return config.head_dim


def _backbone_matmul_params(config: ModelConfig) -> int:
    if config.family == "qwen3_5_moe":
        return _qwen35_moe_backbone_matmul_params(config)
    head_dim = _head_dim(config)
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
    return config.cache_num_layers * 4 * config.num_heads * _head_dim(config) * seq_len


def _qwen35_moe_backbone_matmul_params(config: ModelConfig) -> int:
    head_dim = _head_dim(config)
    full_q_dim = config.num_heads * head_dim
    full_kv_dim = config.num_kv_heads * head_dim
    linear_key_dim = int(config.linear_num_key_heads or config.num_kv_heads) * int(config.linear_key_head_dim or head_dim)
    linear_value_dim = int(config.linear_num_value_heads or config.num_heads) * int(config.linear_value_head_dim or head_dim)
    linear_conv_dim = 2 * linear_key_dim + linear_value_dim
    moe_hidden = int(config.moe_intermediate_size or config.intermediate_size)
    shared_hidden = int(config.shared_expert_intermediate_size or moe_hidden)
    total = config.hidden_size * _padded_vocab(config.vocab_size)
    for layer_type in config.layer_types or []:
        if layer_type == "full_attention":
            total += config.hidden_size * full_q_dim * 2
            total += 2 * config.hidden_size * full_kv_dim
            total += full_q_dim * config.hidden_size
        else:
            total += config.hidden_size * linear_conv_dim
            total += config.hidden_size * linear_value_dim
            total += 2 * config.hidden_size * int(config.linear_num_value_heads or config.num_heads)
            total += linear_conv_dim * config.linear_conv_kernel_dim
            total += linear_value_dim * config.hidden_size
        total += config.num_experts_per_tok * (2 * moe_hidden * config.hidden_size + config.hidden_size * moe_hidden)
        total += config.hidden_size * config.num_experts
        total += 3 * config.hidden_size * shared_hidden
        total += config.hidden_size
    return total


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


def _processor_block_flops_per_call(
    input_dim: int,
    config: CacheProcessorConfig,
    selected: int,
    *,
    field_evaluations: int = 1,
) -> float:
    in_out = 2 * selected * (input_dim * config.hidden_size + config.hidden_size * input_dim)
    layer_matmul = 2 * selected * config.num_layers * _processor_layer_matmul_params(config)
    head_dim = config.hidden_size // config.num_heads
    attention = config.num_layers * 4 * config.num_heads * head_dim * selected * selected
    return float(in_out + field_evaluations * (layer_matmul + attention))


def _selected_count(valid_tokens: int, process_every: int, topk: int) -> int:
    recent = min(valid_tokens, process_every)
    earlier = max(0, valid_tokens - recent)
    return recent + min(topk, earlier)


def processor_forward_flops_per_sequence(config: ModelConfig, seq_len: int) -> float:
    cp = config.cache_processor
    if config.variant == "tf" or cp is None:
        return 0.0
    head_dim = _head_dim(config)
    cache_layers = config.cache_num_layers
    cache_heads = config.cache_num_heads
    per_layer_dim = cp.stream_factor * config.num_kv_heads * head_dim
    total = 0.0
    field_evaluations = config.flow_processor.num_steps if config.flow_processor is not None else 1
    for start in range(cp.process_every, seq_len, cp.process_every):
        selected = _selected_count(start, cp.process_every, cp.topk)
        if config.variant == "bt":
            total += cache_layers * _processor_block_flops_per_call(per_layer_dim, cp, selected)
        elif config.variant == "bgt":
            total += _processor_block_flops_per_call(
                cache_layers * per_layer_dim,
                cp,
                selected,
                field_evaluations=field_evaluations,
            )
        elif config.variant == "icarus":
            selected_heads = cache_heads if cp.topk_heads <= 0 else min(cp.topk_heads, cache_heads)
            if cp.include_self_head_kv and selected_heads < cache_heads:
                selected_heads += 1
            total += _icarus_processor_block_flops_per_call(
                2 * head_dim,
                cp,
                selected,
                cache_heads,
                selected_heads,
                field_evaluations=field_evaluations,
            )
        elif config.variant == "longshot":
            selected_heads = cache_heads if cp.topk_heads <= 0 else min(cp.topk_heads, cache_heads)
            if cp.include_self_head_kv and selected_heads < cache_heads:
                selected_heads += 1
            total += _longshot_processor_block_flops_per_call(
                2 * head_dim,
                cp,
                selected,
                cache_heads,
                selected_heads,
                field_evaluations=field_evaluations,
            )
    return total


def _icarus_processor_block_flops_per_call(
    input_dim: int,
    config: CacheProcessorConfig,
    selected: int,
    cache_heads: int,
    selected_heads: int,
    *,
    field_evaluations: int = 1,
) -> float:
    output_dim = input_dim // 2 if config.update_stream == "v" else input_dim
    in_out = 2 * selected * cache_heads * (input_dim * config.hidden_size + config.hidden_size * output_dim)
    layer_matmul = 2 * selected * cache_heads * config.num_layers * _processor_layer_matmul_params(config)
    head_dim = config.hidden_size // config.num_heads
    attention = config.num_layers * 4 * config.num_heads * head_dim * (cache_heads * selected) * (selected_heads * selected)
    return float(in_out + field_evaluations * (layer_matmul + attention))


def _longshot_processor_block_flops_per_call(
    input_dim: int,
    config: CacheProcessorConfig,
    selected: int,
    cache_heads: int,
    selected_heads: int,
    *,
    field_evaluations: int = 1,
) -> float:
    encoder_len = config.encoder_seq_len or config.process_every + max(config.topk, 0)
    encoder_in = input_dim if config.encoder_method == "query_attention" else input_dim * encoder_len
    output_dim = input_dim // 2 if config.update_stream == "v" else input_dim
    encode_extra = 0
    if config.encoder_method == "query_attention":
        encode_extra = 4 * cache_heads * selected * input_dim
        if config.encoder_query_source == "final_token":
            encode_extra += 2 * cache_heads * input_dim * input_dim
    encode = 2 * cache_heads * encoder_in * config.hidden_size + encode_extra
    decode = 2 * cache_heads * config.hidden_size * output_dim
    layer_matmul = 2 * cache_heads * config.num_layers * _processor_layer_matmul_params(config)
    head_dim = config.hidden_size // config.num_heads
    attention = config.num_layers * 4 * config.num_heads * head_dim * cache_heads * selected_heads
    field = layer_matmul + attention
    if config.iteration_mode == "external":
        return float(config.iterations * (encode + field_evaluations * field + decode))
    return float(encode + config.iterations * field_evaluations * field + decode)


def flops_per_token(config: ModelConfig, seq_len: int) -> float:
    return backbone_flops_per_token(config, seq_len) + processor_forward_flops_per_sequence(config, seq_len) / seq_len


def training_flops_per_token(config: ModelConfig, seq_len: int | None = None) -> float:
    return 3.0 * flops_per_token(config, seq_len or config.sequence_len)


def total_training_flops(config: ModelConfig, tokens: int, seq_len: int | None = None) -> float:
    return training_flops_per_token(config, seq_len) * tokens
