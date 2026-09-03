from __future__ import annotations

from gcw.config import (
    CacheProcessorConfig,
    ModelConfig,
    linear_attention_following_cache_indices,
)

PAD_VOCAB_TO = 128


def _padded_vocab(vocab_size: int) -> int:
    return ((vocab_size + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO) * PAD_VOCAB_TO


def _head_dim(config: ModelConfig) -> int:
    if config.head_dim is None:
        raise ValueError(
            "model head_dim must be resolved before FLOP accounting"
        )
    return config.head_dim


def _backbone_matmul_params(config: ModelConfig) -> int:
    if config.family == "qwen3_5":
        return _qwen35_backbone_matmul_params(config)
    if config.family == "qwen3_5_moe":
        return _qwen35_moe_backbone_matmul_params(config)
    if config.family == "gemma4":
        return _gemma4_backbone_matmul_params(config)
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


def _backbone_attention_flops_per_token(
    config: ModelConfig, seq_len: int
) -> int:
    if config.family == "gemma4":
        if config.layer_types is None:
            raise ValueError("Gemma4 FLOP accounting requires layer_types")
        total = 0
        for i in range(config.num_layers):
            window = (
                min(seq_len, int(config.sliding_window or seq_len))
                if config.layer_types[i] == "sliding_attention"
                else seq_len
            )
            total += (
                4 * config.num_heads * config.head_dim_for_layer(i) * window
            )
        return total
    if config.family in {"qwen3_5", "qwen3_5_moe"}:
        if config.layer_types is None:
            raise ValueError(
                f"{config.family} requires layer_types for attention FLOPs"
            )
        attention_layers = sum(
            layer_type == "full_attention" for layer_type in config.layer_types
        )
    else:
        attention_layers = config.num_layers
    return attention_layers * 4 * config.num_heads * _head_dim(config) * seq_len


def _qwen35_attention_matmul_params(
    config: ModelConfig, layer_type: str
) -> int:
    head_dim = _head_dim(config)
    full_q_dim = config.num_heads * head_dim
    full_kv_dim = config.num_kv_heads * head_dim
    linear_key_dim = int(
        config.linear_num_key_heads or config.num_kv_heads
    ) * int(config.linear_key_head_dim or head_dim)
    linear_value_dim = int(
        config.linear_num_value_heads or config.num_heads
    ) * int(config.linear_value_head_dim or head_dim)
    linear_conv_dim = 2 * linear_key_dim + linear_value_dim
    if layer_type == "full_attention":
        return (
            config.hidden_size * full_q_dim * 2
            + 2 * config.hidden_size * full_kv_dim
            + full_q_dim * config.hidden_size
        )
    return (
        config.hidden_size * linear_conv_dim
        + config.hidden_size * linear_value_dim
        + 2
        * config.hidden_size
        * int(config.linear_num_value_heads or config.num_heads)
        + linear_conv_dim * config.linear_conv_kernel_dim
        + linear_value_dim * config.hidden_size
    )


def _qwen35_backbone_matmul_params(config: ModelConfig) -> int:
    total = config.hidden_size * _padded_vocab(config.vocab_size)
    for layer_type in config.layer_types or []:
        total += _qwen35_attention_matmul_params(config, layer_type)
        total += 3 * config.hidden_size * config.intermediate_size
    return total


def _qwen35_moe_backbone_matmul_params(config: ModelConfig) -> int:
    moe_hidden = int(config.moe_intermediate_size or config.intermediate_size)
    shared_hidden = int(config.shared_expert_intermediate_size or moe_hidden)
    total = config.hidden_size * _padded_vocab(config.vocab_size)
    for layer_type in config.layer_types or []:
        total += _qwen35_attention_matmul_params(config, layer_type)
        total += config.num_experts_per_tok * (
            2 * moe_hidden * config.hidden_size
            + config.hidden_size * moe_hidden
        )
        total += config.hidden_size * config.num_experts
        total += 3 * config.hidden_size * shared_hidden
        total += config.hidden_size
    return total


def _gemma4_attention_matmul_params(config: ModelConfig, layer_idx: int) -> int:
    if config.layer_types is None:
        raise ValueError("Gemma4 FLOP accounting requires layer_types")
    head_dim = config.head_dim_for_layer(layer_idx)
    kv_heads = config.kv_heads_for_layer(layer_idx)
    q_dim = config.num_heads * head_dim
    kv_dim = kv_heads * head_dim
    total = config.hidden_size * q_dim + q_dim * config.hidden_size
    first_shared = config.num_layers - config.num_kv_shared_layers
    if layer_idx < first_shared:
        total += config.hidden_size * kv_dim
        if not (
            config.attention_k_eq_v
            and config.layer_types[layer_idx] == "full_attention"
        ):
            total += config.hidden_size * kv_dim
    return total


def _gemma4_backbone_matmul_params(config: ModelConfig) -> int:
    total = config.hidden_size * _padded_vocab(config.vocab_size)
    ple = int(config.hidden_size_per_layer_input)
    if ple:
        total += config.hidden_size * config.num_layers * ple
    first_shared = config.num_layers - config.num_kv_shared_layers
    for i in range(config.num_layers):
        total += _gemma4_attention_matmul_params(config, i)
        intermediate = config.intermediate_size * (
            2 if config.use_double_wide_mlp and i >= first_shared > 0 else 1
        )
        total += 3 * config.hidden_size * intermediate
        if ple:
            total += config.hidden_size * ple + ple * config.hidden_size
    return total


def backbone_flops_per_token(config: ModelConfig, seq_len: int) -> float:
    return float(
        2 * _backbone_matmul_params(config)
        + _backbone_attention_flops_per_token(config, seq_len)
    )


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
    in_out = (
        2
        * selected
        * (input_dim * config.hidden_size + config.hidden_size * input_dim)
    )
    layer_matmul = (
        2
        * selected
        * config.num_layers
        * _processor_layer_matmul_params(config)
    )
    head_dim = config.hidden_size // config.num_heads
    attention = (
        config.num_layers
        * 4
        * config.num_heads
        * head_dim
        * selected
        * selected
    )
    return float(in_out + field_evaluations * (layer_matmul + attention))


def _selected_count(valid_tokens: int, process_every: int, topk: int) -> int:
    recent = min(valid_tokens, process_every)
    earlier = max(0, valid_tokens - recent)
    return recent + min(topk, earlier)


def processor_forward_flops_per_sequence(
    config: ModelConfig, seq_len: int
) -> float:
    cp = config.cache_processor
    if config.variant == "tf" or cp is None:
        return 0.0
    head_dim = _head_dim(config)
    cache_layers = config.cache_num_layers
    cache_heads = config.cache_num_heads
    per_layer_dim = cp.stream_factor * config.num_kv_heads * head_dim
    total = 0.0
    if config.diffusion_processor is not None:
        field_evaluations = config.diffusion_processor.num_steps
    else:
        field_evaluations = (
            config.flow_processor.num_steps
            if config.flow_processor is not None
            else 1
        )
    for start in range(cp.process_every, seq_len, cp.process_every):
        selected = _selected_count(start, cp.process_every, cp.topk)
        if config.variant == "bt":
            total += cache_layers * _processor_block_flops_per_call(
                per_layer_dim, cp, selected
            )
        elif config.variant == "bgt":
            total += _processor_block_flops_per_call(
                cache_layers * per_layer_dim,
                cp,
                selected,
                field_evaluations=field_evaluations,
            )
        elif config.variant == "icarus":
            selected_heads = (
                cache_heads
                if cp.topk_heads <= 0
                else min(cp.topk_heads, cache_heads)
            )
            if cp.include_self_head_kv:
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
            linear_heads = _longshot_linear_state_heads(config)
            if cp.head_selection_method == "learned_contextual":
                head_budget = cp.topk_heads + (
                    _longshot_topk_linear_heads(cp) if linear_heads > 0 else 0
                )
                selected_heads = min(head_budget, cache_heads + linear_heads)
            else:
                selected_heads = min(cp.topk_heads, cache_heads) + min(
                    _longshot_topk_linear_heads(cp),
                    linear_heads,
                )
            if cp.include_self_head_kv:
                selected_heads += 1
            total += _longshot_processor_block_flops_per_call(
                2 * head_dim,
                cp,
                selected,
                cache_heads,
                selected_heads,
                field_evaluations=field_evaluations,
                model_config=config,
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
    in_out = (
        2
        * selected
        * cache_heads
        * (input_dim * config.hidden_size + config.hidden_size * output_dim)
    )
    layer_matmul = (
        2
        * selected
        * cache_heads
        * config.num_layers
        * _processor_layer_matmul_params(config)
    )
    head_dim = config.hidden_size // config.num_heads
    attention = (
        config.num_layers
        * 4
        * config.num_heads
        * head_dim
        * (cache_heads * selected)
        * (selected_heads * selected)
    )
    return float(in_out + field_evaluations * (layer_matmul + attention))


def _longshot_processor_block_flops_per_call(
    input_dim: int,
    config: CacheProcessorConfig,
    selected: int,
    cache_heads: int,
    selected_heads: int,
    *,
    field_evaluations: int = 1,
    model_config: ModelConfig | None = None,
) -> float:
    linear_heads = (
        0
        if model_config is None
        else _longshot_linear_state_heads(model_config)
    )
    total_heads = cache_heads + linear_heads
    encoder_len = config.encoder_seq_len or config.process_every + max(
        config.topk, 0
    )
    encoder_in = (
        input_dim
        if config.encoder_method == "query_attention"
        else input_dim * encoder_len
    )
    output_dim = input_dim // 2 if config.update_stream == "v" else input_dim
    kv_in_out = (
        encoder_in * config.hidden_size + config.hidden_size * output_dim
    )
    encode_extra = 0
    if config.encoder_method == "query_attention":
        encode_extra = 4 * cache_heads * selected * input_dim
        if config.encoder_query_source == "final_token":
            encode_extra += 2 * cache_heads * input_dim * input_dim
    linear_in_out = 0
    if model_config is not None and linear_heads > 0:
        layer_pairs = linear_attention_following_cache_indices(model_config)
        linear_key_dim = int(
            model_config.linear_key_head_dim or _head_dim(model_config)
        )
        linear_value_dim = int(
            model_config.linear_value_head_dim or linear_key_dim
        )
        linear_value_heads = int(
            model_config.linear_num_value_heads or model_config.num_heads
        )
        linear_layer_count = linear_heads // linear_value_heads
        context_dim = config.hidden_size
        if config.linear_state_query_mode == "per_linear_state":
            query_count = linear_layer_count
            query_out = linear_value_heads * linear_key_dim
        elif config.linear_state_query_mode in {"shared", "global"}:
            query_count = len(
                {following_cache_idx for _, following_cache_idx in layer_pairs}
            )
            query_out = linear_key_dim
        else:
            raise ValueError(
                f"unsupported cache_processor.linear_state_query_mode: {config.linear_state_query_mode}"
            )
        feature_dim = linear_key_dim + linear_value_dim
        linear_output_dim = feature_dim + (
            2 if config.linear_state_update_mode == "gated_delta" else 0
        )
        linear_in_out = linear_heads * (
            feature_dim * config.hidden_size
            + config.hidden_size * linear_output_dim
        )
        encode_extra += 2 * query_count * context_dim * query_out
        encode_extra += 4 * linear_heads * linear_key_dim * linear_value_dim
    encode_decode = 2 * (cache_heads * kv_in_out + linear_in_out) + encode_extra
    layer_matmul = (
        2
        * total_heads
        * config.num_layers
        * _processor_layer_matmul_params(config)
    )
    head_dim = config.hidden_size // config.num_heads
    attention = (
        config.num_layers
        * 4
        * config.num_heads
        * head_dim
        * total_heads
        * selected_heads
    )
    field = layer_matmul + attention
    initial_selection = _longshot_initial_head_selection_flops(
        input_dim,
        config,
        selected,
        cache_heads,
        linear_heads,
        model_config,
    )
    if config.iteration_mode == "external":
        return float(
            config.iterations
            * (encode_decode + field_evaluations * field + initial_selection)
        )
    reselection = 0
    if model_config is None or model_config.flow_processor is None:
        reselection = (config.iterations - 1) * (
            _longshot_iteration_head_selection_flops(
                config, total_heads, selected_heads
            )
        )
    return float(
        encode_decode
        + initial_selection
        + config.iterations * field_evaluations * field
        + reselection
    )


def _longshot_contextual_head_selection_flops(
    config: CacheProcessorConfig, heads: int
) -> int:
    hidden = config.hidden_size
    selector_hidden = config.head_selector_hidden_size
    norm_and_context = 4 * heads * hidden
    projections = 2 * (heads + 1) * hidden * selector_hidden
    dot_products = 2 * heads * selector_hidden
    identity_bias = heads if config.head_selector_use_identity_bias else 0
    return norm_and_context + projections + dot_products + identity_bias


def _longshot_initial_head_selection_flops(
    input_dim: int,
    config: CacheProcessorConfig,
    selected_positions: int,
    cache_heads: int,
    linear_heads: int,
    model_config: ModelConfig | None,
) -> int:
    if config.head_selection_method == "learned_contextual":
        return _longshot_contextual_head_selection_flops(
            config, cache_heads + linear_heads
        )
    if config.score == "kv_norm":
        kv_score = cache_heads * selected_positions
    else:
        kv_score = 2 * cache_heads * selected_positions * input_dim
    linear_score = 0
    if model_config is not None and linear_heads > 0:
        linear_value_dim = int(
            model_config.linear_value_head_dim
            or model_config.linear_key_head_dim
            or _head_dim(model_config)
        )
        linear_score = 2 * linear_heads * linear_value_dim
    return kv_score + linear_score


def _longshot_iteration_head_selection_flops(
    config: CacheProcessorConfig,
    heads: int,
    selected_heads: int,
) -> int:
    method = config.internal_iteration_score
    if method == "learned_contextual":
        return _longshot_contextual_head_selection_flops(config, heads)
    if method == "attn_l2":
        return 2 * config.num_layers * heads * config.hidden_size
    if method == "kv_norm":
        kv_dim = config.num_kv_heads * config.hidden_size // config.num_heads
        return 4 * config.num_layers * heads * kv_dim
    if method == "cache_head_attention":
        full_score_attention = (
            2
            * config.num_layers
            * config.hidden_size
            * heads
            * max(heads - selected_heads, 0)
        )
        score_reduction = config.num_layers * config.num_heads * heads * heads
        return full_score_attention + score_reduction
    raise ValueError(f"unsupported internal_iteration_score: {method}")


def _longshot_linear_state_heads(config: ModelConfig) -> int:
    cp = config.cache_processor
    if cp is None or not cp.linear_state_cache:
        return 0
    return len(linear_attention_following_cache_indices(config)) * int(
        config.linear_num_value_heads or config.num_heads
    )


def _longshot_topk_linear_heads(config: CacheProcessorConfig) -> int:
    return (
        config.topk_heads
        if config.topk_linear_heads is None
        else config.topk_linear_heads
    )


def flops_per_token(config: ModelConfig, seq_len: int) -> float:
    return (
        backbone_flops_per_token(config, seq_len)
        + processor_forward_flops_per_sequence(config, seq_len) / seq_len
    )


def training_flops_per_token(
    config: ModelConfig, seq_len: int | None = None
) -> float:
    return 3.0 * flops_per_token(config, seq_len or config.sequence_len)


def total_training_flops(
    config: ModelConfig, tokens: int, seq_len: int | None = None
) -> float:
    return training_flops_per_token(config, seq_len) * tokens
