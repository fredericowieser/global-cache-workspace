import os
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from gcw.models import build_model
from gcw.models.cache_processors.longshot import GlobalCacheProcessor
from gcw.models.cache_processors.ops import gather_per_cache_head_kv
from gcw.performance.longshot_scores import native_cache_squared_l2
from test_gemma4_longshot import mixed_config

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        os.environ.get("GCW_RUN_GEMMA_CACHE_CUDA_TESTS") != "1"
        or not torch.cuda.is_available(),
        reason="set GCW_RUN_GEMMA_CACHE_CUDA_TESTS=1 on a CUDA host",
    ),
]


@pytest.mark.parametrize("subset", [None, [1, 3]])
@pytest.mark.parametrize("autocast", [False, True])
@torch.no_grad()
def test_cuda_static_decode_and_generation_parity(subset, autocast):
    torch.manual_seed(91)
    cfg = replace(
        mixed_config(out_proj_init="normal"), processor_layer_indices=subset
    )
    model = build_model(cfg).cuda().eval()
    if not autocast:
        model = model.bfloat16()
    tokens = torch.randint(1, 128, (2, 19), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        static = model.backbone.allocate_generation_cache(2, 24, tokens.device)
        assert all(k.dtype == torch.bfloat16 for k in static.keys)
        dynamic, start = None, 0
        for steps in [3, 2, 7, 1, 6]:
            if start:
                model.cache_processor(static)
                dynamic = model.cache_processor(dynamic)
            actual, static = model.backbone.forward_chunk_static(
                tokens[:, start : start + steps], start, static
            )
            expected, dynamic = model.backbone.forward_chunk(
                tokens[:, start : start + steps], start, dynamic
            )
            torch.testing.assert_close(actual, expected, atol=0.025, rtol=0.025)
            start += steps
        torch.testing.assert_close(
            model.generate_fast(tokens[:, :9], 5),
            model.generate(tokens[:, :9], 5),
            atol=0,
            rtol=0,
        )
    for spec in model.cache_processor.shape_groups:
        from gcw.models.cache_processors.state import CacheState

        group = CacheState(
            [static.layer(i)[0] for i in spec.layers],
            [static.layer(i)[1] for i in spec.layers],
            static.scores,
        )
        indices = group.all_indices()[:, None].expand(
            -1, len(spec.head_indices), -1
        )
        merged = gather_per_cache_head_kv(group, indices)
        torch.testing.assert_close(
            native_cache_squared_l2(group, max_temp_bytes=1024),
            merged.float().square().sum(-1),
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("subset", [None, [1, 3]])
def test_cuda_bf16_checkpointed_list_training_matches_packed(
    monkeypatch, subset
):
    torch.manual_seed(92)
    cfg = replace(
        mixed_config(out_proj_init="normal"), processor_layer_indices=subset
    )
    model = build_model(cfg).cuda().bfloat16().train()
    reference = deepcopy(model)
    tokens = torch.randint(1, 128, (2, 13), device="cuda")
    targets = torch.randint(1, 128, (2, 13), device="cuda")
    with monkeypatch.context() as patch:
        patch.setattr(
            GlobalCacheProcessor,
            "recurrent_cache_format",
            property(lambda self: "windowed"),
        )
        expected = reference(
            tokens,
            targets,
            activation_checkpoint_windows=2,
            truncated_bptt_windows=2,
        )
        expected.backward()
    actual = model(
        tokens,
        targets,
        activation_checkpoint_windows=2,
        truncated_bptt_windows=2,
    )
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
    for parameter, ref_parameter in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        if parameter.grad is None:
            assert ref_parameter.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, ref_parameter.grad, atol=0.002, rtol=0.02
            )
