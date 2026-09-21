from types import SimpleNamespace

import pytest
import torch

from gcw.models.backbones.gemma4.model import Gemma4TF
from gcw.models.backbones.transformer import TF
from gcw.models.cache_processors.state import CacheState, StaticCacheState
from gcw.models.model import LanguageModel
from gcw.models.cache_processors.update_metrics import UPDATE_BATCH, UpdateBatch


@pytest.fixture(
    params=[
        "reference",
        "fast",
        "gemma-nonuniform",
        "wrapped-reference",
        "wrapped-fast",
    ]
)
def scripted_generation(request, monkeypatch):
    return _scripted_generation(request.param, monkeypatch)


@pytest.fixture(params=["reference", "fast", "gemma-nonuniform"])
def scripted_backbone_generation(request, monkeypatch):
    return _scripted_generation(request.param, monkeypatch)


def _scripted_generation(kind, monkeypatch):
    def build(script, sequence_len=4):
        cls = Gemma4TF if kind == "gemma-nonuniform" else TF
        backbone = cls.__new__(cls)
        torch.nn.Module.__init__(backbone)
        backbone.config = SimpleNamespace(
            sequence_len=sequence_len,
            head_dim=1,
            cache_num_layers=2,
            num_kv_heads=1,
            cache_layer_indices=[0, 1],
            kv_heads_for_layer=lambda layer: 1,
            head_dim_for_layer=lambda layer: layer + 1,
        )
        backbone.embed_tokens = SimpleNamespace(weight=torch.empty(16, 1))
        calls = []
        predictions = torch.tensor(script).T

        def forward(input_ids, start_pos, cache):
            expected_cache = (
                StaticCacheState if kind.endswith("fast") else CacheState
            )
            assert isinstance(cache, expected_cache)
            step = min(len(calls), len(predictions) - 1)
            calls.append((input_ids.clone(), start_pos))
            logits = torch.full((*input_ids.shape, 16), -100.0)
            logits[:, -1].scatter_(1, predictions[step, :, None], 100.0)
            return logits, cache

        monkeypatch.setattr(backbone, "forward_chunk", forward)
        monkeypatch.setattr(backbone, "forward_chunk_static", forward)
        fast = kind.endswith("fast") or kind == "gemma-nonuniform"
        if kind.startswith("wrapped-"):
            config = SimpleNamespace(
                processor_generation_mode="post_prefill_relative",
                cache_processor=SimpleNamespace(
                    process_every=64, generation_process_every=None
                ),
            )
            model = LanguageModel(config, backbone, None)
        else:
            model = backbone

        def generate(prompt, max_new_tokens, **kwargs):
            if model is backbone:
                kwargs.setdefault("process_prefill_windows", False)
            method = model.generate_fast if fast else model.generate
            return method(torch.tensor(prompt), max_new_tokens, **kwargs)

        return generate, calls

    return build


@pytest.mark.parametrize("eos", [[2, 3], (2, 3)])
@pytest.mark.parametrize("pad", [None, 0, 11])
def test_multiple_eos_staggered_finish(scripted_generation, eos, pad):
    generate, calls = scripted_generation([[3, 9, 9], [6, 7, 2]])
    out = generate(
        [[8, 5], [8, 5]], 6, eos_token_id=eos, pad_token_id=pad
    )
    fill = 2 if pad is None else pad
    assert out.tolist() == [[8, 5, 3, fill, fill], [8, 5, 6, 7, 2]]
    assert len(calls) == 3
    assert calls[2][0].tolist() == [[fill], [7]]


@pytest.mark.parametrize("eos", [None, [], ()])
@pytest.mark.parametrize("cap", [0, 3])
def test_no_eos_obeys_cap(scripted_generation, eos, cap):
    generate, calls = scripted_generation([[2, 3, 4], [3, 2, 5]])
    out = generate([[8], [8]], cap, eos_token_id=eos)
    assert out.tolist() == [[8] + [2, 3, 4][:cap], [8] + [3, 2, 5][:cap]]
    assert len(calls) == cap + 1


@pytest.mark.parametrize("eos", [2, [2, 3], (2, 3)])
def test_eos_not_reached_obeys_cap(scripted_generation, eos):
    generate, calls = scripted_generation([[6, 7, 8]])
    out = generate([[5]], 3, eos_token_id=eos)
    assert out.tolist() == [[5, 6, 7, 8]]
    assert len(calls) == 4


def test_scalar_eos_staggered_finish(scripted_generation):
    generate, calls = scripted_generation([[2, 9], [6, 2]])
    out = generate([[8], [8]], 5, eos_token_id=2)
    assert out.tolist() == [[8, 2, 2], [8, 6, 2]]
    assert len(calls) == 2


@pytest.mark.parametrize("eos", [None, [], ()])
def test_stop_sequences_without_eos_fill_zero(scripted_generation, eos):
    generate, calls = scripted_generation([[4, 5, 9], [6, 4, 5]])
    out = generate(
        [[8], [8]], 6, eos_token_id=eos, stop_sequences=[[], [4, 5]]
    )
    assert out.tolist() == [[8, 4, 5, 0], [8, 6, 4, 5]]
    assert len(calls) == 3


def test_eos_and_task_stop_finish_different_rows(scripted_generation):
    generate, calls = scripted_generation([[3, 9, 9], [6, 4, 5]])
    out = generate(
        [[8], [8]],
        6,
        eos_token_id=[2, 3],
        stop_sequences=[[4, 5]],
        pad_token_id=0,
    )
    assert out.tolist() == [[8, 3, 0, 0], [8, 6, 4, 5]]
    assert len(calls) == 3


@pytest.mark.parametrize("sequence_len", [2, 4])
def test_stop_cannot_start_in_prompt(scripted_generation, sequence_len):
    generate, calls = scripted_generation([[5, 4, 5, 9]], sequence_len)
    out = generate(
        [[7, 8, 4]], 6, stop_sequences=[[4, 5]], eos_token_id=[2, 3]
    )
    assert out.tolist() == [[7, 8, 4][-sequence_len:] + [5, 4, 5]]
    assert len(calls) == 3


@pytest.mark.parametrize(
    "prefill_windows,cadence,expected",
    [
        (
            True,
            {"generation_process_every": 1},
            [(2, "prefill"), (4, "prefill"), (5, "decoding"), (6, "decoding")],
        ),
        (
            False,
            {
                "relative_generation_cadence": True,
                "process_after_prefill": True,
            },
            [(4, "prefill"), (6, "decoding")],
        ),
        (False, {"relative_generation_cadence": True}, [(6, "decoding")]),
        (
            False,
            {"generation_process_every": 2},
            [(4, "prefill"), (6, "decoding")],
        ),
    ],
)
def test_update_phase_follows_cache_contents(
    scripted_backbone_generation, prefill_windows, cadence, expected
):
    generate, calls = scripted_backbone_generation([[7]], sequence_len=8)
    batch = UpdateBatch(1, "cpu")
    batch.phase = "decoding"
    observed = []

    def processor(cache, **kwargs):
        position = calls[-1][1] + calls[-1][0].shape[1]
        if batch.active[0]:
            observed.append((position, batch.phase))
        batch.add(1, 1, "value", torch.ones(1, 1), batch.active)
        return cache

    token = UPDATE_BATCH.set(batch)
    try:
        generate(
            [[1, 2, 3, 4]],
            4,
            processor=processor,
            process_every=2,
            process_prefill_windows=prefill_windows,
            **cadence,
        )
        assert batch.phase == "decoding"
    finally:
        UPDATE_BATCH.reset(token)
    assert observed == expected
    assert sum(
        value["applications"] for value in batch.rows()[0].values()
    ) == len(expected)


@pytest.mark.parametrize("cap", [0, 1])
def test_update_phases_with_no_contributing_decode(
    scripted_backbone_generation, cap
):
    generate, _ = scripted_backbone_generation([[7]], sequence_len=8)
    batch = UpdateBatch(1, "cpu")

    def processor(cache, **kwargs):
        batch.add(1, 1, "value", torch.ones(1, 1), batch.active)
        return cache

    token = UPDATE_BATCH.set(batch)
    try:
        generate(
            [[1, 2, 3, 4]],
            cap,
            processor=processor,
            process_every=2,
            process_prefill_windows=True,
            generation_process_every=1,
        )
        assert batch.phase == "prefill"
    finally:
        UPDATE_BATCH.reset(token)
    row = batch.rows()[0]
    assert set(row) == {"prefill:1:1:value"}
    assert row["prefill:1:1:value"]["applications"] == 1


def test_generation_restores_phase_after_failure(scripted_backbone_generation):
    generate, _ = scripted_backbone_generation([[7]])
    batch = UpdateBatch(1, "cpu")

    def processor(cache, **kwargs):
        if batch.phase == "decoding":
            raise RuntimeError("decode failed")
        return cache

    token = UPDATE_BATCH.set(batch)
    try:
        with pytest.raises(RuntimeError, match="decode failed"):
            generate([[1]], 4, processor=processor, generation_process_every=1)
        assert batch.phase == "prefill"
    finally:
        UPDATE_BATCH.reset(token)
