from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.data.training import TrainingBatch
from gcw.models import build_model
from gcw.training.distributed import _align_fsdp_batch


def _align_batch_worker(rank: int, init_file: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        for widths in [
            (1024, 980),
            (845, 1024),
            (1024, 909),
            (931, 1024),
            (845, 845),
        ]:
            for masked in (False, True):
                width = widths[rank]
                x = torch.arange(2 * width).reshape(2, width)
                y = x.clone()
                y[:, :3] = -100
                mask = torch.ones_like(x, dtype=torch.bool) if masked else None
                if mask is not None:
                    mask[0, -1] = False
                    y[0, -1] = -100
                batch = TrainingBatch(x, y, mask)
                aligned = _align_fsdp_batch(batch)
                assert aligned.x.shape == (2, max(widths))
                torch.testing.assert_close(aligned.x[:, :width], x)
                torch.testing.assert_close(aligned.y[:, :width], y)
                assert torch.equal(aligned.y[aligned.y != -100], y[y != -100])
                if width == max(widths):
                    assert aligned is batch
                else:
                    assert (aligned.x[:, width:] == 0).all()
                    assert (aligned.y[:, width:] == -100).all()
                    assert not aligned.attention_mask[:, width:].any()
                    expected_mask = (
                        mask
                        if mask is not None
                        else torch.ones_like(x, dtype=torch.bool)
                    )
                    torch.testing.assert_close(
                        aligned.attention_mask[:, :width], expected_mask
                    )
                window_counts = [
                    torch.zeros((), dtype=torch.long) for _ in range(2)
                ]
                dist.all_gather(
                    window_counts, torch.tensor((aligned.x.shape[1] + 63) // 64)
                )
                assert window_counts[0].item() == window_counts[1].item()
    finally:
        dist.destroy_process_group()


def test_rank_local_batches_align_window_counts(tmp_path):
    mp.spawn(
        _align_batch_worker, args=(str(tmp_path / "init"),), nprocs=2, join=True
    )


@pytest.mark.parametrize("variant", ["tf", "bt", "bgt"])
def test_alignment_preserves_causal_loss_and_gradients(monkeypatch, variant):
    torch.manual_seed(24)
    config = ModelConfig(
        variant=variant,
        family="smollm",
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=32,
        cache_processor=None
        if variant == "tf"
        else CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
        ),
    )
    model = build_model(config)
    model.init_weights()
    x = torch.randint(0, 32, (2, 9))
    y = x.roll(-1, dims=1)
    y[:, :3] = -100
    y[:, -1] = -100
    batch = TrainingBatch(x, y, torch.ones_like(x, dtype=torch.bool))
    original_loss = model(x, y, attention_mask=batch.attention_mask)
    original_loss.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)

    def remote_width(width, op):
        assert op == dist.ReduceOp.MAX
        width.fill_(16)

    monkeypatch.setattr(dist, "all_reduce", remote_width)
    aligned = _align_fsdp_batch(batch)
    padded_loss = model(
        aligned.x, aligned.y, attention_mask=aligned.attention_mask
    )
    padded_loss.backward()
    torch.testing.assert_close(padded_loss, original_loss)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(
                parameter.grad, gradients[name], rtol=1e-4, atol=1e-6
            )
