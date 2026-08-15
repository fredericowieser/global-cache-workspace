import json

import torch
import pytest

from gcw.data.shards import ShardWriter
from gcw.data.training import FineWebBatchStream, IndexedJSONL, RandomBatchStream, SFTBatchStream


class TinyTokenizer:
    eot_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(char) % 64 + 1 for char in text]


def test_random_stream_is_rank_distinct_and_resumable():
    rank0 = RandomBatchStream(batch_size=2, sequence_len=4, vocab_size=128, device="cpu", seed=24, rank=0)
    rank1 = RandomBatchStream(batch_size=2, sequence_len=4, vocab_size=128, device="cpu", seed=24, rank=1)

    first0 = next(rank0)
    first1 = next(rank1)
    cursor = rank0.cursor
    expected = next(rank0)
    resumed = RandomBatchStream(
        batch_size=2,
        sequence_len=4,
        vocab_size=128,
        device="cpu",
        seed=999,
        rank=0,
        cursor=cursor,
    )

    assert not torch.equal(first0.x, first1.x)
    assert torch.equal(next(resumed).x, expected.x)


def test_fineweb_stream_resumes_at_next_global_microbatch(tmp_path):
    writer = ShardWriter(tmp_path, "train", shard_tokens=1000)
    writer.add_tokens([index % 128 for index in range(129)])
    writer.close()

    rank0 = FineWebBatchStream(tmp_path, batch_size=2, sequence_len=4, device="cpu", rank=0, world_size=2)
    rank1 = FineWebBatchStream(tmp_path, batch_size=2, sequence_len=4, device="cpu", rank=1, world_size=2)
    batch0 = next(rank0)
    batch1 = next(rank1)
    cursor = rank0.cursor
    expected = next(rank0)
    resumed = FineWebBatchStream(
        tmp_path,
        batch_size=2,
        sequence_len=4,
        device="cpu",
        rank=0,
        world_size=2,
        cursor=cursor,
    )

    assert rank1.cursor == cursor
    assert set(batch0.x.flatten().tolist()).isdisjoint(batch1.x.flatten().tolist())
    assert torch.equal(next(resumed).x, expected.x)


def test_sft_stream_partitions_global_rows_between_ranks():
    rows = [
        {"prompt": f"Q{index}", "answer": f"A{index}"}
        for index in range(8)
    ]
    rank0 = SFTBatchStream(
        rows,
        TinyTokenizer(),
        batch_size=1,
        sequence_len=16,
        device="cpu",
        rank=0,
        world_size=2,
    )
    rank1 = SFTBatchStream(
        rows,
        TinyTokenizer(),
        batch_size=1,
        sequence_len=16,
        device="cpu",
        rank=1,
        world_size=2,
    )

    batch0 = next(rank0)
    batch1 = next(rank1)

    assert rank0.cursor == rank1.cursor
    assert not torch.equal(batch0.x, batch1.x)
    for batch in (batch0, batch1):
        assert batch.attention_mask is not None
        assert batch.attention_mask.dtype == torch.bool
        assert batch.attention_mask.shape == batch.x.shape
        assert batch.attention_mask.all()


def test_indexed_jsonl_supports_random_access_and_slices(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": str(i), "answer": "x"}) for i in range(4)) + "\n")
    rows = IndexedJSONL(path)

    assert len(rows) == 4
    assert rows[2]["prompt"] == "2"
    assert [row["prompt"] for row in rows[1:3]] == ["1", "2"]
    assert rows._handle is not None
    rows.close()
    assert rows._handle is None
    assert rows[1]["prompt"] == "1"
    rows.close()


def test_sft_stream_rejects_a_global_batch_larger_than_the_dataset():
    with pytest.raises(ValueError, match="disjoint rows"):
        SFTBatchStream(
            [{"prompt": "Q", "answer": "A"}],
            TinyTokenizer(),
            batch_size=1,
            sequence_len=16,
            device="cpu",
            rank=0,
            world_size=2,
        )
