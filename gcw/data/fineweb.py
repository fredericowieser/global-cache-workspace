from __future__ import annotations

from pathlib import Path

from tqdm import tqdm

from gcw.data.shards import ShardWriter
from gcw.data.tokenizers import get_tokenizer


def prepare_fineweb(out: str | Path, shard_tokens: int = 100_000_000, val_docs: int = 10_000, limit: int | None = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to prepare FineWeb shards.") from exc

    tokenizer = get_tokenizer("cosmo2")
    out = Path(out)
    train = ShardWriter(out, "train", shard_tokens)
    val = ShardWriter(out, "val", shard_tokens)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for i, row in enumerate(tqdm(ds, desc="fineweb")):
        if limit is not None and i >= limit:
            break
        ids = tokenizer.encode(row["text"]) + [tokenizer.eot_id]
        (val if i < val_docs else train).add_tokens(ids)
    val.close()
    train.close()
