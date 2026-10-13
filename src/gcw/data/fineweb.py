from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from gcw.data.shards import SHARD_SCHEMA, ShardWriter
from gcw.data.tokenizers import get_tokenizer, tokenizer_content_identity


def prepare_fineweb(
    out: str | Path,
    shard_tokens: int = 100_000_000,
    val_docs: int = 10_000,
    limit: int | None = None,
    *,
    tokenizer: str = "cosmo2",
    train_tokens: int | None = None,
    dataset_revision: str | None = None,
) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Install datasets to prepare FineWeb shards."
        ) from exc

    if val_docs < 0 or (limit is not None and limit <= 0) or shard_tokens <= 0:
        raise ValueError("Invalid FineWeb preparation limits")
    if train_tokens is not None and (
        isinstance(train_tokens, bool)
        or not isinstance(train_tokens, int)
        or train_tokens <= 0
    ):
        raise ValueError("train_tokens must be a positive integer")
    tokenizer_name = tokenizer
    encoder = get_tokenizer(tokenizer_name)
    identity = tokenizer_content_identity(encoder)
    if identity["max_token_id"] > 2**32 - 1:
        raise ValueError("Tokenizer IDs exceed uint32 storage")
    dtype = "<u2" if identity["max_token_id"] < 2**16 else "<u4"
    out = Path(out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"FineWeb output must be empty: {out}")
    marker = out / ".fineweb-incomplete"
    with marker.open("x"):
        pass
    train = ShardWriter(out, "train", shard_tokens, dtype=dtype)
    val = ShardWriter(out, "val", shard_tokens, dtype=dtype)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        **({"revision": dataset_revision} if dataset_revision else {}),
    )
    documents = {"train": 0, "val": 0}
    truncated_tokens = 0
    for i, row in enumerate(tqdm(ds, desc="fineweb")):
        if limit is not None and i >= limit:
            break
        ids = encoder.encode(row["text"]) + [identity["eos_token_id"]]
        if i >= val_docs and train_tokens is not None:
            remaining = train_tokens - train.total_tokens
            truncated_tokens = max(0, len(ids) - remaining)
            ids = ids[:remaining]
        (val if i < val_docs else train).add_tokens(ids)
        documents["val" if i < val_docs else "train"] += 1
        if train_tokens is not None and train.total_tokens == train_tokens:
            break
    val.close()
    train.close()
    if train_tokens is not None and train.total_tokens != train_tokens:
        raise ValueError(
            f"FineWeb source/document limit supplied {train.total_tokens} training tokens; "
            f"requested {train_tokens}. Use a fresh output directory and a larger document limit."
        )
    manifest = {
        "schema": SHARD_SCHEMA,
        "source": {
            "dataset": "HuggingFaceFW/fineweb-edu",
            "subset": "sample-10BT",
            "split": "train",
            **({"revision": dataset_revision} if dataset_revision else {}),
        },
        "preparation": {
            "val_docs": val_docs,
            "limit": limit,
            "shard_tokens": shard_tokens,
            "train_tokens": train_tokens,
        },
        "tokenizer": {"source": tokenizer_name, **identity},
        "dtype": dtype,
        "documents": documents,
        "truncated_final_document_tokens": truncated_tokens,
        "shards": {"train": train.inventory, "val": val.inventory},
    }
    temporary = out / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(out / "manifest.json")
    marker.unlink()
