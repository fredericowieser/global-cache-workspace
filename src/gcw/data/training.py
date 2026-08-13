from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from gcw.config import TrainConfig
from gcw.data.shards import default_data_dir, list_shards, read_shard
from gcw.data.sft import iter_jsonl, sft_batch
from gcw.data.tokenizers import get_tokenizer


@dataclass(frozen=True)
class TrainingBatch:
    x: torch.Tensor
    y: torch.Tensor
    attention_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class RandomCursor:
    rng_state: tuple[int, ...]
    kind: str = "random"


@dataclass(frozen=True)
class FineWebCursor:
    epoch: int = 0
    shard_index: int = 0
    step_in_shard: int = 0
    kind: str = "fineweb"


@dataclass(frozen=True)
class SFTCursor:
    epoch: int = 0
    row: int = 0
    kind: str = "sft"


LoaderCursor = RandomCursor | FineWebCursor | SFTCursor


class TrainingBatchStream(Protocol):
    @property
    def cursor(self) -> LoaderCursor: ...

    def __next__(self) -> TrainingBatch: ...


class RandomBatchStream:
    def __init__(
        self,
        *,
        batch_size: int,
        sequence_len: int,
        vocab_size: int,
        device: torch.device | str,
        seed: int,
        rank: int,
        cursor: RandomCursor | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device)
        if cursor is None:
            self.generator.manual_seed(seed + rank)
        else:
            state = torch.tensor(cursor.rng_state, dtype=torch.uint8)
            self.generator.set_state(state)

    @property
    def cursor(self) -> RandomCursor:
        state = self.generator.get_state().cpu().tolist()
        return RandomCursor(rng_state=tuple(state))

    def __next__(self) -> TrainingBatch:
        x = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.sequence_len),
            device=self.device,
            generator=self.generator,
        )
        y = torch.roll(x, shifts=-1, dims=1)
        return TrainingBatch(x=x, y=y)


class FineWebBatchStream:
    def __init__(
        self,
        data_dir: str | Path,
        *,
        batch_size: int,
        sequence_len: int,
        device: torch.device | str,
        rank: int,
        world_size: int,
        cursor: FineWebCursor | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        self.shards = list_shards(self.data_dir, "train")
        if not self.shards:
            raise FileNotFoundError(f"No train shards found in {self.data_dir}")
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self._cursor = cursor or FineWebCursor()

    @property
    def cursor(self) -> FineWebCursor:
        return self._cursor

    def __next__(self) -> TrainingBatch:
        while True:
            shard = read_shard(self.shards[self._cursor.shard_index])
            global_micro_tokens = self.batch_size * self.sequence_len * self.world_size
            steps = max(0, (len(shard) - 1) // global_micro_tokens)
            if self._cursor.step_in_shard >= steps:
                self._advance_shard()
                continue
            rank_offset = self.rank * self.batch_size * self.sequence_len
            base = self._cursor.step_in_shard * global_micro_tokens + rank_offset
            length = self.batch_size * self.sequence_len
            tokens = torch.from_numpy(shard[base : base + length + 1].astype("int64"))
            x = tokens[:-1].view(self.batch_size, self.sequence_len).to(self.device)
            y = tokens[1:].view(self.batch_size, self.sequence_len).to(self.device)
            self._cursor = FineWebCursor(
                epoch=self._cursor.epoch,
                shard_index=self._cursor.shard_index,
                step_in_shard=self._cursor.step_in_shard + 1,
            )
            return TrainingBatch(x=x, y=y)

    def _advance_shard(self) -> None:
        shard_index = self._cursor.shard_index + 1
        epoch = self._cursor.epoch
        if shard_index == len(self.shards):
            shard_index = 0
            epoch += 1
        self._cursor = FineWebCursor(epoch=epoch, shard_index=shard_index, step_in_shard=0)


class SFTBatchStream:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer,
        *,
        batch_size: int,
        sequence_len: int,
        device: torch.device | str,
        rank: int,
        world_size: int,
        response_prefix: str = "",
        prompt_suffix: str = "",
        chat_template: bool = False,
        enable_thinking: bool = False,
        loss_scope: Literal["answer_only", "full_sequence"] = "answer_only",
        cursor: SFTCursor | None = None,
    ) -> None:
        if not rows:
            raise ValueError("SFT rows must not be empty")
        self.rows = rows
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        global_rows = batch_size * world_size
        if len(rows) < global_rows:
            raise ValueError(f"SFT dataset has {len(rows)} rows but global batch requires {global_rows} disjoint rows")
        self.response_prefix = response_prefix
        self.prompt_suffix = prompt_suffix
        self.chat_template = chat_template
        self.enable_thinking = enable_thinking
        self.loss_scope = loss_scope
        self._cursor = cursor or SFTCursor()
        if self._cursor.row >= len(rows):
            raise ValueError(f"SFT cursor row {self._cursor.row} exceeds dataset length {len(rows)}")

    @property
    def cursor(self) -> SFTCursor:
        return self._cursor

    def __next__(self) -> TrainingBatch:
        rank_start = self._cursor.row + self.rank * self.batch_size
        batch_rows = [self.rows[(rank_start + offset) % len(self.rows)] for offset in range(self.batch_size)]
        x, y, attention_mask = sft_batch(
            batch_rows,
            self.tokenizer,
            self.sequence_len,
            device=self.device,
            response_prefix=self.response_prefix,
            prompt_suffix=self.prompt_suffix,
            chat_template=self.chat_template,
            enable_thinking=self.enable_thinking,
            loss_scope=self.loss_scope,
            return_attention_mask=True,
        )
        global_rows = self.batch_size * self.world_size
        absolute_next = self._cursor.row + global_rows
        self._cursor = SFTCursor(
            epoch=self._cursor.epoch + absolute_next // len(self.rows),
            row=absolute_next % len(self.rows),
        )
        return TrainingBatch(x=x, y=y, attention_mask=attention_mask)


def build_training_stream(
    config: TrainConfig,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    cursor: LoaderCursor | None = None,
) -> TrainingBatchStream:
    if config.data.kind == "random":
        if cursor is not None and not isinstance(cursor, RandomCursor):
            raise TypeError(f"expected RandomCursor, got {type(cursor).__name__}")
        random_cursor = cursor if isinstance(cursor, RandomCursor) else None
        return RandomBatchStream(
            batch_size=config.batch_size,
            sequence_len=config.sequence_len,
            vocab_size=config.model.vocab_size,
            device=device,
            seed=config.seed,
            rank=rank,
            cursor=random_cursor,
        )
    if config.mode == "sft":
        if config.data.dataset is None:
            raise ValueError("SFT mode requires data.dataset JSONL path")
        tokenizer_name = config.data.tokenizer
        if tokenizer_name == "hf":
            if config.data.tokenizer_source is None:
                raise ValueError("data.tokenizer_source is required when data.tokenizer='hf'")
            tokenizer_name = f"hf:{config.data.tokenizer_source}"
        rows = list(iter_jsonl(config.data.dataset))
        if config.data.sft_skip_rows:
            if config.data.sft_skip_rows >= len(rows):
                raise ValueError(f"data.sft_skip_rows={config.data.sft_skip_rows} exceeds dataset length {len(rows)}")
            rows = rows[config.data.sft_skip_rows :]
        if config.data.sft_take_rows is not None:
            rows = rows[: config.data.sft_take_rows]
        if cursor is not None and not isinstance(cursor, SFTCursor):
            raise TypeError(f"expected SFTCursor, got {type(cursor).__name__}")
        sft_cursor = cursor if isinstance(cursor, SFTCursor) else None
        return SFTBatchStream(
            rows,
            get_tokenizer(tokenizer_name),
            batch_size=config.batch_size,
            sequence_len=config.sequence_len,
            device=device,
            rank=rank,
            world_size=world_size,
            response_prefix=config.data.response_prefix,
            prompt_suffix=config.data.prompt_suffix,
            chat_template=config.data.sft_chat_template,
            enable_thinking=config.data.sft_enable_thinking,
            loss_scope=config.data.sft_loss_scope,
            cursor=sft_cursor,
        )
    if cursor is not None and not isinstance(cursor, FineWebCursor):
        raise TypeError(f"expected FineWebCursor, got {type(cursor).__name__}")
    fineweb_cursor = cursor if isinstance(cursor, FineWebCursor) else None
    return FineWebBatchStream(
        config.data.data_dir or default_data_dir(),
        batch_size=config.batch_size,
        sequence_len=config.sequence_len,
        device=device,
        rank=rank,
        world_size=world_size,
        cursor=fineweb_cursor,
    )


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cursor_to_dict(cursor: LoaderCursor) -> dict[str, Any]:
    return asdict(cursor)


def cursor_from_dict(value: dict[str, Any]) -> LoaderCursor:
    kind = value.get("kind")
    fields = {key: item for key, item in value.items() if key != "kind"}
    if kind == "random":
        fields["rng_state"] = tuple(fields["rng_state"])
        return RandomCursor(**fields)
    if kind == "fineweb":
        return FineWebCursor(**fields)
    if kind == "sft":
        return SFTCursor(**fields)
    raise ValueError(f"unsupported loader cursor kind: {kind!r}")


def training_data_identity(config: TrainConfig) -> dict[str, Any]:
    if config.data.kind == "random":
        return {"kind": "random"}
    if config.mode == "sft":
        if config.data.dataset is None:
            raise ValueError("SFT mode requires data.dataset JSONL path")
        path = Path(config.data.dataset).expanduser().resolve()
        return {
            "kind": "sft",
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    data_dir = Path(config.data.data_dir).expanduser().resolve() if config.data.data_dir else default_data_dir().resolve()
    shards = [Path(path) for path in list_shards(data_dir, "train")]
    if not shards:
        raise FileNotFoundError(f"No train shards found in {data_dir}")
    return {
        "kind": "fineweb",
        "path": str(data_dir),
        "shards": [{"name": path.name, "size": path.stat().st_size} for path in shards],
    }
