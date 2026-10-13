import json
import math
import shutil
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import models, normalizers

from gcw.config import (
    DataConfig,
    DiffusionProcessorConfig,
    EvalConfig,
    FlowProcessorConfig,
    LoRAConfig,
    ResumeConfig,
    TrainConfig,
)
from gcw.data import fineweb
from gcw.data.shards import ShardDataset, ShardWriter, data_loader, read_shard
from gcw.data.tokenizers import (
    IntTokenizer,
    Tokenizer,
    tokenizer_content_identity,
)
from gcw.data.training import FineWebBatchStream, training_data_identity
from gcw.evaluation.bpb import run_fineweb
from gcw.evaluation.scoring import evaluate_nll_bpb
from gcw.models import build_model
from gcw.training.core import train
from gcw.training.model_setup import _apply_trainable_policy
from tests.test_models import (
    tiny_config,
    tiny_gemma4_config,
    tiny_qwen35_config,
    tiny_qwen35_moe_config,
)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def prepare(*, high=False, **kwargs):
        tokenizer = IntTokenizer(vocab_size=128)
        if high:
            tokenizer = Tokenizer(
                BackendTokenizer(
                    models.WordLevel(
                        {"<unk>": 0, "document": 70000}, unk_token="<unk>"
                    )
                ),
                vocab_size=70001,
                eot_id=0,
            )
        monkeypatch.setattr(fineweb, "get_tokenizer", lambda _: tokenizer)
        monkeypatch.setattr(
            "gcw.data.tokenizers.get_tokenizer", lambda _: tokenizer
        )
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return [{"text": "document"}] * 12

        monkeypatch.setitem(
            sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset)
        )
        path = tmp_path / ("high" if high else "data")
        fineweb.prepare_fineweb(
            path, tokenizer="int", val_docs=2, shard_tokens=30, **kwargs
        )
        return path, tokenizer, calls

    return prepare


@pytest.mark.parametrize("high", [False, True])
def test_preparation_and_full_sequence_stream(prepared, high):
    path, tokenizer, calls = prepared(high=high)
    dataset = ShardDataset(path)
    assert dataset.dtype == np.dtype("<u4" if high else "<u2")
    assert calls == [
        (
            ("HuggingFaceFW/fineweb-edu",),
            {"name": "sample-10BT", "split": "train", "streaming": True},
        )
    ]
    assert dataset.manifest["documents"] == {"train": 10, "val": 2}
    expected = (tokenizer.encode("document") + [tokenizer.eot_id]) * 2
    assert (
        read_shard(path / "val_000000.bin", dataset.dtype).tolist() == expected
    )
    stream = FineWebBatchStream(
        path, batch_size=1, sequence_len=2, device="cpu", rank=0, world_size=1
    )
    batch = next(stream)
    assert batch.x[0].tolist() == expected[:2]
    assert batch.y[0].tolist() == expected[1:3]
    x, y, _ = next(data_loader(1, 2, data_dir=path))
    assert torch.equal(batch.x, x)
    assert torch.equal(batch.y, y)
    with pytest.raises(FileExistsError):
        fineweb.prepare_fineweb(path, tokenizer="int")


def test_uint32_rank_partition_and_resume(prepared):
    path, _, _ = prepared(high=True)
    kwargs = {
        "batch_size": 1,
        "sequence_len": 3,
        "device": "cpu",
        "world_size": 2,
    }
    rank0 = FineWebBatchStream(path, rank=0, **kwargs)
    rank1 = FineWebBatchStream(path, rank=1, **kwargs)
    first = next(rank0)
    second = next(rank1)
    assert first.x.tolist() == [[70000, 0, 70000]]
    assert second.x.tolist() == [[0, 70000, 0]]
    restored = FineWebBatchStream(path, rank=0, cursor=rank0.cursor, **kwargs)
    for _ in range(5):
        assert torch.equal(next(rank0).x, next(restored).x)
        assert rank0.cursor == restored.cursor


@pytest.mark.parametrize("ids", [[-1], [65536], [1.5], [[1]]])
def test_writer_rejects_invalid_ids(tmp_path, ids):
    with pytest.raises(ValueError, match="Token IDs"):
        ShardWriter(tmp_path, "train", 10).add_tokens(ids)


def test_manifest_validation_and_insufficient_data(prepared):
    path, _, _ = prepared()
    dataset = ShardDataset(path)
    with pytest.raises(ValueError, match="microbatch"):
        dataset.shards("train", 1000)
    with pytest.raises(ValueError, match="vocabulary"):
        dataset.validate_tokenizer("int", 64)
    dataset.validate_tokenizer("int", 256)
    shard = path / "train_000000.bin"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="truncated"):
        ShardDataset(path)


def test_legacy_shards_and_incomplete_preparation(tmp_path):
    writer = ShardWriter(tmp_path, "train", 10)
    writer.add_tokens(range(10))
    writer.close()
    dataset = ShardDataset(tmp_path)
    dataset.validate_tokenizer("cosmo2", 49152)
    with pytest.raises(ValueError, match="Legacy"):
        dataset.validate_tokenizer("hf:other", 49152)
    (tmp_path / ".fineweb-incomplete").touch()
    with pytest.raises(ValueError, match="Incomplete"):
        ShardDataset(tmp_path)


def test_identity_includes_normalization_and_excludes_location():
    backend = BackendTokenizer(
        models.WordLevel({"<unk>": 0, "text": 1}, unk_token="<unk>")
    )
    tokenizer = Tokenizer(backend, vocab_size=2)
    before = tokenizer_content_identity(tokenizer)
    tokenizer.source = "/moved/tokenizer"
    assert tokenizer_content_identity(tokenizer) == before
    backend.normalizer = normalizers.Lowercase()
    assert tokenizer_content_identity(tokenizer) != before


def test_same_vocab_tokenizer_mismatch(prepared, monkeypatch):
    path, _, _ = prepared()
    tokenizer = IntTokenizer(vocab_size=128, eot_id=1)
    monkeypatch.setattr(
        "gcw.data.tokenizers.get_tokenizer", lambda _: tokenizer
    )
    with pytest.raises(ValueError, match="does not match"):
        ShardDataset(path).validate_tokenizer("int", 128)


def test_hf_preparation_requires_eos(tmp_path, monkeypatch):
    from gcw.data.tokenizers import TransformersTokenizer

    tokenizer = TransformersTokenizer.__new__(TransformersTokenizer)
    tokenizer.enc = SimpleNamespace(
        backend_tokenizer=BackendTokenizer(models.WordLevel({"token": 0})),
        get_vocab=lambda: {"token": 0},
        eos_token_id=None,
    )
    tokenizer.eot_id = 0
    monkeypatch.setattr(fineweb, "get_tokenizer", lambda _: tokenizer)
    with pytest.raises(ValueError, match="explicit EOS"):
        fineweb.prepare_fineweb(tmp_path / "data", tokenizer="hf:local")
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize(
    "factory",
    [
        tiny_config,
        pytest.param(
            lambda variant: replace(
                tiny_config(variant), family="llama", rotary_style="llama"
            ),
            id="llama",
        ),
        pytest.param(
            lambda variant: replace(
                tiny_config(variant),
                family="qwen3",
                rotary_style="llama",
                qk_norm=True,
            ),
            id="qwen3",
        ),
        tiny_qwen35_config,
        tiny_qwen35_moe_config,
        tiny_gemma4_config,
    ],
)
@pytest.mark.parametrize(
    "arm",
    [
        "backbone",
        "lora",
        "bt",
        "bgt",
        "icarus",
        "longshot",
        "flow",
        "diffusion",
    ],
)
def test_fineweb_gradients_and_parameter_ownership(prepared, factory, arm):
    path, _, _ = prepared()
    variant = "tf" if arm in {"backbone", "lora"} else arm
    if arm in {"flow", "diffusion"}:
        variant = "bgt"
    config = factory(variant)
    if arm == "flow":
        config = replace(
            config, flow_processor=FlowProcessorConfig(num_steps=2)
        )
    if arm == "diffusion":
        config = replace(
            config,
            diffusion_processor=DiffusionProcessorConfig(
                num_steps=2, noise_scale=0.0
            ),
        )
    policy = arm if variant == "tf" else "cache_processor"
    if arm == "lora":
        config = replace(
            config,
            lora=LoRAConfig(rank=2, alpha=2, targets=("q_proj", "v_proj")),
        )
    model = build_model(config)
    model.init_weights()
    _apply_trainable_policy(model, policy)
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    batch = next(
        FineWebBatchStream(
            path,
            batch_size=1,
            sequence_len=8,
            device="cpu",
            rank=0,
            world_size=1,
        )
    )
    optimizer = torch.optim.SGD(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=0.1,
    )
    loss = model(batch.x, batch.y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    optimizer.step()
    changed = []
    for name, parameter in model.named_parameters():
        if not torch.equal(original[name], parameter):
            changed.append(name)
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(original[name], parameter)
    assert changed
    if policy == "cache_processor":
        assert all(name.startswith("cache_processor.") for name in changed)
    if policy == "lora":
        assert all("lora_" in name for name in changed)


def test_training_resume_and_standalone_evaluation(
    prepared, tmp_path, monkeypatch
):
    from gcw.checkpoint import load_checkpoint
    from gcw.training.checkpoint import CheckpointCompatibilityError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("GCW_KEEP_STEP_CHECKPOINTS", "1")
    monkeypatch.setenv("GCW_STEP_CHECKPOINT_EVERY", "1")
    path, _, _ = prepared()
    config = TrainConfig(
        model=tiny_config(),
        data=DataConfig(kind="fineweb", data_dir=str(path), tokenizer="int"),
        output_dir=str(tmp_path / "run"),
        sequence_len=8,
        max_steps=2,
        save_every=1,
        eval_every=1,
        eval_batches=1,
    )
    train(config)
    model, _, state = load_checkpoint(tmp_path / "run/checkpoint")
    assert state["fineweb_data_identity"] == training_data_identity(config)
    result = run_fineweb(
        tmp_path / "run/checkpoint",
        data_dir=path,
        sequence_len=8,
        eval_tokens=8,
        config=EvalConfig(batch_size=1),
    )
    assert math.isfinite(result["val_nll"])
    assert result["val_perplexity"] == pytest.approx(
        math.exp(result["val_nll"])
    )
    assert "val_bpb" not in result
    assert result["eval_tokens"] == 8
    resumed = replace(
        config,
        output_dir=str(tmp_path / "resumed"),
        resume=ResumeConfig(checkpoint=str(tmp_path / "run/checkpoint-1")),
    )
    train(resumed)
    restored, _, _ = load_checkpoint(tmp_path / "resumed/checkpoint")
    for expected, actual in zip(model.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    manifest = path / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["preparation"]["limit"] = 12
    manifest.write_text(json.dumps(payload))
    with pytest.raises(CheckpointCompatibilityError, match="fingerprint"):
        train(resumed)


def test_nll_includes_eos_and_bpb_preserves_byte_mask():
    class Model:
        def get_device(self):
            return "cpu"

        def eval(self):
            pass

        def __call__(self, *args, **kwargs):
            return torch.tensor([2.0, 4.0])

    batch = (torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    nll, bpb = evaluate_nll_bpb(Model(), [batch], 1, torch.tensor([0, 2]))
    assert nll == 3.0
    assert bpb == pytest.approx(4.0 / (2 * math.log(2)))
    assert evaluate_nll_bpb(Model(), [batch], 1, None) == (3.0, None)


def test_local_hf_tokenizer_preparation_and_relocation(tmp_path, monkeypatch):
    from transformers import PreTrainedTokenizerFast

    source = tmp_path / "tokenizer"
    backend = BackendTokenizer(
        models.WordLevel(
            {"<unk>": 0, "<eos>": 1, "document": 2}, unk_token="<unk>"
        )
    )
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>"
    ).save_pretrained(source)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(
            load_dataset=lambda *args, **kwargs: [{"text": "document"}] * 8
        ),
    )
    path = tmp_path / "data"
    fineweb.prepare_fineweb(path, tokenizer=f"hf:{source}", val_docs=2, limit=5)
    dataset = ShardDataset(path)
    assert dataset.manifest["documents"] == {"train": 3, "val": 2}
    assert (
        read_shard(path / "train_000000.bin", dataset.dtype).tolist()
        == [2, 1] * 3
    )
    moved = tmp_path / "moved"
    shutil.copytree(source, moved)
    dataset.validate_tokenizer(f"hf:{moved}", 8)


@pytest.mark.parametrize("change", ["dtype", "inventory", "checksum", "schema"])
def test_malformed_manifest(prepared, change):
    path, _, _ = prepared()
    manifest = path / "manifest.json"
    value = json.loads(manifest.read_text())
    if change == "dtype":
        value["dtype"] = "<i4"
    elif change == "inventory":
        value["shards"]["train"].append(value["shards"]["train"][0])
    elif change == "checksum":
        value["shards"]["train"][0]["sha256"] = "bad"
    else:
        value["schema"] = "unknown"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ShardDataset(path)


def test_non_smollm_evaluation_and_tokenizer_mismatch(
    prepared, tmp_path, monkeypatch
):
    from gcw.checkpoint import save_checkpoint

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path, tokenizer, _ = prepared()
    config = replace(tiny_config(), family="llama", rotary_style="llama")
    model = build_model(config)
    model.init_weights()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, config, {"tokenizer": "int"})
    result = run_fineweb(
        checkpoint,
        data_dir=path,
        sequence_len=4,
        eval_tokens=8,
        config=EvalConfig(batch_size=1),
    )
    assert result["sequence_len"] == 4
    assert math.isfinite(result["val_nll"])
    assert "val_bpb" not in result
    with pytest.raises(ValueError, match="model limits"):
        run_fineweb(checkpoint, data_dir=path, sequence_len=100)
    tokenizer.eot_id = 1
    with pytest.raises(ValueError, match="does not match"):
        run_fineweb(checkpoint, data_dir=path, sequence_len=4)


def test_interrupted_preparation_is_not_read_as_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(fineweb, "get_tokenizer", lambda _: IntTokenizer())

    def documents():
        yield {"text": "document"}
        raise RuntimeError("source interrupted")

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: documents()),
    )
    path = tmp_path / "data"
    with pytest.raises(RuntimeError, match="interrupted"):
        fineweb.prepare_fineweb(
            path, tokenizer="int", val_docs=0, shard_tokens=2
        )
    assert not (path / "manifest.json").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        ShardDataset(path)
    with pytest.raises(FileExistsError):
        fineweb.prepare_fineweb(path, tokenizer="int")


def test_cli_fineweb_options():
    from gcw.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "prepare",
            "fineweb",
            "--out",
            "data",
            "--tokenizer",
            "hf:local",
            "--val-docs",
            "20",
        ]
    )
    assert args.tokenizer == "hf:local"
    assert args.val_docs == 20
    args = parser.parse_args(
        ["eval", "fineweb", "--checkpoint", "model", "--sequence-len", "64"]
    )
    assert args.sequence_len == 64
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["eval", "fineweb", "--checkpoint", "model", "--sequence-len", "0"]
        )


@pytest.mark.parametrize("budget", [1, 9, 20, 90])
def test_exact_training_token_budget(prepared, budget):
    path, _, _ = prepared(train_tokens=budget)
    dataset = ShardDataset(path)
    assert (
        sum(entry["tokens"] for entry in dataset.inventory["train"]) == budget
    )
    assert sum(entry["tokens"] for entry in dataset.inventory["val"]) == 18
    assert dataset.manifest["documents"]["train"] == (budget + 8) // 9
    assert dataset.manifest["truncated_final_document_tokens"] == (-budget) % 9


def test_insufficient_training_token_budget_stays_incomplete(
    prepared, tmp_path
):
    with pytest.raises(ValueError, match="requested 100"):
        prepared(train_tokens=100)
    assert not (tmp_path / "data/manifest.json").exists()
    with pytest.raises(ValueError, match="Incomplete"):
        ShardDataset(tmp_path / "data")


def test_preparation_pins_source_revision(prepared):
    path, _, calls = prepared(
        train_tokens=20, dataset_revision="pinned-revision"
    )
    assert calls[0][1]["revision"] == "pinned-revision"
    assert (
        ShardDataset(path).manifest["source"]["revision"] == "pinned-revision"
    )
