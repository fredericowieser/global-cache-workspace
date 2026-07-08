from argparse import Namespace
import csv
import gzip
import json
import math

import torch

from gcw.analysis.head_loss_dump import RawDumpSettings, raw_loss_rows_for_example, run
from gcw.analysis.head_synergy import AnalysisExample, CheckpointRecord, HeadMeanAblator, UnigramBaseline
from gcw.checkpoint import save_checkpoint
from gcw.config import CacheProcessorConfig, ModelConfig
from gcw.data.tokenizers import IntTokenizer
from gcw.models import build_model


def tiny_config(variant="tf", *, legacy=False):
    cp = None
    if variant in {"bt", "bgt"}:
        cp = CacheProcessorConfig(
            process_every=4,
            topk=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
        )
    return ModelConfig(
        variant=variant,
        family="smollm",
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        cache_processor=cp,
        legacy_bottleneck_generate=legacy,
    )


def test_head_loss_dump_dry_run_lists_family_jobs_and_checkpoints(capsys):
    args = Namespace(
        root="/shared",
        out_dir="/tmp/out",
        tasks=["gsm8k_openmath"],
        families=["llama32_1b", "llama32_3b", "llama31_8b", "qwen3_06b"],
        variants=["sft", "bt", "bgt"],
        limit_checkpoints=None,
        dry_run=True,
    )

    run(args)

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["family_jobs"]) == 4
    assert len(payload["checkpoints"]) == 12
    assert [record["variant"] for record in payload["family_jobs"][0]["checkpoints"]] == ["sft", "bt", "bgt"]


def test_raw_loss_rows_record_token_losses_and_formulas():
    torch.manual_seed(0)
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    tokenizer = IntTokenizer(vocab_size=256)
    baseline = UnigramBaseline(vocab_size=256)
    baseline.update(tokenizer.encode("abc\nde"))
    ablator = HeadMeanAblator(model)
    ablator.head_means = {0: torch.zeros(2, 8)}
    settings = RawDumpSettings(num_splits=2, group_fraction=0.5, max_length=16, max_target_tokens=8, seed=7)
    example = AnalysisExample("toy", 3, "abc", "de")

    metadata, rows = raw_loss_rows_for_example(
        record=CheckpointRecord("toy", "sft", "/tmp/checkpoint"),
        model=model,
        tokenizer=tokenizer,
        baseline=baseline,
        ablator=ablator,
        example=example,
        settings=settings,
    )

    assert metadata.target_len == 2
    assert len(rows) == 4
    for row in rows:
        expected_raw = (float(row["nll_a"]) + float(row["nll_b"])) - (float(row["nll_full"]) + float(row["nll_base"]))
        expected_useful = float(row["nll_base"]) - float(row["nll_full"])
        assert math.isclose(float(row["raw_token"]), expected_raw)
        assert math.isclose(float(row["useful_token"]), expected_useful)


def test_legacy_bgt_raw_dump_uses_context_cache_boundary(monkeypatch):
    torch.manual_seed(1)
    model = build_model(tiny_config("bgt", legacy=True)).eval()
    model.init_weights()
    tokenizer = IntTokenizer(vocab_size=256)
    baseline = UnigramBaseline(vocab_size=256)
    baseline.update(tokenizer.encode("abc\nde"))
    ablator = HeadMeanAblator(model)
    ablator.head_means = {0: torch.zeros(2, 8)}
    calls = []

    def fake_forward(cache, *, select_all=False):
        calls.append(select_all)
        return cache

    monkeypatch.setattr(model.cache_processor, "forward", fake_forward)
    raw_loss_rows_for_example(
        record=CheckpointRecord("toy", "bgt", "/tmp/checkpoint"),
        model=model,
        tokenizer=tokenizer,
        baseline=baseline,
        ablator=ablator,
        example=AnalysisExample("toy", 0, "abc", "de"),
        settings=RawDumpSettings(num_splits=1, group_fraction=0.5, max_length=16, max_target_tokens=8, seed=1),
    )

    assert calls == [True]


def test_synthetic_head_loss_dump_cli_writes_raw_outputs(tmp_path):
    root = tmp_path / "shared"
    checkpoint = root / "checkpoints" / "sft_backbones" / "toy" / "tf"
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    save_checkpoint(checkpoint, model, model.config, {})
    examples = tmp_path / "examples.jsonl"
    examples.write_text(json.dumps({"task": "toy", "prompt": "abc", "target": "de"}) + "\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    args = Namespace(
        root=str(root),
        out_dir=str(out_dir),
        tasks=["toy"],
        families=["toy"],
        variants=["sft"],
        num_examples=1,
        num_calibration=1,
        num_splits=2,
        group_fraction=0.5,
        max_length=16,
        max_target_tokens=8,
        seed=1,
        device="cpu",
        limit_checkpoints=None,
        examples_jsonl=str(examples),
        dry_run=False,
    )

    run(args)

    family_dir = out_dir / "toy"
    assert (family_dir / "examples.csv").exists()
    assert (family_dir / "manifest.json").exists()
    assert (family_dir / "progress.json").exists()
    assert (family_dir / "summary_preview.csv").exists()
    with gzip.open(family_dir / "losses.csv.gz", "rt", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with (family_dir / "examples.csv").open(encoding="utf-8") as f:
        example_rows = list(csv.DictReader(f))

    assert len(example_rows) == 1
    assert len(rows) == 4
    for row in rows:
        raw = (float(row["nll_a"]) + float(row["nll_b"])) - (float(row["nll_full"]) + float(row["nll_base"]))
        useful = float(row["nll_base"]) - float(row["nll_full"])
        assert math.isclose(float(row["raw_token"]), raw)
        assert math.isclose(float(row["useful_token"]), useful)
