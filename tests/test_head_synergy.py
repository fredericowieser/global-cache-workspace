from argparse import Namespace
import json

import torch

from gcw.analysis.head_synergy import (
    AnalysisExample,
    CheckpointRecord,
    ExampleResult,
    HeadMeanAblator,
    aggregate_normalized_synergy,
    deterministic_masks,
    run,
    score_continuation_nll,
    select_checkpoints,
    summarize,
    synergy_from_losses,
)
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


def test_synergy_formula_matches_fixed_losses():
    full = torch.tensor([2.0, 3.0])
    group_a = torch.tensor([4.0, 5.0])
    group_b = torch.tensor([4.5, 5.5])
    baseline = torch.tensor([6.0, 7.0])

    raw, norm = synergy_from_losses(full, group_a, group_b, baseline)

    assert raw == 0.5
    assert norm == 0.125


def test_aggregate_normalization_happens_after_averaging():
    raw = [1.0, 1.0]
    useful = [1e-9, 2.0]

    norm = aggregate_normalized_synergy(raw, useful)

    assert norm == 1.0 / ((1e-9 + 2.0) / 2.0)


def test_summary_reports_bootstrap_ci_over_examples_not_splits():
    rows = []
    for example_id, raw in enumerate([1.0, 3.0, 5.0]):
        for split_id in range(4):
            rows.append(
                ExampleResult(
                    family="toy",
                    variant="sft",
                    checkpoint="/tmp/checkpoint",
                    task="toy",
                    example_id=example_id,
                    split_id=split_id,
                    mean_synergy_raw=raw,
                    mean_synergy_norm=999999.0,
                    mean_full_nll=1.0,
                    mean_baseline_nll=3.0,
                    n_tokens=2,
                )
            )

    summary = summarize(rows, bootstrap_samples=200, seed=1)[0]

    assert summary.n_examples == 3
    assert summary.n_tokens == 6
    assert summary.mean_synergy_raw == 3.0
    assert summary.mean_synergy_norm == 1.5
    assert summary.raw_ci_low <= summary.mean_synergy_raw <= summary.raw_ci_high


def test_checkpoint_selection_covers_published_sft_bt_bgt_layout():
    records = select_checkpoints("/shared", ("llama32_1b", "qwen3_06b"), ("sft", "bt", "bgt"))

    assert len(records) == 6
    assert records[0] == CheckpointRecord("llama32_1b", "sft", "/shared/checkpoints/sft_backbones/llama32_1b/tf")
    assert records[2] == CheckpointRecord("llama32_1b", "bgt", "/shared/checkpoints/bottleneck_llm/llama32_1b/bgt/final")


def test_head_ablation_all_heads_active_preserves_output():
    torch.manual_seed(0)
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    x = torch.randint(0, 256, (1, 5))
    ablator = HeadMeanAblator(model)
    ablator.head_means = {0: torch.zeros(2, 8)}
    all_heads = {0: torch.ones(2)}

    with torch.no_grad():
        expected = model.forward_full(x)
        with ablator.ablated(all_heads):
            actual = model.forward_full(x)

    assert torch.allclose(actual, expected)


def test_head_ablation_replaces_masked_head_with_mean():
    torch.manual_seed(1)
    model = build_model(tiny_config("tf")).eval()
    model.init_weights()
    x = torch.randint(0, 256, (1, 5))
    ablator = HeadMeanAblator(model)
    ablator.head_means = {0: torch.zeros(2, 8)}
    masks = {0: torch.tensor([0.0, 1.0])}
    seen = []

    ablator.apply(masks)
    handle = model.layers[0].self_attn.o_proj.register_forward_pre_hook(lambda _module, inputs: seen.append(inputs[0].detach().clone()))
    with torch.no_grad():
        model.forward_full(x)
    handle.remove()
    ablator.remove()

    heads = seen[0].reshape(1, 5, 2, 8)
    assert torch.equal(heads[:, :, 0], torch.zeros_like(heads[:, :, 0]))


def test_tiny_tf_bt_bgt_scorers_return_target_token_nlls():
    tokenizer = IntTokenizer(vocab_size=256)
    example = AnalysisExample("toy", 0, "abc", "de")
    for variant in ("tf", "bt", "bgt"):
        model = build_model(tiny_config(variant)).eval()
        model.init_weights()
        nll = score_continuation_nll(model, tokenizer, example, max_length=16, max_target_tokens=8)
        assert nll.shape == (2,)
        assert torch.isfinite(nll).all()


def test_legacy_bgt_scorer_processes_context_cache_boundary(monkeypatch):
    tokenizer = IntTokenizer(vocab_size=256)
    model = build_model(tiny_config("bgt", legacy=True)).eval()
    model.init_weights()
    calls = []

    def fake_forward(cache, *, select_all=False):
        calls.append(select_all)
        return cache

    monkeypatch.setattr(model.cache_processor, "forward", fake_forward)
    example = AnalysisExample("toy", 0, "abc", "de")

    score_continuation_nll(model, tokenizer, example, max_length=16, max_target_tokens=8)

    assert calls == [True]


def test_deterministic_masks_are_reproducible():
    model = build_model(tiny_config("tf")).eval()
    ablator = HeadMeanAblator(model)

    a1, b1 = deterministic_masks(ablator, split_id=3, seed=7, group_fraction=0.5)
    a2, b2 = deterministic_masks(ablator, split_id=3, seed=7, group_fraction=0.5)

    assert torch.equal(a1[0], a2[0])
    assert torch.equal(b1[0], b2[0])


def test_synergy_cli_dry_run_lists_selected_records(capsys):
    args = Namespace(
        root="/shared",
        out_dir="unused",
        tasks=["gsm8k_openmath"],
        families=["llama32_1b"],
        variants=["sft", "bt"],
        limit_checkpoints=None,
        dry_run=True,
    )

    run(args)

    payload = json.loads(capsys.readouterr().out)
    assert [record["variant"] for record in payload["checkpoints"]] == ["sft", "bt"]
    assert payload["tasks"] == ["gsm8k_openmath"]


def test_synthetic_cli_run_writes_summary_outputs(tmp_path):
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
        num_splits=1,
        bootstrap_samples=10,
        norm_eps=1e-6,
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

    assert (out_dir / "toy_sft" / "examples.jsonl").exists()
    assert (out_dir / "toy_sft" / "summary.csv").exists()
    assert (out_dir / "combined_summary.md").exists()
    assert json.loads((out_dir / "progress.json").read_text())["status"] == "completed"
