import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from gcw.cli import build_parser
from gcw.config import DiffusionProcessorConfig, EvalConfig
from gcw.evaluation.update_metrics import (
    CacheUpdateMetrics,
    request_metric_keys,
)
from gcw.models import build_model
from gcw.models.cache_processors.update_metrics import (
    UPDATE_BATCH,
    DiffusionUpdateObservation,
    UpdateBatch,
    outer_difference_norm,
)
from test_diffusion_controls import model_config
from test_longshot_update_gating import (
    _scheduled_model_config,
    _scheduled_cache,
)


def make_model(*, linear=False, method="scheduled", **options):
    config = (
        _scheduled_model_config(**options)
        if linear
        else model_config(**options)
    )
    if method == "fixed":
        config = replace(
            config, diffusion_processor=DiffusionProcessorConfig(num_steps=4)
        )
    model = build_model(config).eval()
    model.init_weights()
    with torch.no_grad():
        model.cache_processor.block.out_proj.weight.normal_(std=0.1)
        model.cache_processor.block.out_proj.bias.normal_(std=0.1)
        adapter = model.cache_processor.block.linear_state_adapter
        if adapter is not None:
            adapter.output_proj.weight.normal_(std=0.1)
            adapter.output_proj.bias.normal_(std=0.1)
    return model


@pytest.mark.parametrize("close", [False, True])
def test_outer_difference_matches_explicit_matrices(close):
    torch.manual_seed(31)
    key = F.normalize(torch.randn(2, 3, 5), dim=-1)
    value = torch.randn(2, 3, 7)
    old_key = key + torch.randn_like(key) * (1e-5 if close else 1)
    old_value = value + torch.randn_like(value) * (1e-5 if close else 1)
    actual = outer_difference_norm(key, value, (old_key, old_value))
    expected = (
        (
            key.double().unsqueeze(-1) * value.double().unsqueeze(-2)
            - old_key.double().unsqueeze(-1) * old_value.double().unsqueeze(-2)
        )
        .flatten(2)
        .norm(dim=-1)
    )
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=1e-7)
    assert torch.equal(
        outer_difference_norm(key, value, (key, value)), torch.zeros(2, 3)
    )
    assert torch.equal(
        outer_difference_norm(torch.zeros_like(key), value, None),
        torch.zeros(2, 3),
    )


@pytest.mark.parametrize("linear", [False, True])
@pytest.mark.parametrize("method", ["fixed", "scheduled"])
def test_observation_preserves_model_and_records_each_step(
    tmp_path, linear, method
):
    model = make_model(linear=linear, method=method)
    tokens = torch.randint(0, 32, (2, 8))
    rng = torch.random.get_rng_state().clone()
    weights = {
        name: tensor.clone() for name, tensor in model.state_dict().items()
    }
    with torch.no_grad():
        expected = model(tokens, tokens)
    diagnostics = CacheUpdateMetrics(model, tmp_path / "updates.jsonl")
    with diagnostics.batch([("task", "a", "0"), ("task", "b", "0")]):
        actual = model(tokens, tokens)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(
        torch.equal(weights[name], tensor)
        for name, tensor in model.state_dict().items()
    )
    metadata = diagnostics.finish()
    assert metadata["record_count"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "updates.jsonl").read_text().splitlines()
    ]
    for row in rows:
        assert row["phase_steps"] == {"prefill": row["steps"], "decoding": []}
        assert len(row["steps"]) == 4
        for step in row["steps"]:
            groups = {"latent_kv", "key", "value"}
            for stream in ("key", "value"):
                groups.update(
                    {
                        f"{stream}_direction_to_original",
                        f"{stream}_cache_direction_from_original",
                    }
                )
            if linear:
                groups.update(
                    {
                        "latent_linear",
                        "linear_key",
                        "linear_value",
                        "linear_state",
                    }
                )
            if step["solver_step"] > 1:
                groups.update(
                    f"{stream}_direction_change"
                    for stream in (
                        ("key", "value", "linear_key", "linear_value")
                        if linear
                        else ("key", "value")
                    )
                )
            assert set(step["metrics"]) == groups
            for metric in step["metrics"].values():
                assert metric["applications"] == (3 if linear else 1)
                assert 0 <= metric["min"] <= metric["mean"] <= metric["max"]
                assert metric["std"] >= 0


def test_consecutive_readouts_include_bias_once_and_exclude_gate():
    block = SimpleNamespace(
        kv_cache_heads=2,
        output_dim=3,
        linear_state_adapter=None,
        out_proj=torch.nn.Linear(3, 3),
    )
    with torch.no_grad():
        block.out_proj.weight.copy_(torch.eye(3))
        block.out_proj.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))
    batch = UpdateBatch(1, "cpu")
    observer = DiffusionUpdateObservation(
        batch,
        SimpleNamespace(block=block, config=SimpleNamespace(update_stream="v")),
        None,
        [],
        batch.active,
    )
    initial = torch.zeros(1, 2, 3)
    first = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    second = first + torch.tensor([[[0.0, 0.0, 3.0], [0.0, -1.0, 0.0]]])
    with torch.no_grad():
        observer.step(initial, first, initial, final=False)
        observer.step(first, second, initial, final=True)
        observer.kv(block.out_proj(second))
    row = batch.rows()[0]
    assert row["prefill:1:1:value"]["sum"][1] == pytest.approx(3.0)
    assert row["prefill:1:2:value"]["sum"] == pytest.approx(
        [1.0, 3.0, 2.0, 1.0]
    )
    assert row["prefill:1:2:latent_kv"]["sum"] == pytest.approx(
        [1.0, 3.0, 2.0, 1.0]
    )


def test_averages_statistics_across_applications_and_options(tmp_path):
    diagnostics = CacheUpdateMetrics(make_model(), tmp_path / "metrics.jsonl")
    with diagnostics.batch([("task", "one", "option0")]):
        batch = UPDATE_BATCH.get()
        for phase, magnitudes in [
            ("prefill", [1.0, 3.0]),
            ("decoding", [5.0, 9.0]),
        ]:
            batch.phase = phase
            batch.add(1, 1, "value", torch.tensor([magnitudes]), batch.active)
    with diagnostics.batch([("task", "one", "option1")]):
        batch = UPDATE_BATCH.get()
        batch.add(1, 1, "value", torch.tensor([[3.0, 3.0]]), batch.active)
    diagnostics.finish()
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert row["request_count"] == 2
    prefill = row["phase_steps"]["prefill"][0]["metrics"]["value"]
    decoding = row["phase_steps"]["decoding"][0]["metrics"]["value"]
    assert prefill == pytest.approx(
        {
            "min": 2,
            "max": 3,
            "mean": 2.5,
            "std": 0.5,
            "applications": 2,
            "heads": 2,
        }
    )
    assert decoding == pytest.approx(
        {
            "min": 5,
            "max": 9,
            "mean": 7,
            "std": 2,
            "applications": 1,
            "heads": 2,
        }
    )
    assert row["steps"][0]["metrics"]["value"] == pytest.approx(
        {
            "min": 3.0,
            "max": 5.0,
            "mean": 4.0,
            "std": 1.0,
            "applications": 3,
            "heads": 2,
        }
    )


def test_padding_failed_batches_and_no_application_rows(tmp_path):
    model = make_model()
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    tokens = torch.randint(0, 32, (2, 12))
    keys = [("task", "short", "0"), ("task", "long", "0")]
    with pytest.raises(RuntimeError, match="failed"):
        with diagnostics.batch(keys):
            model(tokens, tokens)
            raise RuntimeError("failed batch")
    assert not diagnostics.requests
    assert UPDATE_BATCH.get() is None
    with diagnostics.batch(keys, lengths=[4, 12]):
        model(tokens, tokens)
    diagnostics.finish()
    rows = {
        row["example_id"]: row
        for row in map(
            json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()
        )
    }
    assert rows["short"]["steps"] == []
    assert rows["short"]["phase_steps"] == {"prefill": [], "decoding": []}
    assert rows["long"]["steps"][0]["metrics"]["value"]["applications"] == 2


def test_external_iterations_and_value_only_writeback(tmp_path):
    model = make_model(
        iterations=2, iteration_mode="external", update_stream="v"
    )
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    tokens = torch.randint(0, 32, (1, 8))
    with diagnostics.batch([("task", "one", "0")]):
        model(tokens, tokens)
    diagnostics.finish()
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert [
        (step["external_iteration"], step["solver_step"])
        for step in row["steps"]
    ] == [(i, j) for i in (1, 2) for j in range(1, 5)]
    for step in row["steps"]:
        groups = set(step["metrics"])
        assert groups == {
            "latent_kv",
            "value",
            "value_direction_to_original",
            "value_cache_direction_from_original",
        } | ({"value_direction_change"} if step["solver_step"] > 1 else set())


def test_sidecar_resume_authentication_and_target_validation(tmp_path):
    model = make_model()
    path = tmp_path / "metrics.jsonl"
    diagnostics = CacheUpdateMetrics(model, path)
    with diagnostics.batch([("task", "one", "0")]):
        pass
    metadata = diagnostics.finish()
    results = {
        "results": {"task": {}},
        "gcw_runtime": {"task": {"cache_update_metrics": metadata}},
    }
    resumed = CacheUpdateMetrics(model, path, resume=True)
    resumed.validate_resume(results)
    assert len(resumed.completed) == 1
    path.write_text("")
    with pytest.raises(ValueError, match="missing or changed"):
        CacheUpdateMetrics(model, path, resume=True).validate_resume(results)
    with pytest.raises(ValueError, match="require enabled native"):
        CacheUpdateMetrics(torch.nn.Linear(2, 2), tmp_path / "other.jsonl")
    with pytest.raises(ValueError, match="evaluation-only"):
        with CacheUpdateMetrics(model.train(), tmp_path / "train.jsonl").batch(
            [("task", "one", "0")]
        ):
            pass


def test_metrics_reject_compiled_recurrence_before_creating_sidecar(tmp_path):
    model = make_model()
    model.cache_processor.block.diffusion.recurrence_backend = "compile"
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(ValueError, match="require eager diffusion recurrence"):
        CacheUpdateMetrics(model, path)
    assert not path.exists()


def test_metrics_and_scoring_without_distributed_support(tmp_path, monkeypatch):
    import gcw.evaluation.update_metrics as metrics_module
    import gcw.evaluation.scoring as scoring_module

    model = make_model()
    metrics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    unavailable = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(metrics_module, "dist", unavailable)
    monkeypatch.setattr(scoring_module, "dist", unavailable)
    tokens = torch.randint(0, 32, (2, 8))
    batches = [(tokens, tokens)] * 2
    token_bytes = torch.ones(32, dtype=torch.long)
    baseline = scoring_module.evaluate_nll_bpb(model, batches, 2, token_bytes)
    observed = scoring_module.evaluate_nll_bpb(
        model, batches, 2, token_bytes, update_metrics=metrics
    )
    assert observed == baseline
    assert metrics.finish()["record_count"] == 4
    for row in map(
        json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()
    ):
        assert row["phase_steps"] == {"prefill": row["steps"], "decoding": []}


def test_task_digests_preserve_schema_and_resume_across_runs(tmp_path):
    model = make_model()
    path = tmp_path / "metrics.jsonl"
    metrics = CacheUpdateMetrics(model, path)
    with metrics.batch([("b", "2", "0"), ("a", "1", "0")]):
        pass
    first = metrics.finish(["a", "b", "empty"])
    rows = list(map(json.loads, path.read_text().splitlines()))
    assert first["task_sha256"] == {
        task: hashlib.sha256(
            json.dumps(
                [row for row in rows if row["task"] == task],
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        for task in ("a", "b", "empty")
    }
    with metrics.batch([("c", "3", "0")]):
        pass
    second = metrics.finish(["c"])
    results = {
        "results": {task: {} for task in ("a", "b", "c", "empty")},
        "gcw_runtime": {
            "first": {"cache_update_metrics": first},
            "second": {"cache_update_metrics": second},
        },
    }
    resumed = CacheUpdateMetrics(model, path, resume=True)
    resumed.validate_resume(results)
    assert resumed.completed == metrics.completed


@pytest.mark.parametrize("failure", ["before", "after", None])
def test_receipt_publication_preserves_only_committed_sidecar(
    tmp_path, monkeypatch, failure
):
    import gcw.evaluation.update_metrics as module

    model = make_model()
    path = tmp_path / "metrics.jsonl"
    out = tmp_path / "receipt.json"
    metrics = CacheUpdateMetrics(model, path)
    with metrics.batch([("task", "one", "0")]):
        pass

    def write_receipt(target, payload, **kwargs):
        assert path.exists()
        assert kwargs == {"exclusive": True, "allow_nan": False}
        if failure == "before":
            raise OSError("receipt publication failed")
        target.write_text(json.dumps(payload))
        if failure == "after":
            raise OSError("directory synchronization failed")

    monkeypatch.setattr(module, "atomic_write_json", write_receipt)
    receipt = {"result": {"nll": 1.0}}
    if failure is None:
        metrics.write_receipt(out, receipt)
    else:
        with pytest.raises(OSError):
            metrics.write_receipt(out, receipt)
    assert path.exists() == out.exists() == (failure != "before")
    if failure == "before":
        retry = CacheUpdateMetrics(model, path)
        with retry.batch([("task", "one", "0")]):
            pass
        failure = None
        retry.write_receipt(out, {"result": {"nll": 1.0}})
    assert (
        json.loads(out.read_text())["cache_update_metrics"]["record_count"] == 1
    )


def test_receipt_failure_does_not_delete_existing_sidecar(
    tmp_path, monkeypatch
):
    import gcw.evaluation.update_metrics as module

    model = make_model()
    path = tmp_path / "metrics.jsonl"
    CacheUpdateMetrics(model, path).finish()
    metrics = CacheUpdateMetrics(model, path, resume=True)

    def fail(*args, **kwargs):
        raise OSError("receipt publication failed")

    monkeypatch.setattr(module, "atomic_write_json", fail)
    with pytest.raises(OSError):
        metrics.write_receipt(tmp_path / "receipt.json", {})
    assert path.exists()


def test_existing_receipt_rejected_before_publishing_sidecar(tmp_path):
    out = tmp_path / "receipt.json"
    out.write_text('{"result": "existing"}')
    path = tmp_path / "metrics.jsonl"
    metrics = CacheUpdateMetrics(make_model(), path)
    with pytest.raises(FileExistsError, match="receipt.json"):
        metrics.write_receipt(out, {})
    assert not path.exists()
    assert out.read_text() == '{"result": "existing"}'


@pytest.mark.parametrize("reference", ["none", "other", "same", "malformed"])
def test_concurrent_receipt_collision_keeps_only_referenced_or_unknown_sidecar(
    tmp_path, monkeypatch, reference
):
    import gcw.evaluation.update_metrics as module

    model = make_model()
    path = tmp_path / "metrics.jsonl"
    out = tmp_path / "receipt.json"
    metrics = CacheUpdateMetrics(model, path)
    existing = {"result": "concurrent"}
    if reference in {"other", "same"}:
        existing["cache_update_metrics"] = {
            "path": str(
                path if reference == "same" else tmp_path / "other.jsonl"
            )
        }
    contents = (
        "incomplete JSON" if reference == "malformed" else json.dumps(existing)
    )
    original_write = module.atomic_write_json

    def collide(target, payload, **kwargs):
        assert path.exists()
        target.write_text(contents)
        original_write(target, payload, **kwargs)

    monkeypatch.setattr(module, "atomic_write_json", collide)
    with pytest.raises(FileExistsError, match="receipt.json"):
        metrics.write_receipt(out, {})
    assert out.read_text() == contents
    assert path.exists() == (reference in {"same", "malformed"})
    if reference in {"none", "other"}:
        retry = CacheUpdateMetrics(model, path)
        with pytest.raises(FileExistsError, match="receipt.json"):
            retry.write_receipt(out, {})
        assert not path.exists()


def test_cli_and_request_identity():
    parser = build_parser()
    args = parser.parse_args(
        [
            "eval",
            "paper7",
            "--checkpoint",
            "checkpoint",
            "--cache-update-metrics",
        ]
    )
    assert args.cache_update_metrics == "auto"
    requests = [
        SimpleNamespace(task_name="task", doc_id=9, idx=i) for i in (0, 1, 0)
    ]
    assert request_metric_keys(requests) == [
        ("task", "9", "0:0"),
        ("task", "9", "1:0"),
        ("task", "9", "0:1"),
    ]


@pytest.mark.parametrize("writer", ["outer_product", "gated_delta"])
def test_linear_contributions_match_actual_writer_matrices(writer):
    model = make_model(
        linear=True,
        linear_state_update_mode=writer,
        live_rank1_linear_writer=writer == "outer_product",
    )
    processor = model.cache_processor
    cache = _scheduled_cache()
    batch = UpdateBatch(2, "cpu")
    observer = DiffusionUpdateObservation(
        batch, processor, cache, processor.linear_state_specs, batch.active
    )
    dim = processor.block.linear_state_adapter.output_dim
    proposals = [torch.randn(2, 2, dim), torch.randn(2, 2, dim)]
    unchanged_kv = proposals[-1].clone()
    if writer == "gated_delta":
        unchanged_kv[..., -2:] += 5
    proposals.append(unchanged_kv)
    adapter = processor.block.linear_state_adapter
    assert adapter.linear_key_dim != adapter.linear_value_dim
    kv_dim = adapter.linear_key_dim + adapter.linear_value_dim
    previous_kv = torch.zeros(2, 2, kv_dim)
    previous = cache.linear_states[0][0]
    expected = []
    expected_kv = {"linear_key": [], "linear_value": []}
    for step, raw in enumerate(proposals, 1):
        observer.step_index = step
        with torch.no_grad():
            oracle_cache = copy.deepcopy(cache)
            processor._scatter_linear_state_updates(
                oracle_cache, raw, processor.linear_state_specs
            )
            proposed = oracle_cache.linear_states[0][0]
            norms = (proposed - previous).flatten(2).norm(dim=-1)
            expected.append(
                torch.stack(
                    [
                        norms.min(1).values,
                        norms.max(1).values,
                        norms.mean(1),
                        norms.std(1, correction=0),
                    ],
                    1,
                )
            )
            kv = raw[..., :kv_dim]
            for name, channels in [
                ("linear_key", slice(None, adapter.linear_key_dim)),
                ("linear_value", slice(adapter.linear_key_dim, None)),
            ]:
                kv_norms = (kv - previous_kv)[..., channels].norm(dim=-1)
                expected_kv[name].append(
                    torch.stack(
                        [
                            kv_norms.min(1).values,
                            kv_norms.max(1).values,
                            kv_norms.mean(1),
                            kv_norms.std(1, correction=0),
                        ],
                        1,
                    )
                )
            observer.linear(raw)
            previous = proposed
            previous_kv = kv
    rows = batch.rows()
    for step in (1, 2, 3):
        for name in expected_kv:
            actual_kv = torch.tensor(
                [row[f"prefill:1:{step}:{name}"]["sum"] for row in rows]
            )
            torch.testing.assert_close(actual_kv, expected_kv[name][step - 1])
            if step == 3:
                assert torch.count_nonzero(actual_kv) == 0
                assert all(
                    f"prefill:1:{step}:{name}_direction_change" not in row
                    for row in rows
                )
            elif step == 2:
                channels = (
                    slice(None, adapter.linear_key_dim)
                    if name == "linear_key"
                    else slice(adapter.linear_key_dim, kv_dim)
                )
                distances = 1 - F.cosine_similarity(
                    (proposals[1] - proposals[0])[..., channels],
                    proposals[0][..., channels],
                    dim=-1,
                )
                actual_direction = torch.tensor(
                    [
                        row[f"prefill:1:2:{name}_direction_change"]["sum"][2]
                        for row in rows
                    ]
                )
                torch.testing.assert_close(actual_direction, distances.mean(1))
        actual = torch.tensor(
            [row[f"prefill:1:{step}:linear_state"]["sum"] for row in rows]
        )
        torch.testing.assert_close(
            actual, expected[step - 1], atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize(
    "gate_type", ["contextual_scalar", "contextual_vector"]
)
def test_pre_gate_metrics_do_not_depend_on_write_gate(tmp_path, gate_type):
    model = make_model(linear=True, update_gate_type=gate_type)
    other = copy.deepcopy(model)
    with torch.no_grad():
        for block in [other.cache_processor.block]:
            block.out_proj.bias[block.output_dim :].add_(3)
            adapter = block.linear_state_adapter
            adapter.output_proj.bias[adapter.output_dim :].add_(3)
    tokens = torch.randint(0, 32, (1, 4))
    records = []
    for i, candidate in enumerate([model, other]):
        diagnostics = CacheUpdateMetrics(candidate, tmp_path / f"{i}.jsonl")
        with diagnostics.batch([("task", "one", "0")]):
            candidate(tokens, tokens)
        diagnostics.finish()
        records.append(json.loads((tmp_path / f"{i}.jsonl").read_text()))
    assert records[0] == records[1]


@pytest.mark.parametrize("fast", [False, True])
def test_generation_excludes_finished_rows(tmp_path, monkeypatch, fast):
    model = make_model()
    method = "forward_chunk_static" if fast else "forward_chunk"
    original = getattr(model.backbone, method)
    predictions = [[0, 5], [7, 6], [7, 0]]
    calls = 0

    def forward(*args, **kwargs):
        nonlocal calls
        logits, cache = original(*args, **kwargs)
        logits = torch.full_like(logits, -100)
        for row, token in enumerate(predictions[min(calls, 2)]):
            logits[row, -1, token] = 100
        calls += 1
        return logits, cache

    monkeypatch.setattr(model.backbone, method, forward)
    model.config.cache_processor.generation_process_every = 1
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    with diagnostics.batch([("task", "first", "0"), ("task", "second", "0")]):
        generate = model.generate_fast if fast else model.generate
        output = generate(
            torch.ones(2, 2, dtype=torch.long), 5, eos_token_id=[0]
        )
    assert output[:, 2:].tolist() == [[0, 0, 0], [5, 6, 0]]
    diagnostics.finish()
    rows = list(
        map(json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines())
    )
    assert rows[0]["steps"] == []
    assert rows[1]["steps"][0]["metrics"]["value"]["applications"] == 2
    assert rows[0]["phase_steps"] == {"prefill": [], "decoding": []}
    for phase in ("prefill", "decoding"):
        assert (
            rows[1]["phase_steps"][phase][0]["metrics"]["value"]["applications"]
            == 1
        )


def test_lm_eval_options_sorting_and_oom_retry(tmp_path, monkeypatch):
    from gcw.evaluation.lm_eval_adapter import GCWEvalLM
    from gcw.data.tokenizers import IntTokenizer

    model = make_model()

    class Tokenizer(IntTokenizer):
        def encode(self, text, **kwargs):
            return [int(char) for char in text]

    lm = GCWEvalLM(
        model,
        Tokenizer(),
        EvalConfig(
            log_samples=True,
            partial_samples_dir=str(tmp_path / "partials"),
            batch_size="auto",
            max_batch_size=4,
            cache_update_metrics_path=str(tmp_path / "metrics.jsonl"),
        ),
    )
    requests = [
        SimpleNamespace(
            task_name="task",
            doc_id=doc,
            idx=idx,
            args=(context, "2"),
            repeats=1,
        )
        for doc, idx, context in [
            (9, 0, "11111"),
            (8, 0, "111"),
            (9, 1, "111111"),
        ]
    ]
    original = model.forward

    def fail_large(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[0].shape[0] > 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return value

    monkeypatch.setattr(model, "forward", fail_large)
    assert len(lm.loglikelihood(requests)) == 3
    partial = [
        json.loads(line)
        for line in lm.partial_samples.path.read_text().splitlines()
    ][1:]
    assert [(row["example_id"], row["request_id"]) for row in partial] == [
        ("8", "0:0"), ("9", "0:0"), ("9", "1:0")
    ]
    assert all(row["request_type"] == "loglikelihood" for row in partial)
    assert all("cache_update_metrics" in row for row in partial)
    lm.update_metrics.finish()
    rows = {
        row["example_id"]: row
        for row in map(
            json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()
        )
    }
    assert rows["9"]["request_count"] == 2
    assert rows["9"]["steps"][0]["metrics"]["value"]["applications"] == 2
    assert rows["8"]["steps"] == []
    for row in rows.values():
        assert row["phase_steps"] == {"prefill": row["steps"], "decoding": []}


def test_distributed_padding_duplicates_are_not_counted(tmp_path, monkeypatch):
    import gcw.evaluation.update_metrics as module

    diagnostics = CacheUpdateMetrics(make_model(), tmp_path / "metrics.jsonl")
    requests = [
        SimpleNamespace(task_name="task", doc_id=0, idx=0, repeats=1)
    ] * 3
    keys = request_metric_keys(requests)
    assert len(set(keys)) == 1
    with diagnostics.batch(keys):
        batch = UPDATE_BATCH.get()
        batch.add(
            1, 1, "value", torch.tensor([[1.0], [2.0], [3.0]]), batch.active
        )
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)

    def gather(destination, requests):
        destination[:] = [requests, requests.copy()]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    diagnostics.finish()
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert row["request_count"] == 1
    assert row["steps"][0]["metrics"]["value"]["applications"] == 1
    assert row["steps"][0]["metrics"]["value"]["mean"] == 1


def test_answer_nll_real_checkpoint_sidecar(tmp_path, monkeypatch):
    import os
    import gcw.runtime as runtime
    import gcw.training.checkpoint.safetensors as checkpoint_io
    import test_answer_nll as fixtures

    if os.name == "nt":
        # The existing checkpoint/capture helpers assume POSIX file descriptors.
        monkeypatch.setattr(checkpoint_io, "_fsync_tree", lambda path: None)
        monkeypatch.setattr(checkpoint_io, "fsync_directory", lambda path: None)
        monkeypatch.setattr(runtime, "fsync_directory", lambda path: None)
        original_open = os.open
        monkeypatch.setattr(
            os,
            "open",
            lambda path, flags, *args, **kwargs: original_open(
                path, flags | os.O_BINARY, *args, **kwargs
            ),
        )

    original = fixtures.build_model

    def build_longshot(config):
        config.variant = "longshot"
        return original(config)

    monkeypatch.setattr(fixtures, "build_model", build_longshot)
    checkpoint = tmp_path / "checkpoint"
    fixtures.native_checkpoint(
        checkpoint,
        processor=True,
        diffusion=DiffusionProcessorConfig(training_method="scheduled"),
    )
    data = fixtures.dataset(tmp_path / "data.jsonl")
    baseline = fixtures.run(checkpoint, data, tmp_path / "baseline.json")
    path = tmp_path / "metrics.jsonl"
    observed = fixtures.run(
        checkpoint,
        data,
        tmp_path / "observed.json",
        cache_update_metrics_path=str(path),
    )
    assert observed["result"] == baseline["result"]
    assert (
        observed["scheduled_field_evaluations"]
        == baseline["scheduled_field_evaluations"]
    )
    assert observed["cache_update_metrics"]["record_count"] == 2
    for row in map(json.loads, path.read_text().splitlines()):
        assert row["phase_steps"] == {"prefill": row["steps"], "decoding": []}
    assert len(path.read_text().splitlines()) == 2
    assert (
        len(
            fixtures.load_answer_nll_rows(
                tmp_path / "observed.json",
                expected_sha256=fixtures.file_sha256(
                    tmp_path / "observed.json"
                ),
            )
        )
        == 2
    )


def test_observation_preserves_cache_and_reuses_final_readouts(tmp_path):
    model = make_model(linear=True)
    processor = model.cache_processor
    cache = _scheduled_cache()
    calls = {"kv": 0, "linear": 0}

    def count(name):
        def hook(*_args):
            calls[name] += 1

        return hook

    handles = [
        processor.block.out_proj.register_forward_hook(count("kv")),
        processor.block.linear_state_adapter.output_proj.register_forward_hook(
            count("linear")
        ),
    ]
    with torch.no_grad():
        expected = processor(copy.deepcopy(cache))
    assert calls == {"kv": 1, "linear": 1}
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    with diagnostics.batch([("task", "0", "0"), ("task", "1", "0")]):
        observed = processor(copy.deepcopy(cache))
    assert calls == {"kv": 5, "linear": 5}
    for actual, reference in zip(
        observed.keys + observed.values, expected.keys + expected.values
    ):
        assert torch.equal(actual, reference)
    for layer in expected.linear_states:
        for actual, reference in zip(
            observed.linear_states[layer], expected.linear_states[layer]
        ):
            assert torch.equal(actual, reference)
    for handle in handles:
        handle.remove()


def test_fixed_diffusion_fullgraph_remains_compilable():
    from gcw.models.cache_processors.diffusion import TaskDirectedDiffusion

    module = TaskDirectedDiffusion(
        4, DiffusionProcessorConfig(num_steps=2)
    ).eval()
    state = torch.randn(1, 2, 4)

    def field(value, _time):
        return value.tanh()

    compiled = torch.compile(
        module.deterministic_recurrence, backend="eager", fullgraph=True
    )
    with torch.no_grad():
        torch.testing.assert_close(
            compiled(state, state, field),
            module.deterministic_recurrence(state, state, field),
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cosine_distances_and_undefined_vectors(dtype):
    left = torch.tensor(
        [
            [1, 0],
            [0, 2],
            [-3, 0],
            [0, 0],
            [1e-13, 0],
            [2e-12, 0],
            [1, 0],
            [1, 0],
            [0.3, 0.7],
        ],
        dtype=dtype,
    ).unsqueeze(1)
    right = torch.tensor(
        [
            [2, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 0],
            [1e-13, 0],
            [0.3, 0.7],
        ],
        dtype=dtype,
    ).unsqueeze(1)
    batch = UpdateBatch(len(left), "cpu")
    observer = DiffusionUpdateObservation(
        batch, SimpleNamespace(block=None), None, [], batch.active
    )
    observer.step_index = 2
    observer.direction("key_direction_change", left, right)
    rows = batch.rows()
    for index, expected in [(0, 0), (1, 1), (2, 2), (5, 0), (8, 0)]:
        metric = rows[index]["prefill:1:2:key_direction_change"]
        assert metric["sum"] == pytest.approx([expected] * 3 + [0], abs=1e-7)
        assert metric["valid_head_observations"] == 1
        assert metric["applications"] == metric["heads"] == 1
    for index in (3, 4, 6, 7):
        assert rows[index] == {}


@pytest.mark.parametrize("value_only", [False, True])
def test_attention_contributions_and_original_comparisons(value_only):
    original = torch.tensor([[[2.0, 0.0, 0.0, 3.0]]])
    proposals = [
        torch.tensor([[[1.0, 0.0, 0.0, -2.0]]]),
        torch.tensor([[[1.0, 2.0, 4.0, -2.0]]]),
    ]
    processor = SimpleNamespace(
        block=None,
        config=SimpleNamespace(update_stream="v" if value_only else "kv"),
    )
    batch = UpdateBatch(1, "cpu")
    observer = DiffusionUpdateObservation(
        batch, processor, None, [], batch.active, original
    )
    for index, proposal in enumerate(proposals, 1):
        observer.step_index = index
        observer.kv(proposal[..., 2:] if value_only else proposal)
    row = batch.rows()[0]
    for name, channels in [("key", slice(0, 2)), ("value", slice(2, 4))]:
        if value_only and name == "key":
            assert not any(":key" in group for group in row)
            continue
        previous = torch.zeros_like(proposals[0][..., channels])
        for step, proposal in enumerate(proposals, 1):
            projected = proposal[..., channels]
            contribution = projected - previous
            baseline = original[..., channels]
            magnitude = contribution.norm().item()
            assert row[f"prefill:1:{step}:{name}"]["sum"] == pytest.approx(
                [magnitude] * 3 + [0]
            )
            for suffix, vector in [
                ("direction_to_original", contribution),
                ("cache_direction_from_original", baseline + projected),
            ]:
                expected = (
                    1 - F.cosine_similarity(vector, baseline, dim=-1).item()
                )
                assert row[f"prefill:1:{step}:{name}_{suffix}"][
                    "sum"
                ] == pytest.approx([expected] * 3 + [0])
            previous = projected
        assert f"prefill:1:1:{name}_direction_change" not in row
        assert row[f"prefill:1:2:{name}_direction_change"][
            "sum"
        ] == pytest.approx([1, 1, 1, 0])
        cumulative_distance = (
            1
            - F.cosine_similarity(
                proposals[1][..., channels], proposals[0][..., channels], dim=-1
            ).item()
        )
        assert cumulative_distance != pytest.approx(1)


def test_direction_aggregation_uses_defined_heads_and_applications(tmp_path):
    metrics = CacheUpdateMetrics(make_model(), tmp_path / "metrics.jsonl")
    for request, comparisons in [
        (
            "0",
            [
                ([0.0, 2.0, 99.0], [True, True, False]),
                ([1.0, 99.0, 99.0], [True, False, False]),
                ([99.0, 99.0, 99.0], [False, False, False]),
            ],
        ),
        ("1", [([2.0, 99.0, 99.0], [True, False, False])]),
    ]:
        with metrics.batch(
            [("task", "one", request), ("task", "finished", request)]
        ):
            batch = UPDATE_BATCH.get()
            batch.active[1] = False
            for index, (distances, valid) in enumerate(comparisons):
                batch.phase = (
                    "prefill" if request == "0" and index == 0 else "decoding"
                )
                batch.add(
                    1,
                    2,
                    "key_direction_change",
                    torch.tensor([distances, distances]),
                    batch.active,
                    torch.tensor([valid, valid]),
                )
            batch.add(
                1,
                2,
                "value_direction_change",
                torch.zeros(2, 3),
                batch.active,
                torch.zeros(2, 3, dtype=torch.bool),
            )
    metrics.finish()
    rows = {
        row["example_id"]: row
        for row in map(
            json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()
        )
    }
    assert rows["finished"]["steps"] == []
    assert rows["one"]["schema"] == "gcw.cache_update_metrics.v3"
    groups = rows["one"]["steps"][0]["metrics"]
    assert set(groups) == {"key_direction_change"}
    assert groups["key_direction_change"] == pytest.approx(
        {
            "min": 1,
            "max": 5 / 3,
            "mean": 4 / 3,
            "std": 1 / 3,
            "heads": 3,
            "applications": 3,
            "valid_head_observations": 4,
        }
    )
    assert rows["one"]["request_count"] == 2
    for phase, expected in [
        (
            "prefill",
            {"min": 0, "max": 2, "mean": 1, "std": 1, "applications": 1},
        ),
        (
            "decoding",
            {"min": 1.5, "max": 1.5, "mean": 1.5, "std": 0, "applications": 2},
        ),
    ]:
        groups = rows["one"]["phase_steps"][phase][0]["metrics"]
        assert set(groups) == {"key_direction_change"}
        assert groups["key_direction_change"] == pytest.approx(
            {
                **expected,
                "heads": 3,
                "valid_head_observations": 2,
            }
        )


@pytest.mark.parametrize("schema", ["v1", "v2"])
def test_old_sidecar_and_metadata_cannot_resume(tmp_path, schema):
    path = tmp_path / "metrics.jsonl"
    model = make_model()
    path.write_text(
        json.dumps({"schema": f"gcw.cache_update_metrics.{schema}"}) + "\n"
    )
    with pytest.raises(ValueError, match="incompatible"):
        CacheUpdateMetrics(model, path, resume=True)
    path.write_text("")
    with pytest.raises(ValueError, match="resume policy differs"):
        CacheUpdateMetrics(model, path, resume=True).validate_resume(
            {
                "gcw_runtime": {
                    "task": {
                        "cache_update_metrics": {
                            "schema": f"gcw.cache_update_metrics.{schema}",
                            "path": str(path),
                        }
                    }
                },
            }
        )


@pytest.mark.parametrize("unrope", [False, True])
def test_original_frontier_coordinates_and_external_iteration_reset(
    tmp_path, monkeypatch, unrope
):
    from gcw.models.cache_processors.ops import rotate_cache_keys
    from gcw.models.cache_processors.update_metrics import UPDATE_OBSERVER

    model = make_model(
        linear=True, unrope_keys=unrope, iterations=2, iteration_mode="external"
    )
    processor = model.cache_processor
    cache = _scheduled_cache()
    incoming_keys = cache.keys[0][:, :, -1:, :].clone()
    if unrope:
        incoming_keys = rotate_cache_keys(
            incoming_keys.unsqueeze(0),
            torch.full((2, 1), 2),
            theta=processor.rope_theta,
            rotary_style=processor.rotary_style,
            rotary_dim=processor.rotary_dim,
            inverse=True,
        ).squeeze(0)
        assert not torch.allclose(incoming_keys, cache.keys[0][:, :, -1:, :])
    expected_frontier = torch.cat(
        [incoming_keys.squeeze(2), cache.values[0][:, :, -1, :]], dim=-1
    )
    frontiers, outputs, proposals = [], [], []

    def capture_frontier(_module, args):
        observer = UPDATE_OBSERVER.get()
        merged = args[0]
        assert observer.original.data_ptr() == merged[:, :, -1, :].data_ptr()
        assert not observer.original.requires_grad
        assert observer.previous_kv is observer.previous_kv_change is None
        assert (
            observer.previous_linear_raw
            is observer.previous_linear_change
            is None
        )
        frontiers.append(observer.original.clone())
        proposals.append([])

    def capture_output(_module, _args, output):
        outputs.append(output[0].clone())

    observe = DiffusionUpdateObservation.kv

    def capture_proposal(observer, raw):
        proposals[-1].append(raw.detach().float().clone())
        observe(observer, raw)

    monkeypatch.setattr(DiffusionUpdateObservation, "kv", capture_proposal)
    handles = [
        processor.block.register_forward_pre_hook(capture_frontier),
        processor.block.register_forward_hook(capture_output),
    ]
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    try:
        with diagnostics.batch([("task", "0", "0"), ("task", "1", "0")]):
            processor(cache)
    finally:
        for handle in handles:
            handle.remove()
    assert len(frontiers) == 2
    torch.testing.assert_close(frontiers[0], expected_frontier)
    torch.testing.assert_close(frontiers[1], outputs[0])
    diagnostics.finish()
    rows = list(
        map(json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines())
    )
    for iteration, (frontier, readouts) in enumerate(
        zip(frontiers, proposals), 1
    ):
        previous = torch.zeros_like(readouts[0])
        assert len(readouts) == 4
        for step, proposal in enumerate(readouts, 1):
            for name, channels in [
                ("key", slice(0, 4)),
                ("value", slice(4, 8)),
            ]:
                baseline = frontier[..., channels]
                for suffix, vector in [
                    (
                        "direction_to_original",
                        (proposal - previous)[..., channels],
                    ),
                    (
                        "cache_direction_from_original",
                        baseline + proposal[..., channels],
                    ),
                ]:
                    expected = 1 - F.cosine_similarity(vector, baseline, dim=-1)
                    for index, row in enumerate(rows):
                        metrics = row["steps"][(iteration - 1) * 4 + step - 1][
                            "metrics"
                        ]
                        actual = metrics[f"{name}_{suffix}"]
                        assert actual["mean"] == pytest.approx(
                            expected[index].item(), abs=2e-7
                        )
                        assert actual["valid_head_observations"] == 1
                        if step == 1:
                            assert f"{name}_direction_change" not in metrics
            previous = proposal


def test_phase_accumulators_remain_bounded_and_transfer_only_at_end(
    monkeypatch,
):
    batch = UpdateBatch(2, "cpu")
    values = torch.tensor([[1.0, 3.0], [5.0, 9.0]])
    original_cpu = torch.Tensor.cpu
    transfers = []

    def cpu(tensor, *args, **kwargs):
        transfers.append(tensor.shape)
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    for _ in range(100):
        for phase in ("prefill", "decoding"):
            batch.phase = phase
            batch.add(1, 1, "value", values, batch.active)
    assert not transfers
    assert len(batch.totals) == 2
    assert (
        sum(
            t.numel()
            for entry in batch.totals.values()
            for t in entry
            if isinstance(t, torch.Tensor)
        )
        == 20
    )
    for row in batch.rows():
        assert set(row) == {"prefill:1:1:value", "decoding:1:1:value"}
        assert all(value["applications"] == 100 for value in row.values())
    assert len(transfers) == 3


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("linear", [False, True])
def test_generation_combined_matches_unsplit_measurements(
    tmp_path, monkeypatch, fast, linear
):
    model = make_model(linear=linear, iterations=2, iteration_mode="external")
    model.config.cache_processor.generation_process_every = 1
    generate = model.generate_fast if fast else model.generate
    tokens = torch.ones(2, 4, dtype=torch.long)
    baseline = generate(tokens, 4)
    diagnostics = CacheUpdateMetrics(model, tmp_path / "metrics.jsonl")
    unsplit = UpdateBatch(2, "cpu")
    readouts = []
    original_kv = DiffusionUpdateObservation.kv

    def kv(observer, raw):
        if observer.active.any():
            readouts.append(
                (observer.batch.phase, observer.iteration, observer.step_index)
            )
        return original_kv(observer, raw)

    monkeypatch.setattr(DiffusionUpdateObservation, "kv", kv)
    with diagnostics.batch([("task", "0", "0"), ("task", "1", "0")]):
        batch = UPDATE_BATCH.get()
        add = batch.add

        def record(*args, **kwargs):
            unsplit.add(*args, **kwargs)
            add(*args, **kwargs)

        monkeypatch.setattr(batch, "add", record)
        actual = generate(tokens, 4)
    assert torch.equal(actual, baseline)
    diagnostics.finish()
    rows = list(
        map(json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines())
    )
    assert {phase for phase, _, _ in readouts} == {"prefill", "decoding"}
    for row, reference in zip(rows, unsplit.rows(), strict=True):
        assert all(row["phase_steps"].values())
        for step in row["steps"]:
            for name, metric in step["metrics"].items():
                key = f"prefill:{step['external_iteration']}:{step['solver_step']}:{name}"
                expected = reference[key]
                count = expected["applications"]
                assert metric == pytest.approx(
                    {
                        **dict(
                            zip(
                                ("min", "max", "mean", "std"),
                                [value / count for value in expected["sum"]],
                            )
                        ),
                        "applications": count,
                        "heads": expected["heads"],
                        **(
                            {
                                "valid_head_observations": expected[
                                    "valid_head_observations"
                                ]
                            }
                            if "valid_head_observations" in expected
                            else {}
                        ),
                    },
                    abs=1e-6,
                )
        applications = sum(
            step["metrics"]["value"]["applications"] for step in row["steps"]
        )
        assert len(readouts) == applications
