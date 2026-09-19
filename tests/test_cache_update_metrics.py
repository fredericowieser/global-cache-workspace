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
        assert len(row["steps"]) == 4
        for step in row["steps"]:
            assert set(step["metrics"]) == (
                {
                    "latent_kv",
                    "kv",
                    "latent_linear",
                    "linear_kv",
                    "linear_state",
                }
                if linear
                else {"latent_kv", "kv"}
            )
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
        batch, SimpleNamespace(block=block), None, [], batch.active
    )
    initial = torch.zeros(1, 2, 3)
    first = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    second = first + torch.tensor([[[0.0, 0.0, 3.0], [0.0, -1.0, 0.0]]])
    with torch.no_grad():
        observer.step(initial, first, initial, final=False)
        observer.step(first, second, initial, final=True)
        observer.kv(block.out_proj(second))
    row = batch.rows()[0]
    assert row["1:1:kv"]["sum"][1] == pytest.approx(3.0)
    assert row["1:2:kv"]["sum"] == pytest.approx([1.0, 3.0, 2.0, 1.0])
    assert row["1:2:latent_kv"]["sum"] == pytest.approx([1.0, 3.0, 2.0, 1.0])


def test_averages_statistics_across_applications_and_options(tmp_path):
    diagnostics = CacheUpdateMetrics(make_model(), tmp_path / "metrics.jsonl")
    with diagnostics.batch([("task", "one", "option0")]):
        batch = UPDATE_BATCH.get()
        for magnitudes in ([1.0, 3.0], [5.0, 9.0]):
            batch.add(1, 1, "kv", torch.tensor([magnitudes]), batch.active)
    with diagnostics.batch([("task", "one", "option1")]):
        batch = UPDATE_BATCH.get()
        batch.add(1, 1, "kv", torch.tensor([[3.0, 3.0]]), batch.active)
    diagnostics.finish()
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert row["request_count"] == 2
    assert row["steps"][0]["metrics"]["kv"] == pytest.approx(
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
    assert rows["long"]["steps"][0]["metrics"]["kv"]["applications"] == 2


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
    kv_dim = adapter.linear_key_dim + adapter.linear_value_dim
    previous_kv = torch.zeros(2, 2, kv_dim)
    previous = cache.linear_states[0][0]
    expected = []
    expected_kv = []
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
            kv_norms = (kv - previous_kv).norm(dim=-1)
            expected_kv.append(
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
        actual_kv = torch.tensor(
            [row[f"1:{step}:linear_kv"]["sum"] for row in rows]
        )
        torch.testing.assert_close(actual_kv, expected_kv[step - 1])
        if step == 3:
            assert torch.count_nonzero(actual_kv) == 0
        actual = torch.tensor(
            [row[f"1:{step}:linear_state"]["sum"] for row in rows]
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
    assert rows[1]["steps"][0]["metrics"]["kv"]["applications"] == 2


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
    lm.update_metrics.finish()
    rows = {
        row["example_id"]: row
        for row in map(
            json.loads, (tmp_path / "metrics.jsonl").read_text().splitlines()
        )
    }
    assert rows["9"]["request_count"] == 2
    assert rows["9"]["steps"][0]["metrics"]["kv"]["applications"] == 2
    assert rows["8"]["steps"] == []


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
        batch.add(1, 1, "kv", torch.tensor([[1.0], [2.0], [3.0]]), batch.active)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)

    def gather(destination, requests):
        destination[:] = [requests, requests.copy()]

    monkeypatch.setattr(module.dist, "all_gather_object", gather)
    diagnostics.finish()
    row = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert row["request_count"] == 1
    assert row["steps"][0]["metrics"]["kv"]["applications"] == 1
    assert row["steps"][0]["metrics"]["kv"]["mean"] == 1


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
