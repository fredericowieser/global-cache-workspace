import json
import types

import pytest
import torch

from gcw.config import EvalConfig
from gcw.evaluation.lm_eval_adapter import GCWEvalLM
from gcw.evaluation.samples import PartialSamples
from gcw.models.cache_processors.update_metrics import UPDATE_BATCH


def read_samples(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("metrics_enabled", [False, True])
def test_partial_samples_survive_interruption_and_oom(
    tmp_path, monkeypatch, reference, metrics_enabled
):
    monkeypatch.setenv("GCW_EVAL_REFERENCE_GENERATE", "1" if reference else "0")
    synced = []
    from gcw.evaluation import samples

    original_sync = samples.os.fsync

    def sync(fd):
        synced.append(fd)
        original_sync(fd)

    monkeypatch.setattr(samples.os, "fsync", sync)
    directory = tmp_path / "partials"
    completed = []
    observed = []

    class Model(torch.nn.Module):
        config = types.SimpleNamespace(
            sequence_len=32, variant="longshot", diffusion_processor=object()
        )
        cache_processor = types.SimpleNamespace(
            block=types.SimpleNamespace(
                diffusion=types.SimpleNamespace(recurrence_backend="eager")
            )
        )

        def get_device(self):
            return torch.device("cpu")

        def generate(self, ids, **kwargs):
            path = next(directory.glob("*.jsonl"))
            observed.append(len(read_samples(path)) - 1)
            batch = UPDATE_BATCH.get()
            if batch is not None:
                batch.add(1, 0, "value", ids[:, :1].float(), batch.active)
            if ids.shape[0] > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            if len(completed) == 2:
                raise KeyboardInterrupt("interrupted before task completion")
            completed.append(ids[0, 0].item())
            return torch.cat((ids, torch.tensor([[8, 9, 0]])), dim=1)

        generate_fast = generate

    model = Model()
    tokenizer = types.SimpleNamespace(
        eot_id=0,
        eos_token_id=0,
        encode=lambda text, **kwargs: [int(c) for c in text],
        decode=lambda tokens: " ".join(map(str, tokens)),
    )
    config = EvalConfig(
        batch_size="auto",
        max_batch_size=4,
        log_samples=True,
        partial_samples_dir=str(directory),
        cache_update_metrics_path=(
            str(tmp_path / "metrics.jsonl") if metrics_enabled else None
        ),
    )
    lm = GCWEvalLM(model, tokenizer, config)
    lm.sample_protocol = {"checkpoint": "test-checkpoint"}
    requests = [
        types.SimpleNamespace(
            args=(str(i + 1) * 2, {"max_gen_toks": 3}),
            task_name="task",
            doc_id=i,
            idx=0,
            repeats=1,
            doc={"id": f"source-{i}", "answer": "8 9"},
        )
        for i in range(4)
    ]
    with pytest.raises(KeyboardInterrupt):
        lm.generate_until(requests)
    records = read_samples(lm.partial_samples.path)
    assert records[0]["protocol"]["checkpoint"] == "test-checkpoint"
    rows = records[1:]
    assert [row["example_id"] for row in rows] == ["0", "1"]
    assert [row["doc"]["id"] for row in rows] == ["source-0", "source-1"]
    assert [row["arguments"][0] for row in rows] == ["11", "22"]
    assert all(row["request_id"] == "0:0" for row in rows)
    assert all(row["response"] == "8 9" for row in rows)
    assert all(row["generation"]["generated_tokens"] == 2 for row in rows)
    assert all(row["generation"]["termination"] == "stop" for row in rows)
    assert observed == ([0, 1, 2] if reference else [0, 0, 0, 1, 2])
    assert len(synced) == 4  # header, directory, and two successful batches
    assert not (tmp_path / "metrics.jsonl").exists()
    if metrics_enabled:
        for i, row in enumerate(rows):
            assert (
                row["cache_update_metrics"]["schema"]
                == "gcw.cache_update_metrics.v3"
            )
            values = row["cache_update_metrics"]["values"]["prefill:1:0:value"]
            assert values["applications"] == 1
            assert values["sum"] == [float(i + 1)] * 3 + [0.0]
    else:
        assert all("cache_update_metrics" not in row for row in rows)


def test_partial_samples_separate_restarts_ranks_and_calls(
    tmp_path, monkeypatch
):
    config = EvalConfig(log_samples=True, partial_samples_dir=str(tmp_path))
    first = PartialSamples(config, {})
    first.path.write_text(first.path.read_text() + '{"type": "sample"')
    original = first.path.read_bytes()
    second = PartialSamples(config, {})
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    third = PartialSamples(config, {})
    assert len({first.path, second.path, third.path}) == 3
    assert first.path.read_bytes() == original
    assert read_samples(third.path)[0]["rank"] == 1
    request = types.SimpleNamespace(args=("prompt", {}), doc_id=3, doc={})
    metrics = types.SimpleNamespace(path=None)
    for _ in range(2):
        second.start_call("generate_until", 1)
        second.write_batch(
            [request], [0], [("a", "3", "0:0")], ["answer"], metrics
        )
    assert [row["call"] for row in read_samples(second.path)[1:]] == [1, 2]


def test_partial_samples_fail_on_write_error(tmp_path, monkeypatch):
    journal = PartialSamples(
        EvalConfig(log_samples=True, partial_samples_dir=str(tmp_path)), {}
    )
    journal.start_call("generate_until", 1)

    def fail(fd):
        raise OSError("storage unavailable")

    monkeypatch.setattr("gcw.evaluation.samples.os.fsync", fail)
    with pytest.raises(OSError, match="storage unavailable"):
        journal.write_batch(
            [types.SimpleNamespace(args=("prompt", {}))],
            [0],
            [("task", "0", "0:0")],
            ["answer"],
            types.SimpleNamespace(path=None),
        )
