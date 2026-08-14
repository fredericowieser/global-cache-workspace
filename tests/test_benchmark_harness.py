from __future__ import annotations

import json
import os

import pytest
import torch

from benchmarks.harness import (
    authenticated_document,
    benchmark_optimizer_step,
    benchmark_processor_operations,
    load_authenticated_json,
    timing_summary,
    verify_authenticated_document,
    write_authenticated_json,
)
from benchmarks.parity import small_cpu_config
from gcw.config import DataConfig, TrainConfig


def tiny_train_config(variant: str = "bt") -> TrainConfig:
    return TrainConfig(
        model=small_cpu_config(variant),
        data=DataConfig(kind="random"),
        batch_size=2,
        sequence_len=7,
        max_steps=2,
        warmup_steps=1,
        save_final=False,
    )


def test_timing_summary_uses_interpolated_percentiles():
    summary = timing_summary([4.0, 1.0, 3.0, 2.0])

    assert summary == {"median_seconds": 2.5, "p10_seconds": 1.3, "p90_seconds": 3.7}


def test_authenticated_benchmark_json_round_trips_and_rejects_tampering(tmp_path):
    path = tmp_path / "benchmark.json"
    payload = {"processor_variant": "bt", "raw_step_seconds": [0.1, 0.2]}

    document = write_authenticated_json(path, payload)

    assert load_authenticated_json(path) == payload
    assert verify_authenticated_document(document) == payload
    altered = json.loads(path.read_text())
    altered["payload"]["processor_variant"] = "bgt"
    with pytest.raises(ValueError, match="integrity"):
        verify_authenticated_document(altered)


def test_authenticated_document_rejects_non_finite_measurements():
    with pytest.raises(ValueError):
        authenticated_document({"raw_step_seconds": [float("nan")]})


def test_cpu_optimizer_step_reports_raw_timings_memory_and_real_token_counts():
    result = benchmark_optimizer_step(tiny_train_config(), torch.device("cpu"), warmup=0, steps=1)

    assert result["kind"] == "complete_training_optimizer_step"
    assert result["processor_variant"] == "bt"
    assert len(result["raw_step_seconds"]) == 1
    assert result["raw_step_seconds"][0] > 0.0
    assert result["peak_memory_bytes"] == 0
    assert result["real_non_padding_tokens_per_step"] == 14
    assert result["real_supervised_tokens_per_step"] == 14
    assert len(result["training_batch_sha256"]) == 64


def test_cpu_processor_suite_exposes_maintained_micro_operations():
    results = benchmark_processor_operations(tiny_train_config("bgt"), torch.device("cpu"), warmup=0, steps=1)

    assert [result["name"] for result in results] == [
        "select_indices",
        "gather_cache",
        "processor_transform",
        "scatter_cache",
        "processor_end_to_end",
    ]
    assert all(result["processor_variant"] == "bgt" for result in results)
    assert all(len(result["raw_step_seconds"]) == 1 for result in results)


@pytest.mark.gpu
@pytest.mark.performance
def test_gpu_benchmark_smoke_is_opt_in_and_never_a_cpu_ci_threshold():
    if os.environ.get("GCW_RUN_GPU_PERF_TESTS") != "1":
        pytest.skip("set GCW_RUN_GPU_PERF_TESTS=1 for the opt-in performance lane")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    result = benchmark_optimizer_step(tiny_train_config("bgt"), torch.device("cuda"), warmup=1, steps=3)

    assert result["median_seconds"] > 0.0
    assert result["peak_memory_bytes"] > 0
