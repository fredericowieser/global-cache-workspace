from __future__ import annotations

import os
from pathlib import Path
import time

import torch

from gcw.runtime import base_dir
from gcw.evaluation.bpb import (
    run_fineweb,  # noqa: F401  (re-exported for the CLI)
)
from gcw.evaluation.lm_eval_adapter import (
    GCWEvalLM,
    HAS_LM_EVAL,
    ProcessorBypass,  # noqa: F401  (re-exported for benchmarks/scripts)
    _distributed_eval_state,
    _eval_device,
    _load_eval_target,
)
from gcw.evaluation.progress import (
    _empty_eval_results,
    _load_partial_results,
    _new_progress,
    _update_progress,
    _write_partial_results,
    _write_progress,
    is_main_process,  # noqa: F401  (re-exported for the CLI)
)
from gcw.evaluation.tasks import (
    GENERAL_RUNS,
    MATH_RUNS,
    TASK_FEWSHOT,
    _filter_skipped_tasks,
    _normalize_until,  # noqa: F401  (re-exported for tests)
    _run_is_complete,
    _should_apply_chat_template,
    _trim_until,  # noqa: F401  (re-exported for tests)
)


def run_all(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        [*GENERAL_RUNS, *MATH_RUNS],
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
    )


def run_general(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        GENERAL_RUNS,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
    )


def run_math(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        MATH_RUNS,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
    )


def run_tasks(
    tasks: list[str],
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
) -> dict:
    runs = [([task], TASK_FEWSHOT.get(task, 0)) for task in tasks]
    return run_task_groups(
        checkpoint,
        runs,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
    )


def run_task_groups(
    checkpoint: str | Path | None,
    runs: list[tuple[list[str], int]],
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
) -> dict:
    if not HAS_LM_EVAL:
        raise ImportError("Install the eval extra: uv sync --extra eval")
    runs = _filter_skipped_tasks(runs)
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(base_dir() / "hf_datasets_eval_v3")
    )
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict

    dist_state, local_rank, world_size = _distributed_eval_state()
    device = _eval_device(local_rank)
    model, tokenizer = _load_eval_target(
        checkpoint=checkpoint,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        device=device,
        dtype=dtype,
        generation_process_every=generation_process_every,
    )
    lm = GCWEvalLM(
        model,
        tokenizer,
        batch_size=batch_size,
    )
    processor = getattr(model, "cache_processor", None)
    if prompt_mode not in {"auto", "plain", "thinking_off", "thinking_on"}:
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    lm.prompt_mode = prompt_mode
    apply_chat_template = _should_apply_chat_template(tokenizer, prompt_mode)
    task_manager = TaskManager(
        include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR"
    )
    merged = _load_partial_results(runs) or _empty_eval_results()
    _write_progress(_new_progress(runs), merged)
    for run_idx, (tasks, fewshot) in enumerate(runs):
        progress = _update_progress(run_idx, "running", runs, merged)
        if _run_is_complete(tasks, merged):
            _write_progress(
                _update_progress(
                    run_idx,
                    "completed",
                    runs,
                    merged,
                    progress=progress,
                    skipped=True,
                ),
                merged,
            )
            continue
        if world_size > 1 and dist_state is not None:
            with dist_state.main_process_first():
                if local_rank == 0:
                    get_task_dict(tasks, task_manager)
        started = time.monotonic()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        try:
            out = lm_eval.simple_evaluate(
                model=lm,
                tasks=tasks,
                num_fewshot=fewshot,
                limit=limit,
                bootstrap_iters=0,
                log_samples=False,
                verbosity="ERROR",
                task_manager=task_manager,
                confirm_run_unsafe_code=True,
                apply_chat_template=apply_chat_template,
            )
        except BaseException as exc:
            _write_progress(
                _update_progress(
                    run_idx,
                    "failed",
                    runs,
                    merged,
                    progress=progress,
                    error=str(exc),
                ),
                merged,
            )
            raise
        if out is None:
            continue
        elapsed = time.monotonic() - started
        pop_calls = getattr(processor, "pop_call_count", None)
        calls = int(pop_calls()) if callable(pop_calls) else 0
        out.setdefault("gcw_runtime", {})[",".join(tasks)] = {
            "elapsed_seconds": elapsed,
            "processor_calls": calls,
            "processor_enabled": processor is not None,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        for key in merged:
            if key in out and isinstance(out[key], dict):
                merged[key].update(out[key])
        _write_partial_results(merged)
        _write_progress(
            _update_progress(
                run_idx, "completed", runs, merged, progress=progress
            ),
            merged,
        )
    merged["gcw_protocol"] = {
        "name": "lm_eval_loglikelihood_v1",
        "prompt_mode": prompt_mode,
        "apply_chat_template": apply_chat_template,
        "thinking_dimension": (
            False
            if prompt_mode == "plain"
            else prompt_mode.startswith("thinking_")
        ),
        "generation_process_every": (
            getattr(
                getattr(model.config, "cache_processor", None),
                "generation_process_every",
                None,
            )
            if processor is not None
            else None
        ),
    }
    return merged
