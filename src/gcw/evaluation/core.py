from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import torch

from gcw.config import EvalConfig
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
    _trim_until,  # noqa: F401  (re-exported for tests)
)


def run_all(
    checkpoint: str | Path | None = None,
    config: EvalConfig | None = None,
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
    loaded_target: tuple[Any, Any] | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        [*GENERAL_RUNS, *MATH_RUNS],
        diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        config=config,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
        loaded_target=loaded_target,
    )


def run_general(
    checkpoint: str | Path | None = None,
    config: EvalConfig | None = None,
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
    loaded_target: tuple[Any, Any] | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        GENERAL_RUNS,
        diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        config=config,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
        loaded_target=loaded_target,
    )


def run_math(
    checkpoint: str | Path | None = None,
    config: EvalConfig | None = None,
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
    loaded_target: tuple[Any, Any] | None = None,
) -> dict:
    return run_task_groups(
        checkpoint,
        MATH_RUNS,
        diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        config=config,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
        loaded_target=loaded_target,
    )


def run_tasks(
    tasks: list[str],
    checkpoint: str | Path | None = None,
    config: EvalConfig | None = None,
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
    loaded_target: tuple[Any, Any] | None = None,
) -> dict:
    runs = [([task], TASK_FEWSHOT.get(task, 0)) for task in tasks]
    return run_task_groups(
        checkpoint,
        runs,
        diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        config=config,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        dtype=dtype,
        generation_process_every=generation_process_every,
        loaded_target=loaded_target,
    )


def run_task_groups(
    checkpoint: str | Path | None,
    runs: list[tuple[list[str], int]],
    config: EvalConfig | None = None,
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
    loaded_target: tuple[Any, Any] | None = None,
) -> dict:
    config = config or EvalConfig()
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
    if loaded_target is None:
        model, tokenizer = _load_eval_target(
            checkpoint=checkpoint,
            hf_model=hf_model,
            processor_enabled=processor_enabled,
            device=device,
            dtype=dtype,
            generation_process_every=generation_process_every,
            diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        )
    else:
        model, tokenizer = loaded_target
    lm = GCWEvalLM(model, tokenizer, config)
    processor = getattr(model, "cache_processor", None)
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
                limit=config.limit,
                bootstrap_iters=0,
                log_samples=False,
                verbosity="ERROR",
                task_manager=task_manager,
                confirm_run_unsafe_code=True,
                apply_chat_template=config.apply_chat_template,
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
        runtime = {
            "elapsed_seconds": elapsed,
            "processor_calls": calls,
            "processor_enabled": processor is not None,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        from gcw.models.cache_processors.diffusion import (
            pop_diffusion_field_evaluation_count,
        )

        field_evaluations = pop_diffusion_field_evaluation_count(model)
        if field_evaluations:
            runtime["diffusion_field_evaluations"] = field_evaluations
        out.setdefault("gcw_runtime", {})[",".join(tasks)] = runtime
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
    protocol: dict[str, object] = {
        "name": "lm_eval_loglikelihood_v1",
        "apply_chat_template": config.apply_chat_template,
        "enable_thinking": config.enable_thinking,
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
    if diffusion_eval_solver_steps is not None:
        protocol["diffusion_eval_solver_steps"] = diffusion_eval_solver_steps
    merged["gcw_protocol"] = protocol
    return merged
