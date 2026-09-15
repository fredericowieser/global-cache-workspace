from __future__ import annotations

import os
from pathlib import Path
import time

import torch

from gcw.config import EvalConfig
from gcw.evaluation.lm_eval_adapter import (
    GCWEvalLM,
    HAS_LM_EVAL,
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
)
from gcw.evaluation.tasks import _filter_skipped_tasks, _run_is_complete
from gcw.runtime import base_dir

PAPER7_TASKS = [
    "gsm8k_openmath",
    "svamp",
    "theoremqa",
    "logiqa",
    "agieval_gaokao_mathqa",
    "gsm8k_hard",
]
PAPER7_RUNS = [([task], 0) for task in PAPER7_TASKS]
PAPER7_MULTIPLE_CHOICE_TASKS = {
    "logiqa",
    "agieval_gaokao_mathqa",
}


def run_paper7(
    checkpoint: str | Path,
    config: EvalConfig | None = None,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    head_selection_method: str | None = None,
) -> dict:
    return _run_paper7_groups(
        checkpoint,
        PAPER7_RUNS,
        config=config,
        dtype=dtype,
        generation_process_every=generation_process_every,
        head_selection_method=head_selection_method,
        legacy=False,
        lm_class=GCWEvalLM,
    )


def _run_paper7_groups(
    checkpoint: str | Path,
    runs: list[tuple[list[str], int]],
    config: EvalConfig | None,
    dtype: torch.dtype | str | None,
    generation_process_every: int | None,
    head_selection_method: str | None,
    *,
    legacy: bool,
    lm_class,
) -> dict:
    evaluation_started = time.monotonic()
    config = config or EvalConfig(apply_chat_template=True)
    if not HAS_LM_EVAL:
        raise ImportError("Install the eval extra: uv sync --extra eval")
    runs = _filter_skipped_tasks(runs)
    if not legacy:
        runs = [([task], fewshot) for tasks, fewshot in runs for task in tasks]
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
        hf_model=None,
        processor_enabled=True,
        device=device,
        dtype=dtype,
        generation_process_every=generation_process_every,
        head_selection_method=head_selection_method,
    )
    has_chat_template = bool(
        getattr(tokenizer, "chat_template", None)
        or getattr(tokenizer, "default_chat_template", None)
    )
    if config.apply_chat_template and not has_chat_template:
        if config.enable_thinking:
            raise ValueError(
                "enable_thinking requires a tokenizer with a chat template"
            )
        config.apply_chat_template = False
    lm = lm_class(model, tokenizer, config)
    processor = getattr(model, "cache_processor", None)
    task_manager = TaskManager(
        include_path=str(Path("configs/tasks_legacy/paper7").resolve()),
        verbosity="ERROR",
    )
    merged = (
        _paper7_empty_results(_load_partial_results(runs))
        or _paper7_empty_results()
    )
    task_protocols = {}
    _write_progress(_new_progress(runs), merged)
    for run_idx, (tasks, fewshot) in enumerate(runs):
        apply_chat_template = config.apply_chat_template
        if not legacy and tasks[0] in PAPER7_MULTIPLE_CHOICE_TASKS:
            apply_chat_template = False
        if not legacy:
            task_protocols[tasks[0]] = {
                "apply_chat_template": apply_chat_template,
                "enable_thinking": (
                    config.enable_thinking if apply_chat_template else False
                ),
                "request_type": (
                    "multiple_choice_likelihood"
                    if tasks[0] in PAPER7_MULTIPLE_CHOICE_TASKS
                    else "answer_generation"
                ),
                "max_new_tokens": (
                    None if tasks[0] in PAPER7_MULTIPLE_CHOICE_TASKS else 32768
                ),
            }
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
        lm.reset_runtime_metrics()
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
        lm_runtime = lm.pop_runtime_metrics()
        if out is None:
            continue
        runtime = {
            "elapsed_seconds": time.monotonic() - started,
            "processor_calls": (
                int(processor.pop_call_count())
                if callable(getattr(processor, "pop_call_count", None))
                else 0
            ),
            "processor_enabled": processor is not None,
            "peak_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if torch.cuda.is_available()
                else 0.0
            ),
            **lm_runtime,
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
        _write_partial_results({**_empty_eval_results(), **merged})
        _write_progress(
            _update_progress(
                run_idx, "completed", runs, merged, progress=progress
            ),
            merged,
        )
    merged.setdefault("gcw_runtime", {})["evaluation"] = {
        "scope": "end_to_end_including_loading_and_setup",
        "elapsed_seconds": time.monotonic() - evaluation_started,
    }
    _write_partial_results({**_empty_eval_results(), **merged})
    if not legacy:
        merged["gcw_protocol"] = {
            "name": "paper7_mixed_prompt_v2",
            "version": 2,
            "tasks": task_protocols,
        }
    return merged


def _paper7_empty_results(results: dict | None = None) -> dict:
    payload = results or {}
    return {
        "results": dict(payload.get("results", {})),
        "groups": dict(payload.get("groups", {})),
        "n-samples": dict(payload.get("n-samples", {})),
        "n-shot": dict(payload.get("n-shot", {})),
        "versions": dict(payload.get("versions", {})),
        "higher_is_better": dict(payload.get("higher_is_better", {})),
        "gcw_runtime": dict(payload.get("gcw_runtime", {})),
    }
