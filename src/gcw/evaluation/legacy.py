from __future__ import annotations

import os
from pathlib import Path

import torch

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
from gcw.evaluation.scoring import truncate_context_continuation
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


class LegacyPaper7EvalLM(GCWEvalLM):
    def __init__(self, model, tokenizer, batch_size: int = 8):
        super().__init__(model, tokenizer, batch_size=batch_size)
        self.add_special_tokens_for_generate = (
            _legacy_generate_add_special_tokens(model)
        )

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt=True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        if _legacy_bottleneck_cache_eval(self.model):
            return self._loglikelihood_tokens_legacy_boundary(requests)
        return super()._loglikelihood_tokens(
            requests, disable_tqdm=disable_tqdm
        )

    @torch.no_grad()
    def _loglikelihood_tokens_legacy_boundary(self, requests):
        from gcw.models.cache_processors.state import CacheState

        device = self.model.get_device()
        out: list[tuple[float, bool] | None] = [None] * len(requests)
        for i, req in enumerate(requests):
            _, ctx, cont = req
            ctx = list(ctx)
            cont = list(cont)
            if not cont:
                out[i] = (0.0, True)
                continue

            ctx, cont = truncate_context_continuation(
                ctx, cont, self.max_length
            )
            ctx_ids = torch.tensor(
                ctx, dtype=torch.long, device=device
            ).unsqueeze(0)
            cont_ids = torch.tensor(
                cont, dtype=torch.long, device=device
            ).unsqueeze(0)

            empty_scores = ctx_ids.new_empty(1, 0, dtype=torch.float32)
            ctx_logits, cache = self.model.backbone.forward_chunk(
                ctx_ids, 0, CacheState([], [], empty_scores)
            )
            cache = self.model.cache_processor(cache, select_all=True)
            cont_logits, _ = self.model.backbone.forward_chunk(
                cont_ids, len(ctx), cache
            )
            logits = torch.cat(
                [ctx_logits[:, -1:, :], cont_logits[:, :-1, :]], dim=1
            )
            logp = torch.log_softmax(logits[0], dim=-1)
            target = cont_ids[0]
            vals = logp.gather(-1, target[:, None]).squeeze(-1)
            out[i] = (
                float(vals.sum().item()),
                bool((logits[0].argmax(-1) == target).all().item()),
            )
        return out


def run_paper7_legacy(
    checkpoint: str | Path,
    limit: int | None = None,
    batch_size: int = 8,
    dtype: torch.dtype | str | None = "auto",
) -> dict:
    return run_paper7_legacy_groups(
        checkpoint,
        PAPER7_RUNS,
        limit=limit,
        batch_size=batch_size,
        dtype=dtype,
    )


def run_paper7_legacy_groups(
    checkpoint: str | Path,
    runs: list[tuple[list[str], int]],
    limit: int | None = None,
    batch_size: int = 8,
    dtype: torch.dtype | str | None = "auto",
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
        hf_model=None,
        processor_enabled=True,
        device=device,
        dtype=dtype,
    )
    lm = LegacyPaper7EvalLM(model, tokenizer, batch_size=batch_size)
    apply_chat_template = bool(
        getattr(tokenizer, "chat_template", None)
        or getattr(tokenizer, "default_chat_template", None)
    )
    task_manager = TaskManager(
        include_path=str(Path("configs/tasks_legacy/paper7").resolve()),
        verbosity="ERROR",
    )
    merged = (
        _legacy_empty_results(_load_partial_results(runs))
        or _legacy_empty_results()
    )
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
    return merged


def _legacy_empty_results(results: dict | None = None) -> dict:
    payload = results or {}
    return {
        "results": dict(payload.get("results", {})),
        "groups": dict(payload.get("groups", {})),
        "n-samples": dict(payload.get("n-samples", {})),
        "n-shot": dict(payload.get("n-shot", {})),
        "versions": dict(payload.get("versions", {})),
        "higher_is_better": dict(payload.get("higher_is_better", {})),
    }


def _legacy_generate_add_special_tokens(model) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "family", None)
        in {"llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"}
    )


def _legacy_bottleneck_cache_eval(model) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "variant", None) in {"bt", "bgt"}
        and hasattr(model, "backbone")
        and hasattr(model, "cache_processor")
    )
