from __future__ import annotations

from pathlib import Path

import torch

from gcw.config import EvalConfig
from gcw.evaluation.lm_eval_adapter import GCWEvalLM
from gcw.evaluation.paper7 import (
    PAPER7_RUNS,
    PAPER7_TASKS as _PAPER7_TASKS,
    _run_paper7_groups,
)
from gcw.evaluation.scoring import truncate_context_continuation

PAPER7_TASKS = _PAPER7_TASKS


class LegacyPaper7EvalLM(GCWEvalLM):
    def __init__(
        self,
        model,
        tokenizer,
        eval_config: EvalConfig | None = None,
    ):
        super().__init__(model, tokenizer, eval_config)
        self.add_special_tokens_for_generate = (
            _legacy_generate_add_special_tokens(model)
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
        legacy=True,
        lm_class=LegacyPaper7EvalLM,
    )


def run_paper7_legacy_groups(
    checkpoint: str | Path,
    runs: list[tuple[list[str], int]],
    config: EvalConfig | None = None,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    head_selection_method: str | None = None,
) -> dict:
    return _run_paper7_groups(
        checkpoint,
        runs,
        config=config,
        dtype=dtype,
        generation_process_every=generation_process_every,
        head_selection_method=head_selection_method,
        legacy=True,
        lm_class=LegacyPaper7EvalLM,
    )


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
