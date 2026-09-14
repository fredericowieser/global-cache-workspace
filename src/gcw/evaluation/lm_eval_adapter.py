from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, TypeVar, cast
import torch
import torch.nn as nn

from gcw.checkpoint import load_checkpoint
from gcw.config import EvalConfig
from gcw.evaluation.scoring import (
    encode_generate_prompt,
    stop_token_sequences,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
)
from gcw.evaluation.tasks import _normalize_until, _trim_until
from gcw.runtime import print0

BatchRow = TypeVar("BatchRow")
BatchResult = TypeVar("BatchResult")

try:
    from lm_eval.api.model import TemplateLM

    HAS_LM_EVAL = True
except ImportError:
    TemplateLM = object
    HAS_LM_EVAL = False


def _distributed_eval_state():
    try:
        from accelerate import PartialState
    except ImportError:
        return None, 0, 1
    state = PartialState()
    return state, state.local_process_index, state.num_processes


def _eval_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _accelerator():
    try:
        from accelerate import Accelerator, PartialState
    except ImportError:
        return _NoopAccelerator()
    return (
        Accelerator()
        if PartialState().num_processes > 1
        else _NoopAccelerator()
    )


class _NoopAccelerator:
    @staticmethod
    def gather(x):
        return x


class ProcessorBypass(nn.Module):
    """Expose a composed checkpoint's exact loaded backbone for control cells."""

    def __init__(self, model: nn.Module):
        super().__init__()
        backbone = getattr(model, "backbone", None)
        if not isinstance(backbone, nn.Module):
            raise ValueError(
                "processor bypass requires a composed GCW checkpoint"
            )
        self.backbone = backbone
        self.config = getattr(backbone, "config")

    def get_device(self) -> torch.device:
        return cast(Any, self.backbone).get_device()

    def forward(self, *args, **kwargs):
        return cast(Any, self.backbone)(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return cast(Any, self.backbone).generate(*args, **kwargs)

    def generate_fast(self, *args, **kwargs):
        return cast(Any, self.backbone).generate_fast(*args, **kwargs)


def _apply_generation_process_every(config, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("generation_process_every must be a positive integer")
    processor_config = getattr(config, "cache_processor", None)
    if processor_config is None:
        raise ValueError(
            "generation_process_every requires a cache-processor checkpoint"
        )
    processor_config.generation_process_every = value


def _apply_head_selection_method(config, value: str | None) -> None:
    if value is None:
        return
    if value != "random":
        raise ValueError(
            "evaluation head-selection override supports only 'random'"
        )
    processor_config = getattr(config, "cache_processor", None)
    if (
        getattr(config, "variant", None) != "longshot"
        or processor_config is None
    ):
        raise ValueError(
            "head_selection_method override requires a Longshot checkpoint"
        )
    processor_config.head_selection_method = value
    processor_config.internal_iteration_score = value


def _load_eval_target(
    *,
    checkpoint: str | Path | None,
    hf_model: str | Path | None,
    processor_enabled: bool,
    device: torch.device,
    dtype: torch.dtype | str | None = "auto",
    generation_process_every: int | None = None,
    head_selection_method: str | None = None,
    diffusion_eval_solver_steps: int | None = None,
):
    if (checkpoint is None) == (hf_model is None):
        raise ValueError(
            "choose exactly one evaluation target: checkpoint or hf_model"
        )
    if diffusion_eval_solver_steps is not None:
        if (
            type(diffusion_eval_solver_steps) is not int
            or diffusion_eval_solver_steps <= 0
        ):
            raise ValueError(
                "diffusion evaluation solver steps must be positive"
            )
        if hf_model is not None:
            raise ValueError(
                "diffusion evaluation solver steps do not apply to HF targets"
            )
        if not processor_enabled:
            raise ValueError(
                "diffusion evaluation solver steps require processor=on"
            )
    if hf_model is not None:
        if generation_process_every is not None:
            raise ValueError(
                "generation_process_every applies only to GCW checkpoints"
            )
        if head_selection_method is not None:
            raise ValueError(
                "head_selection_method applies only to GCW checkpoints"
            )
        return _load_hf_eval_target(hf_model, device)
    if generation_process_every is not None and not processor_enabled:
        raise ValueError(
            "generation_process_every requires processor_enabled=True"
        )
    if head_selection_method is not None and not processor_enabled:
        raise ValueError(
            "head_selection_method requires processor_enabled=True"
        )
    model, cfg, state = load_checkpoint(
        cast(str | Path, checkpoint), device=device, dtype=dtype
    )
    if diffusion_eval_solver_steps is not None:
        from gcw.models.cache_processors.diffusion import (
            set_diffusion_eval_solver_steps,
        )

        set_diffusion_eval_solver_steps(model, diffusion_eval_solver_steps)
    _apply_generation_process_every(cfg, generation_process_every)
    _apply_head_selection_method(cfg, head_selection_method)
    tokenizer = tokenizer_for_checkpoint(cfg, state)
    if not processor_enabled and hasattr(model, "backbone"):
        model = ProcessorBypass(model)
    return model, tokenizer


def _load_hf_eval_target(source: str | Path, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gcw.evaluation.scoring import HFTokenizerAdapter

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    hf_model = cast(
        Any,
        AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=False,
            torch_dtype=(
                torch.bfloat16 if device.type == "cuda" else torch.float32
            ),
        ),
    ).to(device)
    adapter = HFCausalLMAdapter(hf_model)
    return adapter, HFTokenizerAdapter(tokenizer, str(source))


class HFCausalLMAdapter(nn.Module):
    """Minimal native interface used by GCW's lm-eval adapter."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        max_length = int(getattr(model.config, "max_position_embeddings", 2048))
        self.config = type("EvalConfig", (), {"sequence_len": max_length})()

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(
        self, input_ids: torch.Tensor, *_args, **_kwargs
    ) -> torch.Tensor:
        return cast(Any, self.model)(input_ids=input_ids).logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ):
        return cast(Any, self.model).generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            use_cache=True,
        )


class GCWEvalLM(TemplateLM):
    def __init__(
        self,
        model,
        tokenizer,
        eval_config: EvalConfig | None = None,
    ):
        if not HAS_LM_EVAL:
            raise ImportError("Install the eval extra: uv sync --extra eval")
        eval_config = eval_config or EvalConfig()
        super().__init__()
        self.accelerator = _accelerator()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = eval_config.batch_size
        self._auto_batch_sizes = {
            "loglikelihood": eval_config.max_batch_size,
            "generation": eval_config.max_batch_size,
        }
        self.max_length = model.config.sequence_len
        self.add_special_tokens_for_generate = False
        self.eval_config = eval_config

    @property
    def tokenizer_name(self) -> str:
        return getattr(
            self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__
        )

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_id

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt=True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.eval_config.enable_thinking,
        )

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    def _run_batches(
        self,
        rows: list[BatchRow],
        workload: str,
        run_batch: Callable[[list[BatchRow]], list[BatchResult]],
    ) -> list[BatchResult]:
        if self.batch_size != "auto":
            return [
                result
                for start in range(0, len(rows), self.batch_size)
                for result in run_batch(rows[start : start + self.batch_size])
            ]

        results: list[BatchResult] = []
        start = 0
        while start < len(rows):
            batch_size = min(
                self._auto_batch_sizes[workload], len(rows) - start
            )
            try:
                batch_results = run_batch(rows[start : start + batch_size])
            except RuntimeError as exc:
                is_cuda_oom = isinstance(
                    exc, torch.cuda.OutOfMemoryError
                ) or "CUDA out of memory" in str(exc)
                if not is_cuda_oom or batch_size == 1:
                    raise
                reduced = max(1, batch_size // 2)
                self._auto_batch_sizes[workload] = reduced
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print0(
                    f"lm-eval {workload} batch of {batch_size} ran out of "
                    f"memory; retrying with {reduced}"
                )
                continue
            if len(batch_results) != batch_size:
                raise RuntimeError(
                    f"{workload} batch returned {len(batch_results)} results "
                    f"for {batch_size} rows"
                )
            results.extend(batch_results)
            start += batch_size
        return results

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        device = self.model.get_device()
        out: list[tuple[float, bool] | None] = [None] * len(requests)
        order = sorted(
            range(len(requests)),
            key=lambda i: len(requests[i][1]) + len(requests[i][2]),
        )
        def run_batch(idxs: list[int]) -> list[tuple[float, bool]]:
            inputs, continuations = [], []
            for i in idxs:
                _, ctx, cont = requests[i]
                full = (ctx + cont)[-(self.max_length + 1) :]
                inputs.append(full[:-1])
                continuations.append(cont[-self.max_length :])
            max_len = max(len(x) for x in inputs)
            padded = torch.zeros(
                len(inputs), max_len, dtype=torch.long, device=device
            )
            for row, inp in enumerate(inputs):
                padded[row, : len(inp)] = torch.tensor(
                    inp, dtype=torch.long, device=device
                )
            logprobs = torch.log_softmax(self.model(padded), dim=-1)
            batch_results = []
            for row, i in enumerate(idxs):
                cont = continuations[row]
                cont_len = len(cont)
                inp_len = len(inputs[row])
                logp = logprobs[row, inp_len - cont_len : inp_len]
                target = torch.tensor(cont, dtype=torch.long, device=device)
                vals = logp.gather(-1, target[:, None]).squeeze(-1)
                batch_results.append((
                    float(vals.sum().item()),
                    bool((logp.argmax(-1) == target).all().item()),
                ))
            return batch_results

        for i, result in zip(
            order,
            self._run_batches(order, "loglikelihood", run_batch),
            strict=True,
        ):
            out[i] = result
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        for req in requests:
            text = req.args[0]
            out.append(
                sum(
                    x[0]
                    for x in self._loglikelihood_tokens(
                        [(None, [self.eot_token_id], self.tok_encode(text))]
                    )
                )
            )
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        if os.environ.get("GCW_EVAL_REFERENCE_GENERATE") == "1":
            return self._generate_until_reference(requests)
        outs: list[str | None] = [None] * len(requests)
        device = self.model.get_device()
        groups: dict[
            tuple[int, int, tuple[str, ...]], list[tuple[int, list[int]]]
        ] = {}
        for i, req in enumerate(requests):
            prompt, kwargs = req.args
            max_new = int(kwargs.get("max_gen_toks", 256))
            until = _normalize_until(kwargs.get("until"))
            ids = encode_generate_prompt(
                self.tokenizer,
                prompt,
                self.max_length,
                add_special_tokens=self.add_special_tokens_for_generate,
            )
            groups.setdefault((len(ids), max_new, until), []).append((i, ids))

        use_fast = hasattr(self.model, "generate_fast")
        for (_, max_new, until), rows in groups.items():
            stop_seqs = stop_token_sequences(self.tokenizer, until)

            def run_batch(
                chunk: list[tuple[int, list[int]]],
            ) -> list[str]:
                prompt_len = len(chunk[0][1])
                batch_ids = torch.tensor(
                    [row_ids for _, row_ids in chunk],
                    dtype=torch.long,
                    device=device,
                )
                if use_fast:
                    gen = self.model.generate_fast(
                        batch_ids,
                        max_new_tokens=max_new,
                        eos_token_id=self.eot_token_id,
                        stop_sequences=stop_seqs,
                        pad_token_id=self.eot_token_id,
                    )
                else:
                    gen = self.model.generate(
                        batch_ids,
                        max_new_tokens=max_new,
                        eos_token_id=self.eot_token_id,
                    )
                batch_results = []
                for row, _ in enumerate(chunk):
                    tokens = gen[row, prompt_len:].tolist()
                    tokens = trim_generated_token_ids(
                        tokens, stop_seqs, self.eot_token_id
                    )
                    text = _trim_until(self.tok_decode(tokens), until)
                    batch_results.append(text)
                return batch_results

            for (out_idx, _), text in zip(
                rows,
                self._run_batches(rows, "generation", run_batch),
                strict=True,
            ):
                outs[out_idx] = text
        return outs

    @torch.no_grad()
    def _generate_until_reference(self, requests):
        outs = []
        device = self.model.get_device()
        for req in requests:
            prompt, kwargs = req.args
            max_new = int(kwargs.get("max_gen_toks", 256))
            until = _normalize_until(kwargs.get("until"))
            ids = torch.tensor(
                encode_generate_prompt(
                    self.tokenizer,
                    prompt,
                    self.max_length,
                    add_special_tokens=self.add_special_tokens_for_generate,
                ),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            gen = self.model.generate(
                ids, max_new_tokens=max_new, eos_token_id=self.eot_token_id
            )[0].tolist()
            text = self.tok_decode(gen[len(ids[0]) :])
            outs.append(_trim_until(text, until))
        return outs
