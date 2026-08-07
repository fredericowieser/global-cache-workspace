from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
import os
import subprocess
import tempfile
import time

import torch

from gcw.checkpoint import load_checkpoint
from gcw.distill.prompt import ChatPrompt
from gcw.common import base_dir
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, TransformersTokenizer, get_token_bytes
from gcw.eval_support import (
    encode_generate_prompt,
    stop_token_sequences,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
    truncate_context_continuation,
)
from gcw.loss_eval import evaluate_nll_bpb
from gcw.opsd_eval import (
    OpsdEvalCondition,
    build_opsd_eval_prompt,
    greedy_opsd_response,
    greedy_opsd_responses,
)
from gcw.prompt_identity import normalized_prompt_text_sha256, prompt_text_sha256


_encode_generate_prompt = encode_generate_prompt
_stop_token_sequences = stop_token_sequences
_tokenizer_for = tokenizer_for_checkpoint
_trim_generated_token_ids = trim_generated_token_ids
_truncate_context_continuation = truncate_context_continuation

try:
    from lm_eval.api.model import TemplateLM
    HAS_LM_EVAL = True
except ImportError:
    TemplateLM = object
    HAS_LM_EVAL = False

PPT_TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "mmlu"]
PPT_FEWSHOT = {"lambada_openai": 0, "hellaswag": 0, "piqa": 0, "winogrande": 5, "arc_easy": 25, "arc_challenge": 25, "mmlu": 5}
ACC_NORM_TASKS = {"hellaswag", "piqa", "arc_easy", "arc_challenge"}
MATH_TASKS = ["gsm8k_openmath", "hendrycks_math", "svamp", "theoremqa", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"]
PAPER7_TASKS = ["gsm8k_openmath", "svamp", "theoremqa", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"]
OPSD_SCREEN_TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa_opsd_generate",
    "agieval_gaokao_mathqa_opsd_generate",
)
OPSD_FULL_TASK_SAMPLE_COUNTS = {
    "gsm8k_openmath": 1_319,
    "svamp": 300,
    "gsm8k_hard": 1_319,
    "logiqa_opsd_generate": 651,
    "agieval_gaokao_mathqa_opsd_generate": 351,
}
OPSD_DATASET_FILE_PATTERNS = {
    "gsm8k_openmath": "gsm8k-test.arrow",
    "svamp": "svamp-test.arrow",
    "gsm8k_hard": "gsm-hard-train.arrow",
    "logiqa_opsd_generate": "logiqa-test.arrow",
    "agieval_gaokao_mathqa_opsd_generate": "agieval-gaokao-mathqa-test.arrow",
}


@dataclass(frozen=True)
class _OpsdPreparedRequest:
    request_index: int
    prompt: str
    requested_max_new: int
    max_new: int
    until: tuple[str, ...]
    stop_sequence_lists: list[list[int]]
    stop_sequences: tuple[tuple[int, ...], ...]
    chat_prompt: ChatPrompt
MATH_RUNS = [([task], 0) for task in MATH_TASKS]
HENDRYCKS_MATH_LEAF_TASKS = {
    "hendrycks_math_algebra",
    "hendrycks_math_counting_and_prob",
    "hendrycks_math_geometry",
    "hendrycks_math_intermediate_algebra",
    "hendrycks_math_num_theory",
    "hendrycks_math_prealgebra",
    "hendrycks_math_precalc",
}
PAPER7_RUNS = [([task], 0) for task in PAPER7_TASKS]
PPT_RUNS = [
    (["lambada_openai", "hellaswag", "piqa"], 0),
    (["winogrande"], 5),
    (["arc_easy", "arc_challenge"], 25),
    (["mmlu"], 5),
]
FORGETTING_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "social_iqa",
    "winogrande",
    "mmlu",
    "mmlu_pro",
    "bbh_cot_fewshot",
    "humaneval",
    "mbpp",
    "ifeval",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
]


class GCWEvalLM(TemplateLM):
    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
        opsd_condition: OpsdEvalCondition | None = None,
    ):
        if not HAS_LM_EVAL:
            raise ImportError("Install the eval extra: uv sync --extra eval")
        super().__init__()
        self.accelerator = _accelerator()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = model.config.sequence_len
        self.add_special_tokens_for_generate = _legacy_generate_add_special_tokens(model)
        self.opsd_condition = opsd_condition
        self.opsd_generation_records: list[dict] = []

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_id

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt=True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        if _legacy_bottleneck_cache_eval(self.model):
            return self._loglikelihood_tokens_legacy_boundary(requests)
        device = self.model.get_device()
        out: list[tuple[float, bool] | None] = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][1]) + len(requests[i][2]))
        for start in range(0, len(order), self.batch_size):
            idxs = order[start : start + self.batch_size]
            inputs, continuations = [], []
            for i in idxs:
                _, ctx, cont = requests[i]
                full = (ctx + cont)[-(self.max_length + 1) :]
                inputs.append(full[:-1])
                continuations.append(cont[-self.max_length :])
            max_len = max(len(x) for x in inputs)
            padded = torch.zeros(len(inputs), max_len, dtype=torch.long, device=device)
            for row, inp in enumerate(inputs):
                padded[row, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=device)
            logprobs = torch.log_softmax(self.model(padded), dim=-1)
            for row, i in enumerate(idxs):
                cont = continuations[row]
                cont_len = len(cont)
                inp_len = len(inputs[row])
                logp = logprobs[row, inp_len - cont_len : inp_len]
                target = torch.tensor(cont, dtype=torch.long, device=device)
                vals = logp.gather(-1, target[:, None]).squeeze(-1)
                out[i] = (float(vals.sum().item()), bool((logp.argmax(-1) == target).all().item()))
        return out

    @torch.no_grad()
    def _loglikelihood_tokens_legacy_boundary(self, requests):
        from gcw.models.cache_state import CacheState

        device = self.model.get_device()
        out: list[tuple[float, bool] | None] = [None] * len(requests)
        for i, req in enumerate(requests):
            _, ctx, cont = req
            ctx = list(ctx)
            cont = list(cont)
            if not cont:
                out[i] = (0.0, True)
                continue

            ctx, cont = _truncate_context_continuation(ctx, cont, self.max_length)
            ctx_ids = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
            cont_ids = torch.tensor(cont, dtype=torch.long, device=device).unsqueeze(0)

            empty_scores = ctx_ids.new_empty(1, 0, dtype=torch.float32)
            ctx_logits, cache = self.model.backbone.forward_chunk(ctx_ids, 0, CacheState([], [], empty_scores))
            cache = self.model.cache_processor(cache, select_all=True)
            cont_logits, _ = self.model.backbone.forward_chunk(cont_ids, len(ctx), cache)
            logits = torch.cat([ctx_logits[:, -1:, :], cont_logits[:, :-1, :]], dim=1)
            logp = torch.log_softmax(logits[0], dim=-1)
            target = cont_ids[0]
            vals = logp.gather(-1, target[:, None]).squeeze(-1)
            out[i] = (float(vals.sum().item()), bool((logits[0].argmax(-1) == target).all().item()))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        for req in requests:
            text = req.args[0]
            out.append(sum(x[0] for x in self._loglikelihood_tokens([(None, [self.eot_token_id], self.tok_encode(text))])))
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        if self.opsd_condition is not None:
            return self._generate_until_opsd(requests)
        if os.environ.get("GCW_EVAL_REFERENCE_GENERATE") == "1":
            return self._generate_until_reference(requests)
        outs: list[str | None] = [None] * len(requests)
        device = self.model.get_device()
        groups: dict[tuple[int, int, tuple[str, ...]], list[tuple[int, list[int]]]] = {}
        for i, req in enumerate(requests):
            prompt, kwargs = req.args
            max_new = int(kwargs.get("max_gen_toks", 256))
            until = _normalize_until(kwargs.get("until"))
            ids = _encode_generate_prompt(
                self.tokenizer,
                prompt,
                self.max_length,
                add_special_tokens=self.add_special_tokens_for_generate,
            )
            groups.setdefault((len(ids), max_new, until), []).append((i, ids))

        use_fast = hasattr(self.model, "generate_fast")
        for (_, max_new, until), rows in groups.items():
            stop_sequences = _stop_token_sequences(self.tokenizer, until)
            for start in range(0, len(rows), self.batch_size):
                chunk = rows[start : start + self.batch_size]
                prompt_len = len(chunk[0][1])
                batch_ids = torch.tensor([row_ids for _, row_ids in chunk], dtype=torch.long, device=device)
                if use_fast:
                    gen = self.model.generate_fast(
                        batch_ids,
                        max_new_tokens=max_new,
                        eos_token_id=self.eot_token_id,
                        stop_sequences=stop_sequences,
                        pad_token_id=self.eot_token_id,
                    )
                else:
                    gen = self.model.generate(batch_ids, max_new_tokens=max_new, eos_token_id=self.eot_token_id)
                for row, (out_idx, _) in enumerate(chunk):
                    tokens = gen[row, prompt_len:].tolist()
                    tokens = _trim_generated_token_ids(tokens, stop_sequences, self.eot_token_id)
                    text = _trim_until(self.tok_decode(tokens), until)
                    outs[out_idx] = text
        return outs

    @torch.no_grad()
    def _generate_until_opsd(self, requests):
        condition = self.opsd_condition
        if condition is None:
            raise RuntimeError("OPSD generation requires an explicit condition")
        prepared = []
        for request_index, req in enumerate(requests):
            prompt, kwargs = req.args
            requested_max_new = int(kwargs.get("max_gen_toks", 256))
            max_new = condition.max_response_tokens
            until = _normalize_until(kwargs.get("until"))
            stop_sequence_lists = _stop_token_sequences(self.tokenizer, until)
            stop_sequences = tuple(tuple(sequence) for sequence in stop_sequence_lists)
            chat_prompt = build_opsd_eval_prompt(
                self.tokenizer,
                prompt,
                enable_thinking=condition.enable_thinking,
                reasoning_open_text=condition.reasoning_open_text,
            )
            prepared.append(
                _OpsdPreparedRequest(
                    request_index=request_index,
                    prompt=prompt,
                    requested_max_new=requested_max_new,
                    max_new=max_new,
                    until=until,
                    stop_sequence_lists=stop_sequence_lists,
                    stop_sequences=stop_sequences,
                    chat_prompt=chat_prompt,
                )
            )

        grouped: dict[tuple[object, ...], list[_OpsdPreparedRequest]] = {}
        for item in prepared:
            key = (
                len(item.chat_prompt.base_ids),
                len(item.chat_prompt.trigger_ids),
                item.max_new,
                item.stop_sequences,
            )
            grouped.setdefault(key, []).append(item)

        outs = [""] * len(prepared)
        records: list[dict[str, object] | None] = [None] * len(prepared)
        batch_size = max(1, int(getattr(self, "batch_size", 1)))
        for group in grouped.values():
            for offset in range(0, len(group), batch_size):
                chunk = group[offset : offset + batch_size]
                first = chunk[0]
                if len(chunk) == 1:
                    rollouts = (
                        greedy_opsd_response(
                            self.model,
                            self.tokenizer,
                            first.prompt,
                            max_response_tokens=first.max_new,
                            condition=condition,
                            stop_sequences=first.stop_sequences,
                        ),
                    )
                else:
                    rollouts = greedy_opsd_responses(
                        self.model,
                        self.tokenizer,
                        prompts=tuple(item.prompt for item in chunk),
                        max_response_tokens=first.max_new,
                        condition=condition,
                        stop_sequences=first.stop_sequences,
                    )
                for item, rollout in zip(chunk, rollouts, strict=True):
                    request_index = item.request_index
                    stop_sequence_lists = item.stop_sequence_lists
                    until = item.until
                    chat_prompt = item.chat_prompt
                    tokens = _trim_generated_token_ids(
                        list(rollout.response_ids),
                        stop_sequence_lists,
                        self.eot_token_id,
                    )
                    outs[request_index] = _trim_until(self.tok_decode(tokens), until)
                    prompt = item.prompt
                    records[request_index] = {
                        "request_index": request_index,
                        "prompt_sha256": prompt_text_sha256(prompt),
                        "prompt_text_sha256": prompt_text_sha256(prompt),
                        "normalized_prompt_text_sha256": normalized_prompt_text_sha256(
                            prompt
                        ),
                        "base_token_ids_sha256": _token_ids_sha256(chat_prompt.base_ids),
                        "full_prompt_token_ids_sha256": _token_ids_sha256(
                            chat_prompt.full_ids
                        ),
                        "base_tokens": len(chat_prompt.base_ids),
                        "trigger_tokens": len(chat_prompt.trigger_ids),
                        "response_tokens": len(tokens),
                        "max_response_tokens": item.max_new,
                        "task_requested_max_response_tokens": item.requested_max_new,
                        "raw_response_tokens": len(rollout.response_ids),
                        "cache_event_positions": (
                            [0, *range(64, len(rollout.response_ids), 64)]
                            if condition.processor_enabled
                            else []
                        ),
                        "cache_event_lengths": list(rollout.event_cache_lengths),
                    }
        if any(record is None for record in records):
            raise RuntimeError("OPSD batched generation did not produce every request")
        self.opsd_generation_records.extend(records)
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
                _encode_generate_prompt(
                    self.tokenizer,
                    prompt,
                    self.max_length,
                    add_special_tokens=self.add_special_tokens_for_generate,
                ),
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            gen = self.model.generate(ids, max_new_tokens=max_new, eos_token_id=self.eot_token_id)[0].tolist()
            text = self.tok_decode(gen[len(ids[0]) :])
            outs.append(_trim_until(text, until))
        return outs


def run_core(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, [*PPT_RUNS, *MATH_RUNS], limit=limit, batch_size=batch_size)


def run_ppt(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, PPT_RUNS, limit=limit, batch_size=batch_size)


def run_math(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, MATH_RUNS, limit=limit, batch_size=batch_size)


def run_paper7(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, PAPER7_RUNS, limit=limit, batch_size=batch_size)


def run_fineweb(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    eval_tokens: int = 10 * 524288,
    batch_size: int = 16,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, state = load_checkpoint(checkpoint, device=device)
    if cfg.family != "smollm" or cfg.vocab_size != COSMO2_VOCAB_SIZE:
        raise ValueError("FineWeb NLL/BPB eval currently expects a cosmo2 SmolLM checkpoint")
    steps = max(1, eval_tokens // (batch_size * cfg.sequence_len))
    batches = data_loader(batch_size, cfg.sequence_len, split="val", device=device, data_dir=data_dir, infinite=True)
    nll, bpb = evaluate_nll_bpb(model, batches, steps, get_token_bytes(device))
    return {
        "model_step": state.get("step"),
        "val_nll": nll,
        "val_bpb": bpb,
        "eval_tokens": steps * batch_size * cfg.sequence_len,
        "batch_size": batch_size,
        "sequence_len": cfg.sequence_len,
    }


def run_task_groups(
    checkpoint: str | Path,
    runs: list[tuple[list[str], int]],
    limit: int | None = None,
    batch_size: int = 8,
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
    os.environ.setdefault("HF_DATASETS_CACHE", str(base_dir() / "hf_datasets_eval_v3"))
    import lm_eval
    from lm_eval.tasks import TaskManager, get_task_dict
    dist_state, local_rank, world_size = _distributed_eval_state()
    device = _eval_device(local_rank)
    model, cfg, state = load_checkpoint(checkpoint, device=device)
    tokenizer = _tokenizer_for(cfg, state)
    lm = GCWEvalLM(model, tokenizer, batch_size=batch_size)
    apply_chat_template = bool(getattr(tokenizer, "chat_template", None) or getattr(tokenizer, "default_chat_template", None))
    task_manager = TaskManager(include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR")
    merged = _load_partial_results(runs) or _empty_eval_results()
    _write_progress(_new_progress(runs), merged)
    for run_idx, (tasks, fewshot) in enumerate(runs):
        progress = _update_progress(run_idx, "running", runs, merged)
        if _run_is_complete(tasks, merged):
            _write_progress(_update_progress(run_idx, "completed", runs, merged, progress=progress, skipped=True), merged)
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
            _write_progress(_update_progress(run_idx, "failed", runs, merged, progress=progress, error=str(exc)), merged)
            raise
        if out is None:
            continue
        for key in merged:
            if key in out and isinstance(out[key], dict):
                merged[key].update(out[key])
        _write_partial_results(merged)
        _write_progress(_update_progress(run_idx, "completed", runs, merged, progress=progress), merged)
    return merged


def run_opsd_screen(
    checkpoint: str | Path,
    *,
    source_revision: str,
    tokenizer_source: str,
    datasets_cache: str,
    enable_thinking: bool,
    processor_enabled: bool,
    tasks: tuple[str, ...] = OPSD_SCREEN_TASKS,
    limit: int | None = 64,
    batch_size: int = 1,
) -> dict:
    if not HAS_LM_EVAL:
        raise ImportError("Install the eval extra: uv sync --extra eval")
    if limit is not None and limit <= 0:
        raise ValueError("OPSD screen limit must be positive when set")
    if batch_size <= 0:
        raise ValueError("OPSD screen batch_size must be positive")
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("OPSD screen source revision must be a full lowercase Git SHA")
    _verify_source_revision(source_revision)
    datasets_cache_path = Path(datasets_cache).resolve()
    if not datasets_cache_path.is_dir():
        raise ValueError("OPSD screen datasets cache must be an existing directory")
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache_path)
    allowed_tasks = set(OPSD_SCREEN_TASKS)
    if not tasks or not set(tasks) <= allowed_tasks:
        raise ValueError(
            "OPSD screen tasks must be GSM8K, SVAMP, GSM-Hard, LogiQA, or Gaokao"
        )

    import lm_eval
    from lm_eval.tasks import TaskManager

    device = _eval_device(0)
    model, cfg, state = load_checkpoint(checkpoint, device=device)
    tokenizer = TransformersTokenizer(tokenizer_source.removeprefix("hf:"))
    condition = OpsdEvalCondition(
        enable_thinking=enable_thinking,
        processor_enabled=processor_enabled,
    )
    lm = GCWEvalLM(model, tokenizer, batch_size=batch_size, opsd_condition=condition)
    task_manager = TaskManager(include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR")
    evaluated = lm_eval.simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=0,
        limit=limit,
        bootstrap_iters=0,
        log_samples=True,
        verbosity="ERROR",
        task_manager=task_manager,
        confirm_run_unsafe_code=True,
        apply_chat_template=False,
    )
    if evaluated is None:
        raise RuntimeError("lm-eval returned no OPSD screen result")
    checkpoint_path = Path(checkpoint).resolve()
    sample_manifest = _opsd_sample_manifest(evaluated, tasks)
    if len(sample_manifest) != len(lm.opsd_generation_records):
        raise ValueError("OPSD screen lm-eval samples differ from generation records")
    _validate_opsd_full_sample_counts(sample_manifest, tasks=tasks, limit=limit)
    return {
        "schema_version": 1,
        "kind": "opsd_quick_screen_condition",
        "source_revision": source_revision,
        "tokenizer_source": tokenizer_source,
        "tokenizer_identity_sha256": tokenizer.identity_sha256(),
        "datasets_cache": str(datasets_cache_path),
        "condition": {
            "name": condition.name,
            "enable_thinking": enable_thinking,
            "processor_enabled": processor_enabled,
            "process_every": condition.process_every,
            "select_all": condition.select_all,
            "reasoning_open_text": condition.reasoning_open_text,
            "max_response_tokens": condition.max_response_tokens,
        },
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path / "model.safetensors"),
        "checkpoint_manifest": _checkpoint_manifest(checkpoint_path),
        "model_config": asdict(cfg),
        "train_state": _json_safe(state),
        "tasks": list(tasks),
        "task_config_sha256": {
            task: _sha256_file(Path("configs/tasks") / f"{task}.yaml") for task in tasks
        },
        "dataset_files": _opsd_dataset_file_identity(tasks),
        "limit_per_task": limit,
        "execution_batch_size": batch_size,
        "generation_policy": {
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
            "fewshot": 0,
            "apply_chat_template_in_lm_eval": False,
        },
        "sample_manifest": sample_manifest,
        "generation_records": lm.opsd_generation_records,
        "lm_eval": _json_safe(evaluated),
    }


def _token_ids_sha256(token_ids: tuple[int, ...]) -> str:
    return hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _checkpoint_manifest(checkpoint: Path) -> dict[str, object]:
    files = {}
    for name in ("model.safetensors", "config.json", "train_state.json"):
        path = checkpoint / name
        if not path.is_file():
            raise ValueError(f"OPSD screen checkpoint is missing {name}")
        files[name] = {"size_bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    manifest_sha256 = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"files": files, "manifest_sha256": manifest_sha256}


def _validate_opsd_full_sample_counts(
    sample_manifest: Sequence[Mapping[str, object]],
    *,
    tasks: tuple[str, ...],
    limit: int | None,
) -> None:
    if limit is not None:
        return
    observed = Counter(str(row["task"]) for row in sample_manifest)
    expected = {task: OPSD_FULL_TASK_SAMPLE_COUNTS[task] for task in tasks}
    if dict(observed) != expected:
        raise ValueError(
            "OPSD full evaluation sample counts do not match authenticated splits: "
            f"observed={dict(observed)!r}, expected={expected!r}"
        )


def _opsd_sample_manifest(evaluated: dict, tasks: tuple[str, ...]) -> list[dict[str, object]]:
    samples = evaluated.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("OPSD screen requires lm-eval sample logs")
    manifest: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for task in tasks:
        rows = samples.get(task)
        if not isinstance(rows, list):
            raise ValueError(f"OPSD screen sample logs are missing task {task}")
        for row in rows:
            required = ("doc_id", "doc_hash", "prompt_hash", "target_hash")
            if not isinstance(row, dict) or any(key not in row for key in required):
                raise ValueError("OPSD screen sample identity fields are incomplete")
            identity = (task, *(row[key] for key in required))
            if identity in seen:
                continue
            seen.add(identity)
            manifest.append({"task": task, **{key: row[key] for key in required}})
    return manifest


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _opsd_dataset_file_identity(tasks: tuple[str, ...]) -> dict[str, dict[str, str | int]]:
    cache_root = Path(os.environ.get("HF_DATASETS_CACHE", base_dir() / "hf_datasets_eval_v3"))
    patterns = OPSD_DATASET_FILE_PATTERNS
    identity: dict[str, dict[str, str | int]] = {}
    for task in tasks:
        matches = sorted(cache_root.rglob(patterns[task]))
        if len(matches) != 1:
            raise ValueError(f"OPSD screen expected one cached dataset file for {task}, got {len(matches)}")
        path = matches[0].resolve()
        identity[task] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return identity


def _verify_source_revision(expected: str) -> None:
    repository = Path(__file__).resolve().parents[1]
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    if actual != expected:
        raise ValueError("OPSD screen source revision differs from the live checkout")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repository, text=True
    )
    if dirty:
        raise ValueError("OPSD screen requires a clean source checkout")


def _filter_skipped_tasks(runs: list[tuple[list[str], int]]) -> list[tuple[list[str], int]]:
    skip = {task.strip() for task in os.environ.get("GCW_EVAL_SKIP_TASKS", "").split(",") if task.strip()}
    if not skip:
        return runs
    filtered = []
    for tasks, fewshot in runs:
        kept = [task for task in tasks if task not in skip]
        if kept:
            filtered.append((kept, fewshot))
    return filtered


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


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
    return Accelerator() if PartialState().num_processes > 1 else _NoopAccelerator()


def _legacy_generate_add_special_tokens(model) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "family", None) in {"llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4"}
    )


def _legacy_bottleneck_cache_eval(model) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "variant", None) in {"bt", "bgt", "longshot"}
        and hasattr(model, "backbone")
        and hasattr(model, "cache_processor")
    )


class _NoopAccelerator:
    @staticmethod
    def gather(x):
        return x



def _normalize_until(until) -> tuple[str, ...]:
    if until is None:
        return ()
    if isinstance(until, str):
        return (until,)
    return tuple(str(stop) for stop in until if stop is not None)


def _trim_until(text: str, until: tuple[str, ...]) -> str:
    for stop in until:
        if stop and stop in text:
            text = text.split(stop)[0]
    return text


def _run_is_complete(tasks: list[str], merged: dict) -> bool:
    results = merged.get("results", {})
    groups = merged.get("groups", {})
    for task in tasks:
        if task == "hendrycks_math":
            if task not in groups or not HENDRYCKS_MATH_LEAF_TASKS.issubset(results):
                return False
        elif task not in results and task not in groups:
            return False
    return True


def _json_safe(payload: dict) -> dict:
    safe = json.loads(json.dumps(payload, sort_keys=True, default=str))
    if not isinstance(safe, dict):
        raise ValueError("authenticated JSON payload must be an object")
    return safe


def _payload_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write_authenticated_json(path: Path, *, kind: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(payload)
    document = {
        "schema_version": 1,
        "kind": kind,
        "payload": safe,
        "payload_sha256": _payload_sha256(safe),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _load_authenticated_json(path: Path, *, kind: str) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"authenticated {kind} JSON is unreadable") from exc
    if not isinstance(document, dict) or set(document) != {
        "schema_version", "kind", "payload", "payload_sha256"
    }:
        raise ValueError(f"authenticated {kind} JSON schema differs")
    payload = document.get("payload")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != kind
        or not isinstance(payload, dict)
        or document.get("payload_sha256") != _payload_sha256(payload)
    ):
        raise ValueError(f"authenticated {kind} JSON integrity check failed")
    return payload


def _write_partial_results(results: dict) -> None:
    if not is_main_process():
        return
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return
    _atomic_write_authenticated_json(
        Path(partial_dir) / "partial.json",
        kind="flow_math_eval_partial_results",
        payload=results,
    )


def _empty_eval_results() -> dict:
    return {"results": {}, "groups": {}, "n-samples": {}, "n-shot": {}, "versions": {}, "higher_is_better": {}}


def _validate_partial_results(payload: dict, runs: list[tuple[list[str], int]]) -> dict:
    expected_keys = set(_empty_eval_results())
    if set(payload) != expected_keys or any(not isinstance(payload[key], dict) for key in expected_keys):
        raise ValueError("authenticated partial results schema differs")
    requested = {task for tasks, _fewshot in runs for task in tasks}
    for section in ("results", "groups"):
        for task in payload[section]:
            if not any(task == requested_task or task.startswith(f"{requested_task}_") for requested_task in requested):
                raise ValueError(f"authenticated partial results contain unexpected task: {task}")
    return payload


def _load_partial_results(runs: list[tuple[list[str], int]]) -> dict | None:
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return None
    path = Path(partial_dir) / "partial.json"
    if not path.exists():
        return None
    payload = _load_authenticated_json(path, kind="flow_math_eval_partial_results")
    return _validate_partial_results(payload, runs)


def _progress_path() -> Path | None:
    if os.environ.get("GCW_EVAL_PROGRESS_PATH"):
        return Path(os.environ["GCW_EVAL_PROGRESS_PATH"])
    if os.environ.get("GCW_EVAL_PARTIAL_DIR"):
        return Path(os.environ["GCW_EVAL_PARTIAL_DIR"]) / "progress.json"
    return None


def _validate_progress(payload: dict, runs: list[tuple[list[str], int]]) -> dict:
    if set(payload) != {
        "started_at", "started_time", "updated_at", "total_runs", "completed_runs",
        "completed_tasks", "elapsed_sec", "eta_sec", "runs",
    }:
        raise ValueError("authenticated progress schema differs")
    entries = payload.get("runs")
    if not isinstance(entries, list) or len(entries) != len(runs):
        raise ValueError("authenticated progress run list differs")
    allowed_entry_keys = {
        "tasks", "fewshot", "status", "start_time", "end_time", "started_at",
        "ended_at", "duration_sec", "skipped", "error",
    }
    expected_runs = [{"tasks": tasks, "fewshot": fewshot} for tasks, fewshot in runs]
    actual_runs = []
    for entry in entries:
        if not isinstance(entry, dict) or not set(entry).issubset(allowed_entry_keys):
            raise ValueError("authenticated progress entry schema differs")
        if not {"tasks", "fewshot", "status", "start_time", "end_time"}.issubset(entry):
            raise ValueError("authenticated progress entry is incomplete")
        actual_runs.append({"tasks": entry["tasks"], "fewshot": entry["fewshot"]})
    if payload.get("total_runs") != len(runs) or actual_runs != expected_runs:
        raise ValueError("authenticated progress does not match requested evaluation runs")
    return payload


def _new_progress(runs: list[tuple[list[str], int]]) -> dict:
    path = _progress_path()
    if path is not None and path.exists():
        payload = _load_authenticated_json(path, kind="flow_math_eval_progress")
        return _validate_progress(payload, runs)
    now = time.time()
    return {
        "started_at": _format_time(now),
        "started_time": now,
        "updated_at": _format_time(now),
        "total_runs": len(runs),
        "completed_runs": 0,
        "completed_tasks": [],
        "elapsed_sec": 0.0,
        "eta_sec": None,
        "runs": [
            {"tasks": tasks, "fewshot": fewshot, "status": "pending", "start_time": None, "end_time": None}
            for tasks, fewshot in runs
        ],
    }


def _update_progress(
    run_idx: int,
    status: str,
    runs: list[tuple[list[str], int]],
    merged: dict,
    *,
    progress: dict | None = None,
    skipped: bool = False,
    error: str | None = None,
) -> dict:
    progress = progress or _new_progress(runs)
    now = time.time()
    entry = progress["runs"][run_idx]
    if entry.get("start_time") is None:
        entry["start_time"] = now
        entry["started_at"] = _format_time(now)
    entry["status"] = status
    entry["skipped"] = bool(skipped)
    if error:
        entry["error"] = error
    if status in {"completed", "failed"}:
        entry["end_time"] = now
        entry["ended_at"] = _format_time(now)
        entry["duration_sec"] = max(0.0, now - float(entry["start_time"]))
    progress["updated_at"] = _format_time(now)
    progress["completed_runs"] = sum(1 for item in progress["runs"] if item.get("status") == "completed")
    progress["completed_tasks"] = sorted(merged.get("results", {}))
    progress["elapsed_sec"] = max(0.0, now - float(progress.get("started_time", now)))
    progress["eta_sec"] = _estimate_eta(progress)
    return progress


def _estimate_eta(progress: dict) -> float | None:
    runs = progress.get("runs", [])
    completed = [float(item["duration_sec"]) for item in runs if item.get("status") == "completed" and not item.get("skipped") and item.get("duration_sec")]
    remaining = sum(1 for item in runs if item.get("status") not in {"completed", "failed"})
    if not completed or remaining <= 0:
        return None
    return sum(completed) / len(completed) * remaining


def _write_progress(progress: dict, merged: dict) -> None:
    if not is_main_process():
        return
    path = _progress_path()
    if path is None:
        return
    progress["completed_tasks"] = sorted(merged.get("results", {}))
    _atomic_write_authenticated_json(path, kind="flow_math_eval_progress", payload=progress)


def _format_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
