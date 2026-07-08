from __future__ import annotations

import json
from pathlib import Path
import os
import time

import torch

from gcw.checkpoint import load_checkpoint
from gcw.common import base_dir
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, IntTokenizer, get_token_bytes, get_tokenizer
from gcw.loss_eval import evaluate_nll_bpb

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
MATH_RUNS = [(MATH_TASKS, 0)]
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
    def __init__(self, model, tokenizer, batch_size: int = 8):
        if not HAS_LM_EVAL:
            raise ImportError("Install the eval extra: uv sync --extra eval")
        super().__init__()
        self.accelerator = _accelerator()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = model.config.sequence_len
        self.add_special_tokens_for_generate = _legacy_generate_add_special_tokens(model)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        if _legacy_bottleneck_cache_eval(self.model):
            return self._loglikelihood_tokens_legacy_boundary(requests)
        device = self.model.get_device()
        out = [None] * len(requests)
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
        out = [None] * len(requests)
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
        if os.environ.get("GCW_EVAL_REFERENCE_GENERATE") == "1":
            return self._generate_until_reference(requests)
        outs = [None] * len(requests)
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
                ids = torch.tensor([row_ids for _, row_ids in chunk], dtype=torch.long, device=device)
                if use_fast:
                    gen = self.model.generate_fast(
                        ids,
                        max_new_tokens=max_new,
                        eos_token_id=self.eot_token_id,
                        stop_sequences=stop_sequences,
                        pad_token_id=self.eot_token_id,
                    )
                else:
                    gen = self.model.generate(ids, max_new_tokens=max_new, eos_token_id=self.eot_token_id)
                for row, (out_idx, _) in enumerate(chunk):
                    tokens = gen[row, prompt_len:].tolist()
                    tokens = _trim_generated_token_ids(tokens, stop_sequences, self.eot_token_id)
                    text = _trim_until(self.tok_decode(tokens), until)
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
    merged = _load_partial_results() or _empty_eval_results()
    _write_progress(_new_progress(runs), merged)
    for run_idx, (tasks, fewshot) in enumerate(runs):
        progress = _update_progress(run_idx, "running", runs, merged)
        if all(task in merged["results"] for task in tasks):
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
        and getattr(config, "family", None) in {"llama", "qwen3"}
    )


def _legacy_bottleneck_cache_eval(model) -> bool:
    config = getattr(model, "config", None)
    return bool(
        getattr(config, "legacy_bottleneck_generate", False)
        and getattr(config, "variant", None) in {"bt", "bgt"}
        and hasattr(model, "backbone")
        and hasattr(model, "cache_processor")
    )


def _truncate_context_continuation(ctx: list[int], cont: list[int], max_length: int) -> tuple[list[int], list[int]]:
    full = ctx + cont
    if len(full) <= max_length + 1:
        return ctx, cont
    overflow = len(full) - (max_length + 1)
    if overflow < len(ctx):
        return ctx[overflow:], cont
    trimmed = full[overflow:]
    return trimmed[:1], trimmed[1:]


def _encode_generate_prompt(tokenizer, prompt: str, max_length: int, *, add_special_tokens: bool = False) -> list[int]:
    try:
        ids = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    except TypeError:
        ids = tokenizer.encode(prompt)
    return ids[-max_length:]


class _NoopAccelerator:
    @staticmethod
    def gather(x):
        return x


def _tokenizer_for(config, state):
    source = state.get("source") or state.get("tokenizer_source")
    if source and config.family in {"llama", "qwen3"}:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Converted HF checkpoints require transformers for eval tokenization.") from exc
        tok = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
        return _HFTokenizerAdapter(tok)
    if config.family in {"llama", "qwen3"}:
        raise ValueError(
            "HF-family checkpoints require train_state.json to include 'source' "
            "or 'tokenizer_source' so eval uses the original tokenizer."
        )
    if config.family == "smollm" and config.vocab_size == COSMO2_VOCAB_SIZE:
        return get_tokenizer("cosmo2")
    return IntTokenizer(vocab_size=config.vocab_size, eot_id=0)


class _HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.default_chat_template = getattr(tokenizer, "default_chat_template", None)

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=bool(add_special_tokens))

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


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


def _stop_token_sequences(tokenizer, until: tuple[str, ...]) -> list[list[int]]:
    sequences: list[list[int]] = []
    for stop in until:
        if not stop:
            continue
        try:
            ids = tokenizer.encode(stop, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(stop)
        if ids:
            sequences.append(list(ids))
    return sequences


def _trim_generated_token_ids(tokens: list[int], stop_sequences: list[list[int]], eos_token_id: int | None) -> list[int]:
    end = len(tokens)
    if eos_token_id is not None:
        for i, token in enumerate(tokens):
            if token == eos_token_id:
                end = min(end, i)
                break
    for stop in stop_sequences:
        start = _find_subsequence(tokens, stop)
        if start is not None:
            end = min(end, start)
    return tokens[:end]


def _find_subsequence(tokens: list[int], pattern: list[int]) -> int | None:
    if not pattern or len(pattern) > len(tokens):
        return None
    width = len(pattern)
    for start in range(0, len(tokens) - width + 1):
        if tokens[start : start + width] == pattern:
            return start
    return None


def _write_partial_results(results: dict) -> None:
    if not is_main_process():
        return
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return
    path = Path(partial_dir) / "partial.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str))


def _empty_eval_results() -> dict:
    return {"results": {}, "groups": {}, "n-samples": {}, "n-shot": {}, "versions": {}, "higher_is_better": {}}


def _load_partial_results() -> dict | None:
    partial_dir = os.environ.get("GCW_EVAL_PARTIAL_DIR")
    if not partial_dir:
        return None
    path = Path(partial_dir) / "partial.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        return {**_empty_eval_results(), **payload}
    return None


def _progress_path() -> Path | None:
    if os.environ.get("GCW_EVAL_PROGRESS_PATH"):
        return Path(os.environ["GCW_EVAL_PROGRESS_PATH"])
    if os.environ.get("GCW_EVAL_PARTIAL_DIR"):
        return Path(os.environ["GCW_EVAL_PARTIAL_DIR"]) / "progress.json"
    return None


def _new_progress(runs: list[tuple[list[str], int]]) -> dict:
    path = _progress_path()
    if path is not None and path.exists():
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
            return payload
    now = time.time()
    return {
        "started_at": _format_time(now),
        "started_time": now,
        "updated_at": _format_time(now),
        "total_runs": len(runs),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["completed_tasks"] = sorted(merged.get("results", {}))
    path.write_text(json.dumps(progress, indent=2, default=str))


def _format_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
