from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import tempfile
import time
from typing import Any, cast

import torch
import torch.nn as nn

from gcw.checkpoint import load_checkpoint
from gcw.runtime import base_dir
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, get_token_bytes
from gcw.evaluation.scoring import (
    encode_generate_prompt,
    evaluate_nll_bpb,
    stop_token_sequences,
    tokenizer_for_checkpoint,
    trim_generated_token_ids,
    truncate_context_continuation,
)
from gcw.models.losses import checkpoint_loss_backend


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

GENERAL_TASKS = [
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
]
MATH_TASKS = [
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa",
    "agieval_gaokao_mathqa",
]
GENERAL_RUNS = [
    (["hellaswag", "piqa"], 0),
    (["winogrande"], 5),
    (["arc_easy", "arc_challenge"], 25),
    (["mmlu"], 5),
]
MATH_RUNS = [([task], 0) for task in MATH_TASKS]
TASK_FEWSHOT = {
    task: fewshot
    for tasks, fewshot in [*GENERAL_RUNS, *MATH_RUNS]
    for task in tasks
}



class GCWEvalLM(TemplateLM):
    def __init__(
        self,
        model,
        tokenizer,
        batch_size: int = 8,
    ):
        if not HAS_LM_EVAL:
            raise ImportError("Install the eval extra: uv sync --extra eval")
        super().__init__()
        self.accelerator = _accelerator()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = model.config.sequence_len
        self.add_special_tokens_for_generate = False
        self.prompt_mode = "auto"

    @property
    def tokenizer_name(self) -> str:
        return getattr(self.tokenizer, "name_or_path", self.tokenizer.__class__.__name__)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_id

    def apply_chat_template(self, chat_history: list[dict[str, str]], add_generation_prompt=True) -> str:
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        prompt_mode = getattr(self, "prompt_mode", "auto")
        if prompt_mode in {"thinking_off", "thinking_on"}:
            kwargs["enable_thinking"] = prompt_mode == "thinking_on"
        return self.tokenizer.apply_chat_template(chat_history, **kwargs)

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
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


def run_all(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
) -> dict:
    return run_task_groups(
        checkpoint,
        [*GENERAL_RUNS, *MATH_RUNS],
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
    )


def run_general(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
) -> dict:
    return run_task_groups(
        checkpoint,
        GENERAL_RUNS,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
    )


def run_math(
    checkpoint: str | Path | None = None,
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
) -> dict:
    return run_task_groups(
        checkpoint,
        MATH_RUNS,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
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
) -> dict:
    unknown = sorted(set(tasks) - set(TASK_FEWSHOT))
    if unknown:
        raise ValueError(f"unknown benchmark tasks: {unknown}")
    runs = [([task], TASK_FEWSHOT[task]) for task in tasks]
    return run_task_groups(
        checkpoint,
        runs,
        limit=limit,
        batch_size=batch_size,
        prompt_mode=prompt_mode,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
    )


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
    loss_backend = checkpoint_loss_backend(state, torch.device(device))
    nll, bpb = evaluate_nll_bpb(
        model, batches, steps, get_token_bytes(device), loss_backend=loss_backend
    )
    return {
        "model_step": state.get("step"),
        "val_nll": nll,
        "val_bpb": bpb,
        "eval_tokens": steps * batch_size * cfg.sequence_len,
        "batch_size": batch_size,
        "sequence_len": cfg.sequence_len,
    }


def run_task_groups(
    checkpoint: str | Path | None,
    runs: list[tuple[list[str], int]],
    limit: int | None = None,
    batch_size: int = 8,
    prompt_mode: str = "auto",
    *,
    hf_model: str | Path | None = None,
    processor_enabled: bool = True,
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
    model, tokenizer = _load_eval_target(
        checkpoint=checkpoint,
        hf_model=hf_model,
        processor_enabled=processor_enabled,
        device=device,
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
            _write_progress(_update_progress(run_idx, "failed", runs, merged, progress=progress, error=str(exc)), merged)
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
        _write_progress(_update_progress(run_idx, "completed", runs, merged, progress=progress), merged)
    merged["gcw_protocol"] = {
        "name": "lm_eval_loglikelihood_v1",
        "prompt_mode": prompt_mode,
        "apply_chat_template": apply_chat_template,
        "thinking_dimension": False if prompt_mode == "plain" else prompt_mode.startswith("thinking_"),
    }
    return merged


def _should_apply_chat_template(tokenizer: Any, prompt_mode: str) -> bool:
    if prompt_mode == "plain":
        return False
    return bool(getattr(tokenizer, "chat_template", None) or getattr(tokenizer, "default_chat_template", None))


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
            raise ValueError("processor bypass requires a composed GCW checkpoint")
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


def _load_eval_target(
    *,
    checkpoint: str | Path | None,
    hf_model: str | Path | None,
    processor_enabled: bool,
    device: torch.device,
):
    if (checkpoint is None) == (hf_model is None):
        raise ValueError("choose exactly one evaluation target: checkpoint or hf_model")
    if hf_model is not None:
        return _load_hf_eval_target(hf_model, device)
    model, cfg, state = load_checkpoint(cast(str | Path, checkpoint), device=device)
    tokenizer = _tokenizer_for(cfg, state)
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
            torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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

    def forward(self, input_ids: torch.Tensor, *_args, **_kwargs) -> torch.Tensor:
        return cast(Any, self.model)(input_ids=input_ids).logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None = None):
        return cast(Any, self.model).generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=eos_token_id,
            use_cache=True,
        )


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
    return all(task in results or task in groups for task in tasks)


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
    return {
        "results": {},
        "groups": {},
        "n-samples": {},
        "n-shot": {},
        "versions": {},
        "higher_is_better": {},
        "gcw_runtime": {},
    }


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
