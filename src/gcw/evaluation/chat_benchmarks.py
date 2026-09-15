from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence, cast

import torch

from gcw.config import EvalConfig
from gcw.evaluation.benchmarks import (
    BenchmarkBackend,
    GCWBenchmarkBackend,
    HFBenchmarkBackend,
    MAX_NEW_TOKENS,
    _load_tokenizer,
    _trim_at_eos,
    apply_qwen_chat_template,
)
from gcw.evaluation.scoring import (
    classify_generated_tokens,
    extract_numeric_answer,
)

GENERAL_CHAT_TASKS = (
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "mmlu",
)
MATH_CHAT_TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa",
    "agieval_gaokao_mathqa",
)
CHAT_TASKS = (*GENERAL_CHAT_TASKS, *MATH_CHAT_TASKS)
NUMERIC_TASKS = frozenset({"gsm8k_openmath", "svamp", "gsm8k_hard"})
GENERAL_TASK_LIMITS = {
    "hellaswag": 1_000,
    "piqa": 1_000,
    "winogrande": 1_000,
    "arc_easy": 1_000,
    "arc_challenge": 1_000,
    "mmlu": 20,  # Per subject: 57 subjects, at most 1,140 rows.
}
RESULT_SCHEMA = "gcw.chat_benchmark.v4"
_SELECTION_SEED = 24
_CHOICE_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_FINAL_CHOICES_RE = re.compile(
    r"(?i)final\s+answer\s*(?:is|[:=])\s*(?:option[s]?\s*)?"
    r"[\(\[]?([A-Z](?:\s*[,/&]\s*[A-Z])*)"
)
_BOXED_CHOICES_RE = re.compile(
    r"\\boxed\{\s*([A-Z](?:\s*[,/&]\s*[A-Z])*)\s*\}", re.IGNORECASE
)


@dataclass(frozen=True)
class ChatExample:
    index: int
    task: str
    source_task: str
    source_index: int
    prompt: str
    target_kind: str
    target: str
    choices: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedChatExample:
    example: ChatExample
    prompt_ids: tuple[int, ...]


def format_multiple_choice_prompt(
    stem: str,
    choices: Sequence[str],
    *,
    allow_multiple: bool = False,
) -> str:
    if not 2 <= len(choices) <= len(_CHOICE_LABELS):
        raise ValueError(
            "multiple-choice examples require between 2 and 26 choices"
        )
    cleaned_stem = re.sub(
        r"(?:\n|^)\s*Answer:\s*$", "", stem.strip(), flags=re.IGNORECASE
    )
    rendered = "\n".join(
        f"{_CHOICE_LABELS[index]}. {choice.strip()}"
        for index, choice in enumerate(choices)
    )
    instruction = (
        "Reason through the problem, then end with exactly `Final answer: X`, "
        "where X is the correct choice letter."
    )
    if allow_multiple:
        instruction += (
            " If more than one choice is correct, return the letters separated by commas, "
            "for example `Final answer: A,D`."
        )
    return f"{cleaned_stem}\n\nChoices:\n{rendered}\n\n{instruction}"


def format_numeric_prompt(question: str) -> str:
    """Preserve the calibrated Qwen/GSM8K user prompt exactly."""
    return question.strip()


def extract_final_choices(
    response: str, choice_count: int
) -> tuple[int, ...] | None:
    matches = list(_FINAL_CHOICES_RE.finditer(response))
    if not matches:
        matches = list(_BOXED_CHOICES_RE.finditer(response))
    if not matches:
        return None
    labels = re.findall(r"[A-Z]", matches[-1].group(1).upper())
    indices = tuple(sorted({ord(label) - ord("A") for label in labels}))
    return (
        indices
        if indices and all(0 <= index < choice_count for index in indices)
        else None
    )


def extract_final_choice(response: str, choice_count: int) -> int | None:
    indices = extract_final_choices(response, choice_count)
    return indices[0] if indices is not None and len(indices) == 1 else None


def _stable_doc_hash(task: str, index: int, doc: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"seed": _SELECTION_SEED, "task": task, "index": index, "doc": doc},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_docs(
    task: str, docs: Iterable[Mapping[str, Any]], limit: int | None
) -> list[tuple[int, Mapping[str, Any]]]:
    indexed = list(enumerate(docs))
    if limit is None or len(indexed) <= limit:
        return indexed
    return sorted(
        indexed, key=lambda item: _stable_doc_hash(task, item[0], item[1])
    )[:limit]


def _flatten_tasks(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        flattened: list[Any] = []
        for child in value.values():
            flattened.extend(_flatten_tasks(child))
        return flattened
    return [value]


def _task_docs(task: Any) -> Iterable[Mapping[str, Any]]:
    if task.has_test_docs():
        return cast(Iterable[Mapping[str, Any]], task.test_docs())
    if task.has_validation_docs():
        return cast(Iterable[Mapping[str, Any]], task.validation_docs())
    raise ValueError(
        f"task {task.config.task!r} has no test or validation documents"
    )


def _coerce_choice_indices(
    task_name: str,
    task: Any,
    doc: Mapping[str, Any],
    choices: Sequence[str],
) -> tuple[int, ...]:
    if task_name == "winogrande":
        return (int(doc["answer"]) - 1,)
    gold = doc.get("gold")
    if (
        isinstance(gold, (list, tuple))
        and gold
        and all(isinstance(index, int) for index in gold)
    ):
        indices = tuple(sorted(set(int(index) for index in gold)))
        if all(0 <= index < len(choices) for index in indices):
            return indices
    candidates = [
        task.doc_to_target(doc),
        doc.get("gold"),
        doc.get("label"),
        doc.get("answerKey"),
        doc.get("answer"),
        doc.get("target"),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) == 1:
            candidate = candidate[0]
        if isinstance(candidate, bool) or candidate is None:
            continue
        if isinstance(candidate, int) and 0 <= candidate < len(choices):
            return (candidate,)
        text = str(candidate).strip()
        if len(text) == 1 and text.upper() in _CHOICE_LABELS[: len(choices)]:
            return (_CHOICE_LABELS.index(text.upper()),)
        if text.isdigit():
            number = int(text)
            if 0 <= number < len(choices):
                return (number,)
            if 1 <= number <= len(choices):
                return (number - 1,)
        for index, choice in enumerate(choices):
            if text == str(choice).strip():
                return (index,)
    raise ValueError(f"cannot resolve gold choice for {task_name}: {doc!r}")


def _task_stem(task_name: str, task: Any, doc: Mapping[str, Any]) -> str:
    if task_name == "hellaswag":
        return str(doc.get("query") or doc.get("ctx") or task.doc_to_text(doc))
    if task_name == "piqa":
        return str(doc.get("goal") or task.doc_to_text(doc))
    if task_name == "winogrande":
        return str(doc.get("sentence") or task.doc_to_text(doc))
    if task_name == "agieval_gaokao_mathqa" and doc.get("query"):
        return re.split(r"\n\s*选项：", str(doc["query"]), maxsplit=1)[0]
    return str(task.doc_to_text(doc))


def _load_task_examples(
    task_names: Sequence[str],
) -> tuple[list[ChatExample], dict[str, Any]]:
    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(
        include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR"
    )
    examples: list[ChatExample] = []
    identity: dict[str, Any] = {}
    next_index = 0
    for requested_name in task_names:
        loaded = get_task_dict([requested_name], manager)
        leaf_tasks = _flatten_tasks(loaded)
        task_rows = 0
        source_counts: dict[str, int] = {}
        source_configs: dict[str, Any] = {}
        selected_hashes: list[str] = []
        for task in leaf_tasks:
            source_name = str(task.config.task)
            source_configs[source_name] = {
                "dataset_path": str(task.config.dataset_path),
                "dataset_name": task.config.dataset_name,
                "dataset_kwargs": task.config.dataset_kwargs,
                "test_split": task.config.test_split,
                "validation_split": task.config.validation_split,
            }
            per_source_limit = GENERAL_TASK_LIMITS.get(requested_name)
            selected = _select_docs(
                source_name, _task_docs(task), per_source_limit
            )
            source_counts[source_name] = len(selected)
            for source_index, doc in selected:
                selected_hashes.append(
                    _stable_doc_hash(source_name, source_index, doc)
                )
                if requested_name in NUMERIC_TASKS:
                    target = extract_numeric_answer(
                        str(task.doc_to_target(doc))
                    )
                    if target is None:
                        raise ValueError(
                            f"numeric target is not parseable for {source_name} row {source_index}"
                        )
                    example = ChatExample(
                        index=next_index,
                        task=requested_name,
                        source_task=source_name,
                        source_index=source_index,
                        prompt=format_numeric_prompt(
                            str(task.doc_to_text(doc))
                        ),
                        target_kind="numeric",
                        target=str(target),
                    )
                else:
                    choices = tuple(
                        str(choice) for choice in task.doc_to_choice(doc)
                    )
                    gold = _coerce_choice_indices(
                        requested_name, task, doc, choices
                    )
                    example = ChatExample(
                        index=next_index,
                        task=requested_name,
                        source_task=source_name,
                        source_index=source_index,
                        prompt=format_multiple_choice_prompt(
                            _task_stem(requested_name, task, doc),
                            choices,
                            allow_multiple=requested_name
                            == "agieval_gaokao_mathqa",
                        ),
                        target_kind="choice",
                        target=",".join(
                            _CHOICE_LABELS[index] for index in gold
                        ),
                        choices=choices,
                    )
                examples.append(example)
                next_index += 1
                task_rows += 1
        identity[requested_name] = {
            "selected_rows": task_rows,
            "source_counts": source_counts,
            "source_configs": source_configs,
            "selection": (
                "all_rows"
                if requested_name not in GENERAL_TASK_LIMITS
                else "sha256_ranked_seed_24"
            ),
            "per_source_limit": GENERAL_TASK_LIMITS.get(requested_name),
            "ordered_selection_sha256": hashlib.sha256(
                "\n".join(selected_hashes).encode()
            ).hexdigest(),
        }
    return examples, identity


def _prepare_examples(
    examples: Sequence[ChatExample],
    tokenizer: Any,
    enable_thinking: bool,
) -> list[PreparedChatExample]:
    return [
        PreparedChatExample(
            example=example,
            prompt_ids=apply_qwen_chat_template(
                tokenizer,
                example.prompt,
                enable_thinking=enable_thinking,
            ),
        )
        for example in examples
    ]


def evaluate_chat_examples(
    backend: BenchmarkBackend,
    tokenizer: Any,
    examples: Sequence[ChatExample],
    *,
    config: EvalConfig,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> dict[str, Any]:
    if not config.apply_chat_template:
        raise ValueError("chat benchmark evaluation requires chat formatting")
    batch_size = config.fixed_batch_size
    prepared = _prepare_examples(examples, tokenizer, config.enable_thinking)
    if backend.supports_left_padding:
        ordered = sorted(prepared, key=lambda item: len(item.prompt_ids))
        batches = [
            ordered[offset : offset + batch_size]
            for offset in range(0, len(ordered), batch_size)
        ]
        grouping = "length_sorted_left_padding"
    else:
        by_length: dict[int, list[PreparedChatExample]] = defaultdict(list)
        for example in prepared:
            by_length[len(example.prompt_ids)].append(example)
        batches = [
            group[offset : offset + batch_size]
            for _length, group in sorted(by_length.items())
            for offset in range(0, len(group), batch_size)
        ]
        grouping = "exact_prompt_token_length"

    generated: list[dict[str, Any] | None] = [None] * len(examples)
    started = time.monotonic()
    peak_memory = 0
    if backend.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(backend.device)
    for chunk in batches:
        width = max(len(item.prompt_ids) for item in chunk)
        input_ids = torch.full(
            (len(chunk), width),
            backend.pad_id,
            dtype=torch.long,
            device=backend.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, item in enumerate(chunk):
            prompt = torch.tensor(
                item.prompt_ids, dtype=torch.long, device=backend.device
            )
            input_ids[row, -len(prompt) :] = prompt
            attention_mask[row, -len(prompt) :] = 1
        output = backend.generate(input_ids, attention_mask, max_new_tokens)
        for item, tokens in zip(
            chunk, output[:, width:].detach().cpu().tolist(), strict=True
        ):
            response_ids, termination = _trim_at_eos(tokens, backend.eos_ids)
            token_stats = classify_generated_tokens(
                tokenizer,
                response_ids,
                enable_thinking=config.enable_thinking,
                termination=termination,
            )
            response = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            source_example = item.example
            if source_example.target_kind == "choice":
                prediction_indices = extract_final_choices(
                    response, len(source_example.choices)
                )
                prediction = (
                    None
                    if prediction_indices is None
                    else ",".join(
                        _CHOICE_LABELS[index] for index in prediction_indices
                    )
                )
                correct = prediction == source_example.target
            else:
                numeric = extract_numeric_answer(response)
                prediction = None if numeric is None else str(numeric)
                correct = prediction == source_example.target
            generated[source_example.index] = {
                **asdict(source_example),
                "prompt_tokens": len(item.prompt_ids),
                "response_tokens": len(response_ids),
                "answer_tokens": token_stats.answer_tokens,
                "thinking_tokens": token_stats.thinking_tokens,
                "thinking_terminated": token_stats.thinking_terminated,
                "thinking_truncated": token_stats.thinking_truncated,
                "termination": termination,
                "response": response,
                "prediction": prediction,
                "parse_success": prediction is not None,
                "correct": correct,
            }
        if backend.device.type == "cuda":
            peak_memory = max(
                peak_memory,
                int(torch.cuda.max_memory_allocated(backend.device)),
            )
    if any(row is None for row in generated):
        raise RuntimeError("chat generation did not produce every row")
    rows = [cast(dict[str, Any], row) for row in generated]
    task_results: dict[str, Any] = {}
    for task in dict.fromkeys(example.task for example in examples):
        selected = [row for row in rows if row["task"] == task]
        correct_count = sum(bool(row["correct"]) for row in selected)
        parsed = sum(bool(row["parse_success"]) for row in selected)
        truncated = sum(
            row["termination"] == "max_new_tokens" for row in selected
        )
        answer_tokens = sum(int(row["answer_tokens"]) for row in selected)
        thinking_tokens = sum(int(row["thinking_tokens"]) for row in selected)
        thinking_truncated = sum(
            bool(row["thinking_truncated"]) for row in selected
        )
        task_results[task] = {
            "correct": correct_count,
            "total": len(selected),
            "accuracy": correct_count / len(selected),
            "parsed": parsed,
            "parse_rate": parsed / len(selected),
            "truncated": truncated,
            "truncation_rate": truncated / len(selected),
            "response_tokens": sum(
                int(row["response_tokens"]) for row in selected
            ),
            "answer_tokens": answer_tokens,
            "answer_tokens_mean": answer_tokens / len(selected),
            "thinking_tokens": thinking_tokens,
            "thinking_tokens_mean": thinking_tokens / len(selected),
            "unterminated_thinking": sum(
                not bool(row["thinking_terminated"]) for row in selected
            ),
            "thinking_truncated": thinking_truncated,
            "thinking_truncation_rate": thinking_truncated / len(selected),
        }
    return {
        "apply_chat_template": config.apply_chat_template,
        "enable_thinking": config.enable_thinking,
        "protocol": "official_chat_template_answer_generation_v1",
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "grouping": grouping,
        "elapsed_seconds": time.monotonic() - started,
        "peak_memory_bytes": peak_memory,
        "tasks": task_results,
        "rows": rows,
    }


def run_chat_benchmarks(
    *,
    tasks: Sequence[str],
    config: EvalConfig | None = None,
    checkpoint: str | Path | None = None,
    hf_model: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    processor_enabled: bool = True,
    max_new_tokens: int = MAX_NEW_TOKENS,
    generation_process_every: int | None = None,
    diffusion_eval_solver_steps: int | None = None,
) -> dict[str, Any]:
    end_to_end_started = time.monotonic()
    config = config or EvalConfig(apply_chat_template=True)
    if not config.apply_chat_template:
        raise ValueError("chat benchmark evaluation requires chat formatting")
    unknown = sorted(set(tasks) - set(CHAT_TASKS))
    if unknown:
        raise ValueError(f"unknown chat benchmark tasks: {unknown}")
    if (checkpoint is None) == (hf_model is None):
        raise ValueError("choose exactly one evaluation target")
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
    if generation_process_every is not None and hf_model is not None:
        raise ValueError(
            "generation_process_every applies only to GCW checkpoints"
        )
    if generation_process_every is not None and not processor_enabled:
        raise ValueError(
            "generation_process_every requires processor_enabled=True"
        )
    if max_new_tokens != MAX_NEW_TOKENS:
        raise ValueError(
            f"chat benchmark protocol requires max_new_tokens={MAX_NEW_TOKENS}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("chat benchmark evaluation requires CUDA")
    device = torch.device("cuda")
    if hf_model is not None:
        backend: BenchmarkBackend = HFBenchmarkBackend(
            Path(hf_model).expanduser().resolve(),
            device,
            processor_enabled=processor_enabled,
        )
        tokenizer_source = (
            Path(tokenizer_path or hf_model).expanduser().resolve()
        )
    else:
        backend = GCWBenchmarkBackend(
            Path(cast(str | Path, checkpoint)).expanduser().resolve(),
            device,
            processor_enabled=processor_enabled,
            generation_process_every=generation_process_every,
            diffusion_eval_solver_steps=diffusion_eval_solver_steps,
        )
        if tokenizer_path is None:
            raise ValueError(
                "native chat benchmark evaluation requires tokenizer_path"
            )
        tokenizer_source = Path(tokenizer_path).expanduser().resolve()
    tokenizer, tokenizer_identity = _load_tokenizer(tokenizer_source)
    examples, dataset_identity = _load_task_examples(tasks)
    result = evaluate_chat_examples(
        backend,
        tokenizer,
        examples,
        config=config,
        max_new_tokens=max_new_tokens,
    )
    backend_model = getattr(backend, "model", None)
    if isinstance(backend_model, torch.nn.Module):
        from gcw.models.cache_processors.diffusion import (
            pop_diffusion_field_evaluation_count,
        )

        field_evaluations = pop_diffusion_field_evaluation_count(backend_model)
        if field_evaluations:
            result["diffusion_field_evaluations"] = field_evaluations
    result["end_to_end_elapsed_seconds"] = time.monotonic() - end_to_end_started
    payload = {
        "schema": RESULT_SCHEMA,
        "target": backend.identity,
        "tokenizer": tokenizer_identity,
        "datasets": dataset_identity,
        "processor_requested": backend.identity["processor_requested"],
        "processor_present": backend.identity["processor_present"],
        "processor_enabled": backend.identity["processor_enabled"],
        "generation_process_every": backend.identity.get(
            "generation_process_every"
        ),
        **result,
    }
    if diffusion_eval_solver_steps is not None:
        payload["diffusion_eval_solver_steps"] = diffusion_eval_solver_steps
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}
