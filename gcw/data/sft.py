from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


def iter_jsonl(path: str | Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prepare_openmath(source: str | Path, out: str | Path, limit: int | None = None) -> int:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in _iter_openmath_rows(source):
            prompt, answer = _prompt_answer(row)
            if not prompt or not answer:
                continue
            f.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
            count += 1
            if limit is not None and count >= limit:
                break
    return count


def _iter_openmath_rows(source: str | Path) -> Iterable[dict]:
    path = Path(source)
    if path.is_file():
        yield from iter_jsonl(path)
        return
    jsonl_files = sorted(path.glob("*.jsonl"))
    if jsonl_files:
        for file in jsonl_files:
            yield from iter_jsonl(file)
        return
    try:
        from datasets import DatasetDict, load_from_disk
    except ImportError as exc:
        raise ImportError("Install datasets to prepare OpenMathInstruct Arrow data.") from exc
    data = load_from_disk(str(path))
    if isinstance(data, DatasetDict):
        data = data["train"] if "train" in data else next(iter(data.values()))
    for row in data:
        yield dict(row)


def _prompt_answer(row: dict) -> tuple[str, str]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        prompt_parts = [str(msg.get("content", "")) for msg in messages[:-1] if isinstance(msg, dict)]
        answer = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        return "\n".join(part for part in prompt_parts if part).strip(), str(answer).strip()
    prompt = row.get("prompt") or row.get("question") or row.get("problem") or row.get("instruction") or row.get("input") or ""
    answer = (
        row.get("answer")
        or row.get("generated_solution")
        or row.get("solution")
        or row.get("response")
        or row.get("output")
        or row.get("completion")
        or row.get("expected_answer")
        or ""
    )
    return str(prompt).strip(), str(answer).strip()


def sft_batch(rows: list[dict], tokenizer, max_length: int, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    ids = []
    labels = []
    for row in rows:
        prompt = row.get("prompt") or row.get("question") or row.get("instruction") or ""
        answer = row.get("answer") or row.get("response") or row.get("output") or ""
        prompt_ids = tokenizer.encode(prompt)
        answer_ids = tokenizer.encode(answer) + [tokenizer.eot_id]
        seq = (prompt_ids + answer_ids)[:max_length]
        lab = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
        ids.append(seq)
        labels.append(lab)
    width = min(max_length, max(len(x) for x in ids))
    x = torch.zeros(len(ids), width, dtype=torch.long)
    y = torch.full((len(ids), width), -100, dtype=torch.long)
    for i, (seq, lab) in enumerate(zip(ids, labels)):
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        y[i, : len(lab)] = torch.tensor(lab, dtype=torch.long)
    return x.to(device), y.to(device)
