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
