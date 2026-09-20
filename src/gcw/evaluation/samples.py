"""Durable request outputs saved before harness filtering and scoring."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from gcw.config import EvalConfig


class PartialSamples:
    def __init__(self, config: EvalConfig, protocol: dict[str, Any]):
        directory = Path(
            config.partial_samples_dir or "data/eval-runs/partial-samples"
        )
        directory.mkdir(parents=True, exist_ok=True)
        rank = int(os.environ.get("RANK", "0"))
        self.path = directory / f"{uuid4().hex}.rank-{rank}.jsonl"
        self.call = 0
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "schema": "gcw.partial_samples.v1",
                        "type": "run",
                        "rank": rank,
                        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "config": asdict(config),
                        "protocol": protocol,
                    },
                    allow_nan=False,
                    default=str,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        print(f"Partial samples [rank {rank}]: {self.path}", flush=True)

    def start_call(self, request_type: str, total: int) -> None:
        self.call += 1
        self.request_type = request_type
        self.total = total

    def write_batch(
        self,
        requests: Sequence[Any],
        indices: Sequence[int],
        keys: Sequence[tuple[str, str, str]],
        responses: Sequence[Any],
        metrics: Any,
        *,
        generation: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        rows = []
        for offset, (index, key, response) in enumerate(
            zip(indices, keys, responses, strict=True)
        ):
            request = requests[index]
            row = {
                "type": "sample",
                "call": self.call,
                "request_type": self.request_type,
                "total_requests": self.total,
                "request_index": index,
                "task": key[0],
                "example_id": key[1],
                "request_id": key[2],
                "doc_id": getattr(request, "doc_id", None),
                "doc": getattr(request, "doc", None),
                "arguments": request.args,
                "response": response,
            }
            if generation is not None:
                row["generation"] = generation[offset]
            if metrics.path is not None:
                row["cache_update_metrics"] = {
                    "aggregation": "per_request_sums_of_per_application_head_statistics",
                    "values": metrics.requests[key],
                }
            rows.append(json.dumps(row, allow_nan=False, default=str) + "\n")
        with self.path.open("a", encoding="utf-8") as stream:
            stream.writelines(rows)
            stream.flush()
            os.fsync(stream.fileno())
