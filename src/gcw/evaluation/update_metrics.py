"""Transactional per-request collection and per-example diagnostic sidecars."""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import Any
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist

from gcw.models.cache_processors.update_metrics import UPDATE_BATCH, UpdateBatch
from gcw.runtime import atomic_write_json

SCHEMA = "gcw.cache_update_metrics.v3"


class CacheUpdateMetrics:
    def __init__(
        self, model: Any, path: str | Path | None, *, resume: bool = False
    ) -> None:
        self.path = Path(path).resolve() if path else None
        self.model = model
        self.requests: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.completed: dict[tuple[str, str], dict[str, Any]] = {}
        if self.path is None:
            return
        config = getattr(model, "config", None)
        if (
            getattr(config, "variant", None) != "longshot"
            or getattr(config, "diffusion_processor", None) is None
            or getattr(model, "cache_processor", None) is None
        ):
            raise ValueError(
                "cache update metrics require enabled native Longshot diffusion"
            )
        if model.cache_processor.block.diffusion.recurrence_backend != "eager":
            raise ValueError(
                "cache update metrics require eager diffusion recurrence"
            )
        if self.path.exists():
            if not resume:
                raise FileExistsError(self.path)
            for line in self.path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row.get("schema") != SCHEMA:
                    raise ValueError(
                        "incompatible cache update metrics sidecar"
                    )
                key = (row["task"], row["example_id"])
                if key in self.completed:
                    raise ValueError("duplicate cache update metrics example")
                self.completed[key] = row

    @contextmanager
    def batch(
        self,
        keys: Sequence[tuple[str, str, str]],
        *,
        lengths: Sequence[int] | None = None,
    ) -> Iterator[None]:
        if self.path is None:
            yield
            return
        if self.model.training:
            raise ValueError("cache update metrics are evaluation-only")
        if UPDATE_BATCH.get() is not None:
            raise RuntimeError("nested cache update metric batches")
        batch = UpdateBatch(len(keys), self.model.get_device(), lengths)
        token = UPDATE_BATCH.set(batch)
        try:
            with torch.no_grad():
                yield
            rows = batch.rows()
        finally:
            UPDATE_BATCH.reset(token)
        for key, row in zip(keys, rows, strict=True):
            self.requests.setdefault(key, row)

    def validate_resume(self, results: dict) -> None:
        if self.path is None:
            return
        authenticated = set()
        expected_digests: list[tuple[str, str]] = []
        for runtime in results.get("gcw_runtime", {}).values():
            if not isinstance(runtime, dict):
                continue
            metadata = runtime.get("cache_update_metrics")
            if metadata:
                if metadata.get("schema") != SCHEMA or metadata.get(
                    "path"
                ) != str(self.path):
                    raise ValueError(
                        "cache update metrics resume policy differs"
                    )
                for task, digest in metadata["task_sha256"].items():
                    expected_digests.append((task, digest))
                    authenticated.add(task)
        actual_digests = self._task_digests(authenticated)
        if any(
            actual_digests[task] != digest for task, digest in expected_digests
        ):
            raise ValueError("cache update metrics sidecar missing or changed")
        done = set(results.get("results", {})) | set(results.get("groups", {}))
        if not done <= authenticated:
            raise ValueError(
                "completed evaluation lacks cache update metrics; use a fresh evaluation"
            )
        self.completed = {
            key: row
            for key, row in self.completed.items()
            if key[0] in authenticated
        }

    def _task_digests(self, tasks: set[str]) -> dict[str, str]:
        grouped: dict[str, list[dict[str, Any]]] = {
            task: [] for task in sorted(tasks)
        }
        for (task, _), row in sorted(self.completed.items()):
            if task in grouped:
                grouped[task].append(row)
        return {
            task: hashlib.sha256(
                json.dumps(rows, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            for task, rows in grouped.items()
        }

    def write_receipt(self, path: Path, receipt: dict[str, Any]) -> None:
        """Publish an exclusive receipt, rolling back an unreferenced new sidecar."""
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
        existing_sidecar = self.path is not None and self.path.exists()
        metadata = self.finish()
        sidecar_stat = None
        if metadata is not None:
            receipt["cache_update_metrics"] = metadata
            if self.path is not None and not existing_sidecar:
                sidecar_stat = self.path.stat()
        try:
            atomic_write_json(path, receipt, exclusive=True, allow_nan=False)
        except BaseException:
            if (
                sidecar_stat is not None
                and self.path is not None
                and self.path.exists()
                and os.path.samestat(sidecar_stat, self.path.stat())
            ):
                referenced = True
                try:
                    published = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(published, dict):
                        published_metrics = published.get(
                            "cache_update_metrics"
                        )
                        if published_metrics is None:
                            referenced = False
                        elif isinstance(published_metrics, dict) and isinstance(
                            published_metrics.get("path"), str
                        ):
                            referenced = (
                                Path(published_metrics["path"]).resolve()
                                == self.path
                            )
                except FileNotFoundError:
                    referenced = False
                except (OSError, ValueError):
                    pass
                if not referenced and os.path.samestat(
                    sidecar_stat, self.path.stat()
                ):
                    self.path.unlink()
            raise

    @staticmethod
    def _merge_values(target: dict[str, Any], values: dict[str, Any]) -> None:
        for key, value in values.items():
            total = target.setdefault(
                key,
                {
                    "sum": [0.0] * 4,
                    "applications": 0,
                    "heads": value["heads"],
                },
            )
            if total["heads"] != value["heads"]:
                raise ValueError(
                    "cache update metrics head population changed across requests"
                )
            total["sum"] = [a + b for a, b in zip(total["sum"], value["sum"])]
            total["applications"] += value["applications"]
            if "valid_head_observations" in value:
                total["valid_head_observations"] = (
                    total.get("valid_head_observations", 0)
                    + value["valid_head_observations"]
                )

    @staticmethod
    def _steps(values: dict[str, Any]) -> list[dict[str, Any]]:
        steps: dict[tuple[int, int], dict[str, Any]] = {}
        for key, value in sorted(values.items()):
            external, step, name = key.split(":")
            item = steps.setdefault(
                (int(external), int(step)),
                {
                    "external_iteration": int(external),
                    "solver_step": int(step),
                    "metrics": {},
                },
            )
            count = value["applications"]
            item["metrics"][name] = {
                **dict(
                    zip(
                        ("min", "max", "mean", "std"),
                        [x / count for x in value["sum"]],
                    )
                ),
                "applications": count,
                "heads": value["heads"],
            }
            if "valid_head_observations" in value:
                item["metrics"][name]["valid_head_observations"] = value[
                    "valid_head_observations"
                ]
        return [steps[key] for key in sorted(steps)]

    def finish(
        self, tasks: Sequence[str] | None = None
    ) -> dict[str, Any] | None:
        if self.path is None:
            return None
        requests = self.requests
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            gathered: list[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, requests)
            requests = {}
            for rank_requests in gathered:
                for key, row in rank_requests.items():
                    requests.setdefault(key, row)
        examples: dict[tuple[str, str], dict[str, Any]] = {}
        for (task, example, request), values in requests.items():
            record = examples.setdefault(
                (task, str(example)), {"requests": set(), "values": {}}
            )
            record["requests"].add(str(request))
            self._merge_values(record["values"], values)
        for (task, example), record in examples.items():
            phases: dict[str, dict[str, Any]] = {"prefill": {}, "decoding": {}}
            for key, value in record["values"].items():
                phase, metric = key.split(":", 1)
                phases[phase][metric] = value
            combined: dict[str, Any] = {}
            for values in phases.values():
                self._merge_values(combined, values)
            self.completed[(task, example)] = {
                "schema": SCHEMA,
                "task": task,
                "example_id": example,
                "request_count": len(record["requests"]),
                "steps": self._steps(combined),
                "phase_steps": {
                    phase: self._steps(values)
                    for phase, values in phases.items()
                },
            }
        task_names = set(tasks or []) | {key[0] for key in examples}
        if not distributed or dist.get_rank() == 0:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", dir=self.path.parent
            )
            temporary = Path(name)
            try:
                with os.fdopen(
                    descriptor, "w", encoding="utf-8", newline="\n"
                ) as stream:
                    for _, row in sorted(self.completed.items()):
                        stream.write(
                            json.dumps(row, allow_nan=False, sort_keys=True)
                            + "\n"
                        )
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
        self.requests.clear()
        return {
            "schema": SCHEMA,
            "path": str(self.path),
            "complete": True,
            "record_count": sum(key[0] in task_names for key in self.completed),
            "aggregation": "mean_of_per_application_head_statistics",
            "std_correction": 0,
            "solver_step_scaling": "included",
            "writeback_gate_scaling": "excluded",
            "first_step_baseline": "zero",
            "task_sha256": self._task_digests(task_names),
        }


def request_metric_keys(requests: Sequence[Any]) -> list[tuple[str, str, str]]:
    occurrences: dict[tuple[str, str, str], int] = {}
    keys = []
    for index, request in enumerate(requests):
        task = getattr(request, "task_name", None) or "requests"
        example = getattr(request, "doc_id", None)
        example = str(index if example is None else example)
        idx = str(getattr(request, "idx", 0))
        key = (task, example, idx)
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        repeats = getattr(request, "repeats", None)
        if isinstance(repeats, int) and repeats > 0:
            occurrence %= repeats
        keys.append((task, example, f"{idx}:{occurrence}"))
    return keys
