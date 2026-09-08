from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from gcw.config import TrainConfig, to_dict
from gcw.params import trainable_adapter_params
from gcw.runs import RunRecord


class RunTracker:
    """Mirror one canonical local metric stream to optional W&B logging."""

    def __init__(
        self, record: RunRecord, *, wandb_run: Any = None, log_every: int = 1
    ):
        self.record = record
        self.wandb_run = wandb_run
        self.log_every = max(1, log_every)
        self._last_metrics: dict[str, Any] = {}

    @classmethod
    def for_training(
        cls,
        config: TrainConfig,
        metadata: Mapping[str, Any],
        model: Any | None = None,
    ) -> RunTracker:
        """Create a local training record and its optional W&B mirror."""
        name = config.wandb.name or Path(config.output_dir).name
        record = RunRecord.create(
            kind="training",
            experiment_name=name,
            config=config,
            state_path=config.output_dir,
        )
        run_metadata = dict(metadata)
        if model is not None:
            run_metadata["trainable_params"] = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            run_metadata["lora_params"] = trainable_adapter_params(model)
        run = _start_wandb(config, run_metadata, record)
        record.bind_wandb(
            run_id=getattr(run, "id", None),
            url=getattr(run, "url", None),
            mode=config.wandb.mode,
        )
        return cls(record, wandb_run=run, log_every=config.wandb.log_every)

    def log(
        self, metrics: Mapping[str, Any], *, step: int, mirror: bool = True
    ) -> None:
        """Persist a metric event locally, then mirror it when configured."""
        compact = {
            key: value for key, value in metrics.items() if value is not None
        }
        self.record.append_metrics(compact, step=step)
        self._last_metrics.update(compact)
        if mirror and self.wandb_run is not None and step % self.log_every == 0:
            try:
                self.wandb_run.log(dict(compact), step=step)
            except Exception as error:  # W&B is a non-canonical remote sink.
                self.record.append_log(
                    f"W&B logging disabled after {type(error).__name__}: {error}"
                )
                if os.environ.get("GCW_WANDB_REQUIRED") == "1":
                    raise
                self.wandb_run = None

    def log_message(self, message: str) -> None:
        """Append a human-readable local log line."""
        self.record.append_log(message)

    def finish(
        self,
        *,
        summary: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Finalize local status and close the optional remote run."""
        final = dict(self._last_metrics)
        if summary is not None:
            final.update(summary)
        if self.wandb_run is not None:
            try:
                if summary is not None:
                    for key in (
                        "completed_step",
                        "planned_max_steps",
                        "stopped_at_checkpoint",
                    ):
                        if key in summary:
                            self.wandb_run.summary[key] = summary[key]
                self.wandb_run.finish(exit_code=1 if error is not None else 0)
            except TypeError:
                self.wandb_run.finish()
            except Exception as finish_error:
                self.record.append_log(
                    f"W&B finish failed with {type(finish_error).__name__}: {finish_error}"
                )
                if os.environ.get("GCW_WANDB_REQUIRED") == "1":
                    raise
        self.record.finish(summary=final, error=error)


def _start_wandb(
    config: TrainConfig, metadata: Mapping[str, Any], record: RunRecord
) -> Any:
    if not config.wandb.enabled:
        return None
    try:
        import wandb

        os.environ.setdefault("WANDB_DIR", str(record.paths.root / "_wandb"))
        init_kwargs: dict[str, Any] = {
            "project": config.wandb.project,
            "name": config.wandb.name or Path(config.output_dir).name,
            "group": config.wandb.group,
            "tags": list(config.wandb.tags),
            "config": {**to_dict(config), "run_budget": dict(metadata)},
        }
        if config.wandb.entity:
            init_kwargs["entity"] = config.wandb.entity
        if config.wandb.mode:
            init_kwargs["mode"] = config.wandb.mode
        if config.wandb.id:
            init_kwargs["id"] = config.wandb.id
        if config.wandb.resume:
            init_kwargs["resume"] = config.wandb.resume
        run = wandb.init(**init_kwargs)
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        run.define_metric("train/step")
        run.define_metric("*", step_metric="train/step")
        run.summary["architecture"] = config.model.architecture_name
        run.summary["training_method"] = (
            "truncated_bptt" if config.truncated_bptt_windows else "full_bptt"
        )
        run.summary["truncated_bptt_windows"] = config.truncated_bptt_windows
        for key in (
            "non_embedding_params",
            "processor_params",
            "trainable_params",
            "lora_params",
            "total_training_tokens",
            "total_training_flops",
            "loss_backend",
        ):
            if key in metadata:
                run.summary[key] = metadata[key]
        return run
    except Exception as error:
        record.append_log(
            f"W&B initialization failed with {type(error).__name__}: {error}"
        )
        if os.environ.get("GCW_WANDB_REQUIRED") == "1":
            raise
        return None
