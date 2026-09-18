"""Native teacher-forced answer-only NLL, with externally anchored input identities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors.torch import load as load_safetensors

from gcw.checkpoint import (
    load_checkpoint_bytes,
    _upgrade_qwen35_gated_delta_keys,
)
from gcw.config import EvalConfig, to_dict
from gcw.evaluation.update_metrics import CacheUpdateMetrics
from gcw.data.sft import sft_batch
from gcw.data.tokenizers import (
    IntTokenizer,
    Tokenizer,
    TransformersTokenizer,
    get_tokenizer,
)
from gcw.models.cache_processors.diffusion import (
    pop_diffusion_field_evaluation_count,
    set_diffusion_eval_solver_steps,
)
from gcw.performance.config import ReferenceLossConfig
from gcw.performance.losses import configure_reference_loss_execution
from gcw.provenance import (
    canonical_json_sha256,
    file_sha256,
    normalized_prompt_text_sha256,
    package_source_members,
    source_tree_sha256,
    tensor_state_sha256,
)
from gcw.qualification.inputs import capture_file
from gcw.training.checkpoint.safetensors import (
    _parse_manifest_bytes,
    _verify_files,
)


@dataclass(frozen=True)
class AnswerNLLScore:
    nll_sum: float
    targets: int

    def __post_init__(self) -> None:
        if type(self.targets) is not int or self.targets <= 0:
            raise ValueError("zero or invalid answer target count")
        if not math.isfinite(self.nll_sum) or self.nll_sum < 0:
            raise ValueError("nonfinite or negative answer NLL")

    @property
    def nll(self) -> float:
        return self.nll_sum / self.targets


@torch.no_grad()
def answer_nll(logits: torch.Tensor, targets: torch.Tensor) -> AnswerNLLScore:
    """Score already-shifted native SFT labels; never shift them a second time.

    Only scored positions are cast to FP32, in chunks of at most 32 targets.
    Indexing by coordinates also avoids copying non-contiguous full logits.
    """
    if logits.ndim != 3 or logits.shape[:-1] != targets.shape:
        raise ValueError(
            "expected matching [batch, time, vocab] logits and targets"
        )
    if not logits.is_floating_point() or targets.dtype != torch.long:
        raise ValueError("logits must be floating point and targets int64")
    if logits.device != targets.device or logits.shape[-1] <= 0:
        raise ValueError("logit/target devices differ or vocabulary is empty")
    batch, time = (targets != -100).nonzero(as_tuple=True)
    if not batch.numel():
        raise ValueError("zero valid answer targets")
    total = 0.0
    for start in range(0, batch.numel(), 32):
        b, t = batch[start : start + 32], time[start : start + 32]
        labels = targets[b, t]
        values = logits[b, t].float()
        if bool(((labels < 0) | (labels >= logits.shape[-1])).any()):
            raise ValueError("answer target outside vocabulary")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("nonfinite scored logits")
        total += float(F.cross_entropy(values, labels, reduction="sum"))
    return AnswerNLLScore(total, int(batch.numel()))


@dataclass(frozen=True)
class AnswerNLLDataOptions:
    sequence_len: int
    tokenizer: str
    response_prefix: str = ""
    prompt_suffix: str = ""
    loss_scope: Literal["answer_only"] = "answer_only"

    def __post_init__(self) -> None:
        if type(self.sequence_len) is not int or self.sequence_len <= 1:
            raise ValueError("sequence_len must be an integer greater than one")
        if not isinstance(self.tokenizer, str) or not self.tokenizer:
            raise ValueError("an explicit tokenizer is required")
        if self.loss_scope != "answer_only":
            raise ValueError("evaluation requires answer_only input policy")
        if not isinstance(self.response_prefix, str) or not isinstance(
            self.prompt_suffix, str
        ):
            raise ValueError("prefix and suffix must be strings")

    @classmethod
    def from_fingerprint(
        cls, fingerprint: dict[str, Any]
    ) -> AnswerNLLDataOptions:
        data = fingerprint["data"]
        tokenizer = data["tokenizer"]
        if tokenizer == "hf":
            if not data.get("tokenizer_source"):
                raise ValueError("HF tokenizer requires an explicit source")
            tokenizer = "hf:" + data["tokenizer_source"]
        return cls(
            sequence_len=fingerprint["batch"]["sequence_len"],
            tokenizer=tokenizer,
            response_prefix=data["response_prefix"],
            prompt_suffix=data["prompt_suffix"],
            loss_scope=data["sft_loss_scope"],
        )


@dataclass(frozen=True)
class AnswerNLLData:
    path: Path
    sha256: str
    count: int

    def read(self) -> list[dict[str, str]]:
        """Hash and parse the same whole-file bytes, including all nonblank rows."""
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("expected data count must be positive")
        blob = capture_file(self.path, role="answer_nll_data")
        if blob.identity.sha256 != self.sha256:
            raise ValueError("data SHA256 differs from expected identity")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for line in blob.content.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                not isinstance(row, dict)
                or set(row) != {"prompt", "answer"}
                or not all(isinstance(value, str) for value in row.values())
            ):
                raise ValueError(
                    "only canonical string prompt/answer rows are supported"
                )
            identity = normalized_prompt_text_sha256(row["prompt"])
            if identity in seen:
                raise ValueError("data contains normalized-prompt duplicates")
            seen.add(identity)
            rows.append(row)
        if len(rows) != self.count:
            raise ValueError(
                f"whole-file count {len(rows)} != expected {self.count}"
            )
        return rows


@dataclass(frozen=True)
class AnswerNLLRow:
    row_index: int
    normalized_prompt_sha256: str
    row_sha256: str
    input_sha256: str
    targets_sha256: str
    attention_mask_sha256: str
    tokenizer_sha256: str
    options_sha256: str
    data_sha256: str
    length: int
    at_length_cap: bool
    nll_sum: float
    targets: int
    scheduled_field_evaluations: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.scheduled_field_evaluations) is not int
            or self.scheduled_field_evaluations < 0
        ):
            raise ValueError("invalid scheduled field evaluation count")


@dataclass(frozen=True)
class PairedAnswerNLL:
    reference: AnswerNLLScore
    candidate: AnswerNLLScore

    @property
    def delta_nll(self) -> float:
        """Candidate minus reference; negative means lower candidate NLL."""
        return self.candidate.nll - self.reference.nll


def paired_answer_nll(
    reference: Sequence[AnswerNLLRow],
    candidate: Sequence[AnswerNLLRow],
    *,
    expected_count: int,
) -> PairedAnswerNLL:
    """Compare complete, ordered, identity-matched rows in linear time.

    When reading records from disk, verify their enclosing receipt's externally
    anchored SHA256 first. Row identities alone are not signatures.
    """
    if (
        type(expected_count) is not int
        or expected_count <= 0
        or len(reference) != expected_count
        or len(candidate) != expected_count
    ):
        raise ValueError(
            "paired rows are empty or missing from the expected dataset"
        )
    seen: set[str] = set()
    left_total = right_total = 0.0
    targets = 0
    for index, (left, right) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        left_identity, right_identity = asdict(left), asdict(right)
        for key in ("nll_sum", "scheduled_field_evaluations"):
            left_identity.pop(key)
            right_identity.pop(key)
        if left.row_index != index or left_identity != right_identity:
            raise ValueError("paired rows reordered or identities mismatched")
        if left.normalized_prompt_sha256 in seen:
            raise ValueError("duplicate paired prompt identity")
        seen.add(left.normalized_prompt_sha256)
        AnswerNLLScore(left.nll_sum, left.targets)
        AnswerNLLScore(right.nll_sum, right.targets)
        left_total += left.nll_sum
        right_total += right.nll_sum
        targets += left.targets
    return PairedAnswerNLL(
        AnswerNLLScore(left_total, targets),
        AnswerNLLScore(right_total, targets),
    )


def load_answer_nll_rows(
    path: Path, *, expected_sha256: str
) -> tuple[AnswerNLLRow, ...]:
    """Read paired-comparison records from an externally anchored complete receipt."""
    blob = capture_file(path, role="answer_nll_receipt")
    if blob.identity.sha256 != expected_sha256:
        raise ValueError(
            "answer NLL receipt SHA256 differs from expected identity"
        )
    receipt = json.loads(blob.content)
    if (
        receipt.get("schema")
        not in {
            "gcw.answer_nll.v1",
            "gcw.answer_nll.v2",
        }
        or receipt.get("complete") is not True
    ):
        raise ValueError("incomplete or unsupported answer NLL receipt")
    payload = receipt["rows"]
    if (
        canonical_json_sha256(payload) != receipt["rows_sha256"]
        or len(payload) != receipt["n_examples"]
        or len(payload) != receipt["data"]["count"]
    ):
        raise ValueError(
            "answer NLL receipt rows differ from declared identity or count"
        )
    rows = tuple(AnswerNLLRow(**row) for row in payload)
    if (
        sum(row.scheduled_field_evaluations for row in rows)
        != receipt["scheduled_field_evaluations"]
    ):
        raise ValueError("scheduled field evaluation totals differ from rows")
    paired_answer_nll(rows, rows, expected_count=receipt["data"]["count"])
    options_hash = canonical_json_sha256(receipt["data_options"])
    for row in rows:
        if (
            row.data_sha256 != receipt["data"]["sha256"]
            or row.tokenizer_sha256 != receipt["tokenizer_sha256"]
            or row.options_sha256 != options_hash
        ):
            raise ValueError(
                "answer NLL row differs from receipt input identity"
            )
    total = AnswerNLLScore(
        math.fsum(row.nll_sum for row in rows), sum(row.targets for row in rows)
    )
    if receipt["result"] != {**asdict(total), "nll": total.nll}:
        raise ValueError("answer NLL receipt totals differ from rows")
    return rows


def tokenizer_identity_sha256(
    tokenizer: IntTokenizer | Tokenizer | TransformersTokenizer,
) -> str:
    """Bind native tokenizer settings and the actual tokenization pipeline."""
    identity: dict[str, Any] = {
        "type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "eot_id": tokenizer.eot_id,
    }
    if type(tokenizer) is IntTokenizer:
        identity["encoding"] = "utf8_bytes_mod_vocab_v1"
    elif type(tokenizer) is Tokenizer:
        identity["pipeline"] = tokenizer.enc.to_str()
    elif type(tokenizer) is TransformersTokenizer:
        identity["native_identity_sha256"] = tokenizer.identity_sha256()
        backend = getattr(tokenizer.enc, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "answer NLL requires a serializable fast HF tokenizer"
            )
        identity["pipeline"] = backend.to_str()
    else:
        raise ValueError("unsupported tokenizer identity implementation")
    return canonical_json_sha256(identity)


def _validate_loaded_config(
    saved: dict[str, Any], loaded: dict[str, Any]
) -> list[str]:
    expected = dict(saved)
    migrations: list[str] = []
    legacy = expected.get("legacy_bottleneck_generate")
    if legacy is not None:
        mode = "post_prefill_relative" if legacy else "windowed"
        if (
            type(legacy) is not bool
            or expected.get("processor_generation_mode") != mode
        ):
            raise ValueError("ambiguous legacy generation configuration")
        expected["legacy_bottleneck_generate"] = None
        migrations.append(
            "legacy_bottleneck_generate -> processor_generation_mode"
        )
    if loaded != expected:
        raise ValueError("loaded model configuration differs from checkpoint")
    return migrations


def evaluate_answer_nll(
    *,
    checkpoint: Path,
    expected_manifest_sha256: str,
    data: AnswerNLLData,
    out: Path,
    options: AnswerNLLDataOptions | None = None,
    config: EvalConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str | None = "auto",
    expected_step: int | None = None,
    expected_tokenizer_sha256: str | None = None,
    expected_loaded_state_sha256: str | None = None,
    disjoint_from: Sequence[AnswerNLLData] = (),
    solver_steps: int | None = None,
    cache_update_metrics_path: str | None = None,
) -> dict[str, Any]:
    """Load a native version-2 training checkpoint and atomically publish one receipt.

    Uses the native inference loader, not training resume. Processor-only payloads
    additionally require an externally anchored hash of the complete loaded state,
    because their manifest does not bind the external backbone's tensor bytes.
    """
    end_to_end_started = time.monotonic()
    checkpoint, out = Path(checkpoint), Path(out)
    configured_metrics_path = (
        config.cache_update_metrics_path if config else None
    )
    if (
        cache_update_metrics_path
        and configured_metrics_path
        and (
            Path(cache_update_metrics_path).resolve()
            != Path(configured_metrics_path).resolve()
        )
    ):
        raise ValueError("conflicting cache update metrics paths")
    metrics_path = cache_update_metrics_path or configured_metrics_path
    if metrics_path is not None and (
        checkpoint.resolve() in Path(metrics_path).resolve().parents
        or Path(metrics_path).resolve() == out.resolve()
    ):
        raise ValueError(
            "cache update metrics must be outside the checkpoint and separate from the receipt"
        )
    if config is not None and config.batch_size != 1:
        raise ValueError("answer NLL evaluation requires batch_size=1")
    if checkpoint.resolve() in out.resolve().parents:
        raise ValueError("evaluation output must be outside the checkpoint")
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    package_root = Path(__file__).resolve().parents[1]
    source_hash = source_tree_sha256(package_source_members(package_root))
    rows = data.read()
    prompts = {normalized_prompt_text_sha256(row["prompt"]) for row in rows}
    for excluded in disjoint_from:
        excluded_prompts = {
            normalized_prompt_text_sha256(row["prompt"])
            for row in excluded.read()
        }
        if prompts & excluded_prompts:
            raise ValueError(
                "normalized-prompt overlap with disjoint_from data"
            )
    manifest_blob = capture_file(
        checkpoint / "manifest.json", role="checkpoint_manifest"
    )
    if manifest_blob.identity.sha256 != expected_manifest_sha256:
        raise ValueError(
            "checkpoint manifest SHA256 differs from expected identity"
        )
    manifest = _parse_manifest_bytes(manifest_blob.content, checkpoint)
    _verify_files(checkpoint, manifest)
    payloads = {
        name: capture_file(checkpoint / name, role=name)
        for name in ("config.json", "train_state.json", "model.safetensors")
    }
    for name, blob in payloads.items():
        if blob.identity.sha256 != manifest["files"][name]:
            raise ValueError(f"checkpoint snapshot hash mismatch: {name}")
    saved_config = json.loads(payloads["config.json"].content)["model"]
    saved_state = json.loads(payloads["train_state.json"].content)
    step = saved_state.get("step")
    if type(step) is not int or step < 0 or step != manifest["step"]:
        raise ValueError("checkpoint manifest and train state steps differ")
    if expected_step is not None and step != expected_step:
        raise ValueError("unexpected checkpoint step")
    fingerprint = manifest.get("training_fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("model") != saved_config
    ):
        raise ValueError("checkpoint model differs from training fingerprint")
    if (
        fingerprint.get("data", {}).get("sft_loss_scope", "answer_only")
        != "answer_only"
    ):
        raise ValueError("checkpoint has non-answer_only input policy")
    fingerprint_data = fingerprint["data"]
    fingerprint_eval_config = EvalConfig(
        batch_size=1,
        apply_chat_template=fingerprint_data.get("sft_chat_template", False),
        enable_thinking=fingerprint_data.get("sft_enable_thinking", False),
    )
    if config is None:
        eval_config = fingerprint_eval_config
    else:
        eval_config = config
        if (
            eval_config.apply_chat_template
            != fingerprint_eval_config.apply_chat_template
            or eval_config.enable_thinking
            != fingerprint_eval_config.enable_thinking
        ):
            raise ValueError(
                "evaluation chat settings differ from the training fingerprint"
            )
    if options is None:
        options = AnswerNLLDataOptions.from_fingerprint(fingerprint)
    if eval_config.apply_chat_template and options.response_prefix:
        raise ValueError(
            "response_prefix must be empty with apply_chat_template"
        )
    tokenizer = get_tokenizer(options.tokenizer)
    tokenizer_hash = tokenizer_identity_sha256(tokenizer)
    if (
        expected_tokenizer_sha256 is not None
        and tokenizer_hash != expected_tokenizer_sha256
    ):
        raise ValueError("tokenizer SHA256 differs from expected identity")
    weights = load_safetensors(payloads["model.safetensors"].content)
    upgraded = _upgrade_qwen35_gated_delta_keys(weights)
    weight_migrations = sorted(set(weights) - set(upgraded))
    if len(weights) != len(upgraded):
        raise ValueError("checkpoint contains colliding legacy tensor names")
    processor_only = bool(upgraded) and all(
        key.startswith("cache_processor.") for key in upgraded
    )
    if processor_only and expected_loaded_state_sha256 is None:
        raise ValueError(
            "processor-only checkpoint requires expected_loaded_state_sha256"
        )
    backbone_files: dict[str, str] = {}
    if processor_only and saved_state.get("backbone_checkpoint"):
        backbone_path = (
            Path(saved_state["backbone_checkpoint"]) / "model.safetensors"
        )
        backbone_files[str(backbone_path.resolve())] = file_sha256(
            backbone_path
        )
    model, model_config, state = load_checkpoint_bytes(
        config_bytes=payloads["config.json"].content,
        train_state_bytes=payloads["train_state.json"].content,
        model_bytes=payloads["model.safetensors"].content,
        device=device,
        dtype=dtype,
    )
    migrations = _validate_loaded_config(saved_config, to_dict(model_config))
    if state != saved_state or to_dict(model.config) != to_dict(model_config):
        raise ValueError("loaded model metadata differs from checkpoint")
    if options.sequence_len > model_config.sequence_len:
        raise ValueError("evaluation sequence length exceeds model context")
    diffusion, processor = (
        model_config.diffusion_processor,
        model_config.cache_processor,
    )
    if diffusion is not None and diffusion.inference_mode != "deterministic":
        raise ValueError(
            "answer NLL requires deterministic diffusion inference"
        )
    if processor is not None and (
        processor.head_selection_method == "random"
        or (
            processor.iteration_mode == "internal"
            and processor.iterations > 1
            and processor.internal_iteration_score == "random"
        )
    ):
        raise ValueError("answer NLL requires deterministic head selection")
    actual_state = model.state_dict()
    if not processor_only and set(upgraded) != set(actual_state):
        raise ValueError(
            "checkpoint must contain the complete model or only the processor"
        )
    for name, tensor in upgraded.items():
        actual = actual_state[name]
        if not torch.equal(
            actual, tensor.to(device=actual.device, dtype=actual.dtype)
        ):
            raise ValueError(f"loaded tensor differs from checkpoint: {name}")
    loaded_hash = tensor_state_sha256(actual_state)
    loaded_dtypes = sorted(
        {str(tensor.dtype) for tensor in actual_state.values()}
    )
    if (
        expected_loaded_state_sha256 is not None
        and loaded_hash != expected_loaded_state_sha256
    ):
        raise ValueError(
            "loaded model state SHA256 differs from expected identity"
        )
    del weights, upgraded, actual_state, payloads, blob, tensor, actual
    loaded_config = to_dict(model_config)
    if solver_steps is not None:
        if type(solver_steps) is not int or solver_steps <= 0:
            raise ValueError("solver_steps must be a positive integer")
        set_diffusion_eval_solver_steps(model, solver_steps)
    model.eval()
    update_metrics = CacheUpdateMetrics(model, metrics_path)
    loss_execution = ReferenceLossConfig(
        mode="bounded", max_projected_positions=32
    )
    configure_reference_loss_execution(model, loss_execution)
    effective_config = to_dict(model.config)
    resolved_data_options = {
        **asdict(options),
        "apply_chat_template": eval_config.apply_chat_template,
        "enable_thinking": eval_config.enable_thinking,
    }
    options_hash = canonical_json_sha256(resolved_data_options)
    records: list[AnswerNLLRow] = []
    execution_started = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            x, y, mask = sft_batch(
                [row],
                tokenizer,
                options.sequence_len,
                device=device,
                response_prefix=options.response_prefix,
                prompt_suffix=options.prompt_suffix,
                chat_template=eval_config.apply_chat_template,
                enable_thinking=eval_config.enable_thinking,
                loss_scope=options.loss_scope,
                return_attention_mask=True,
            )
            if not bool(mask.all()) or not bool((y != -100).any()):
                raise ValueError(
                    f"row {index} has padding or zero valid answer targets"
                )
            pop_diffusion_field_evaluation_count(model)
            with update_metrics.batch([("answer_nll", str(index), "0")]):
                token_losses = model(
                    x,
                    y,
                    attention_mask=mask,
                    loss_reduction="none",
                    loss_backend="reference",
                )
            field_evaluations = pop_diffusion_field_evaluation_count(model)
            if (
                token_losses.shape != y.shape
                or token_losses.dtype != torch.float32
            ):
                raise ValueError(
                    "native loss must return FP32 losses aligned with targets"
                )
            scored = token_losses[y != -100]
            if not bool(torch.isfinite(scored).all()) or bool(
                (scored < 0).any()
            ):
                raise ValueError("nonfinite or negative answer NLL")
            score = AnswerNLLScore(
                math.fsum(float(chunk.sum()) for chunk in scored.split(32)),
                int(scored.numel()),
            )
            records.append(
                AnswerNLLRow(
                    row_index=index,
                    normalized_prompt_sha256=normalized_prompt_text_sha256(
                        row["prompt"]
                    ),
                    row_sha256=canonical_json_sha256(row),
                    input_sha256=tensor_state_sha256({"input": x}),
                    targets_sha256=tensor_state_sha256({"targets": y}),
                    attention_mask_sha256=tensor_state_sha256(
                        {"attention_mask": mask}
                    ),
                    tokenizer_sha256=tokenizer_hash,
                    options_sha256=options_hash,
                    data_sha256=data.sha256,
                    length=x.shape[1],
                    at_length_cap=x.shape[1] == options.sequence_len,
                    nll_sum=score.nll_sum,
                    targets=score.targets,
                    scheduled_field_evaluations=field_evaluations,
                )
            )
    elapsed_seconds = time.monotonic() - execution_started
    for source in (data, *disjoint_from):
        if file_sha256(source.path) != source.sha256:
            raise ValueError("data changed during evaluation")
    if file_sha256(checkpoint / "manifest.json") != expected_manifest_sha256:
        raise ValueError("checkpoint manifest changed during evaluation")
    _verify_files(checkpoint, manifest)
    for name, digest in backbone_files.items():
        if file_sha256(Path(name)) != digest:
            raise ValueError("external backbone changed during evaluation")
    if (
        tensor_state_sha256(model.state_dict()) != loaded_hash
        or to_dict(model.config) != effective_config
    ):
        raise ValueError(
            "model state or configuration changed during evaluation"
        )
    if tokenizer_identity_sha256(tokenizer) != tokenizer_hash:
        raise ValueError("tokenizer changed during evaluation")
    if source_tree_sha256(package_source_members(package_root)) != source_hash:
        raise ValueError("GCW source changed during evaluation")
    total = AnswerNLLScore(
        math.fsum(row.nll_sum for row in records),
        sum(row.targets for row in records),
    )
    row_payload = [asdict(row) for row in records]
    receipt = {
        "schema": "gcw.answer_nll.v2",
        "complete": True,
        "checkpoint": str(checkpoint.resolve()),
        "manifest_sha256": expected_manifest_sha256,
        "checkpoint_files": manifest["files"],
        "training_fingerprint": fingerprint,
        "loaded_step": step,
        "loaded_config": loaded_config,
        "effective_config": effective_config,
        "config_migrations": migrations,
        "tensor_key_migrations": weight_migrations,
        "loaded_state_sha256": loaded_hash,
        "loaded_dtypes": loaded_dtypes,
        "processor_only_payload": processor_only,
        "external_backbone_files": backbone_files,
        "data": {
            "path": str(data.path.resolve()),
            "sha256": data.sha256,
            "count": data.count,
        },
        "disjoint_from": [
            {
                "path": str(item.path.resolve()),
                "sha256": item.sha256,
                "count": item.count,
            }
            for item in disjoint_from
        ],
        "data_options": resolved_data_options,
        "eval_config": asdict(eval_config),
        "tokenizer_sha256": tokenizer_hash,
        "solver_steps_override": solver_steps,
        "scheduled_field_evaluations": sum(
            row.scheduled_field_evaluations for row in records
        ),
        "dtype_requested": str(dtype),
        "device": str(device),
        "torch_version": torch.__version__,
        "package_source_sha256": source_hash,
        "batch_size": 1,
        "elapsed_seconds": elapsed_seconds,
        "end_to_end_elapsed_seconds": time.monotonic() - end_to_end_started,
        "ce_dtype": "float32",
        "ce_target_chunk_size": 32,
        "reference_loss_execution": asdict(loss_execution),
        "protocol": "native_sft_shifted_answer_only_bounded_reference_ce_v1",
        "n_examples": len(records),
        "zero_target_rows": 0,
        "cap_rows": sum(row.at_length_cap for row in records),
        "rows": row_payload,
        "rows_sha256": canonical_json_sha256(row_payload),
        "result": {**asdict(total), "nll": total.nll},
    }
    update_metrics.write_receipt(out, receipt)
    return receipt
