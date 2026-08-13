from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gcw.architecture import (
    CompatibilityError,
    CompatibilityReport,
    admission_receipt_metadata,
    resolve_experiment,
    resolve_model,
)
from gcw.config import CacheProcessorConfig, DataConfig, ModelConfig, TrainConfig, load_train_config
from gcw.models import build_model
from gcw.provenance import canonical_json_sha256


def _tiny_model(variant: str = "tf") -> ModelConfig:
    processor = None
    if variant != "tf":
        processor = CacheProcessorConfig(
            process_every=4,
            topk=2,
            topk_heads=2,
            hidden_size=8,
            intermediate_size=16,
            num_heads=1,
            num_kv_heads=1,
        )
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        sequence_len=16,
        cache_processor=processor,
    )


def test_every_canonical_train_config_is_admitted_without_io() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs/train").rglob("*.yaml"))
    assert paths

    for path in paths:
        config = load_train_config(path)
        model_resolution = resolve_model(config.model)
        model, model_receipt = model_resolution.require()
        experiment_resolution = resolve_experiment(config, model_receipt)
        experiment, receipt = experiment_resolution.require()

        assert model.variant == config.model.variant
        assert experiment.model == model
        assert receipt.model_assembly_sha256 == model_receipt.receipt_sha256


def test_model_receipt_is_deterministic_canonical_and_self_authenticating() -> None:
    config = _tiny_model("longshot")
    first_model, first = resolve_model(config).require()
    second_model, second = resolve_model(replace(config)).require()

    assert first_model == second_model
    assert first == second
    assert first.receipt_sha256 == canonical_json_sha256(first.payload())
    assert "receipt_sha256" not in first.payload()
    assert "class" not in str(first.payload()).lower()
    assert "gcw.models" not in str(first.payload())


def test_processor_free_causal_lm_is_first_class() -> None:
    assembly, receipt = resolve_model(_tiny_model("tf")).require()

    assert assembly.processor is None
    assert receipt.processor_component is None
    assert "causal_lm" in receipt.capabilities
    assert "cache_intervention" not in receipt.capabilities


def test_model_resolution_collects_incompatibilities_without_constructing(monkeypatch) -> None:
    config = _tiny_model("longshot")
    assert config.cache_processor is not None
    config.variant = "not-a-variant"  # type: ignore[assignment]
    config.flow_processor = object()  # type: ignore[assignment]
    config.cache_processor.processor_mode = "workspace_broadcast"

    monkeypatch.setattr("gcw.models.build_language_model", lambda *_args, **_kwargs: pytest.fail("allocated"))
    resolution = resolve_model(config)

    assert isinstance(resolution.report, CompatibilityReport)
    assert not resolution.report.compatible
    assert resolution.assembly is None
    assert resolution.receipt is None
    assert {issue.code for issue in resolution.report.issues} >= {
        "unknown_variant",
        "flow_variant",
    }
    with pytest.raises(CompatibilityError, match="unknown model variant"):
        resolution.require()


def test_build_model_dispatches_only_after_admission_and_preserves_model_ownership() -> None:
    for variant in ("tf", "bt", "bgt", "icarus", "longshot"):
        config = _tiny_model(variant)
        model = build_model(config)
        assembly, _ = resolve_model(config).require()

        assert model.config.variant == assembly.variant
        assert all(not key.startswith("model.") for key in model.state_dict())
        if variant == "tf":
            assert assembly.processor is None
            assert not hasattr(model, "backbone")
        else:
            assert assembly.processor is not None
            assert hasattr(model, "backbone")
            assert any(key.startswith("backbone.") for key in model.state_dict())
            assert any(key.startswith("cache_processor.") for key in model.state_dict())


def test_unknown_variant_fails_before_constructor_dispatch(monkeypatch) -> None:
    config = _tiny_model("tf")
    config.variant = "flow_bgt"  # type: ignore[assignment]
    called: list[bool] = []
    monkeypatch.setattr("gcw.models.build_backbone", lambda *_: called.append(True))

    with pytest.raises(CompatibilityError, match="unknown model variant: flow_bgt"):
        build_model(config)
    assert called == []


def test_admission_receipt_metadata_is_self_authenticating_and_runtime_ready() -> None:
    config = TrainConfig(model=_tiny_model("bgt"), data=DataConfig(kind="random"))

    metadata = admission_receipt_metadata(config)

    assert metadata["schema"] == "gcw.admission.v1"
    for name in ("model_assembly_receipt", "composition_receipt"):
        document = metadata[name]
        assert document["receipt_sha256"] == canonical_json_sha256(document["payload"])
        assert document["payload"]


def test_experiment_receipt_separates_protocol_parameter_and_execution_axes() -> None:
    config = TrainConfig(
        model=_tiny_model("bgt"),
        data=DataConfig(kind="random"),
        mode="pretrain",
        distributed="ddp",
        trainable="cache_processor",
    )
    assembly, model_receipt = resolve_model(config.model).require()
    experiment, receipt = resolve_experiment(config, model_receipt).require()

    assert experiment.model == assembly
    assert experiment.data_protocol == "gcw.data.random.v1"
    assert experiment.objective_protocol == "gcw.pretrain.next_token_ce.v1"
    assert experiment.parameter_policy == "processor_only"
    assert experiment.execution_strategy == "ddp"
    assert receipt.trainable_ownership == ("cache_processor",)
    assert receipt.receipt_sha256 == canonical_json_sha256(receipt.payload())


def test_experiment_resolution_rejects_self_consistent_forged_model_receipt() -> None:
    from dataclasses import replace

    config = TrainConfig(model=_tiny_model("bgt"), data=DataConfig(kind="random"))
    _, canonical = resolve_model(config.model).require()
    forged = replace(canonical, generation_policy="forged-generation-policy")
    forged = replace(forged, receipt_sha256=canonical_json_sha256(forged.payload()))

    resolution = resolve_experiment(config, forged)

    assert not resolution.report.compatible
    assert any(issue.code == "model_receipt_noncanonical" for issue in resolution.report.issues)


def test_experiment_resolution_rejects_invented_protocol_components() -> None:
    config = TrainConfig(model=_tiny_model("bgt"), data=DataConfig(kind="random"))
    config.data.kind = "forged_data"
    config.checkpoint_format = "forged_checkpoint"
    _, model_receipt = resolve_model(config.model).require()

    resolution = resolve_experiment(config, model_receipt)

    assert not resolution.report.compatible
    assert {issue.code for issue in resolution.report.issues} >= {
        "unknown_data_kind",
        "unknown_checkpoint_format",
    }


def test_experiment_resolution_rejects_stale_model_receipt_and_collects_protocol_issues() -> None:
    config = TrainConfig(model=_tiny_model("bgt"), data=DataConfig(kind="random"))
    _, stale_receipt = resolve_model(_tiny_model("tf")).require()
    config.mode = "unsupported"  # type: ignore[assignment]
    config.distributed = "unsupported"  # type: ignore[assignment]

    resolution = resolve_experiment(config, stale_receipt)

    assert not resolution.report.compatible
    assert resolution.experiment is None
    assert resolution.receipt is None
    assert {issue.code for issue in resolution.report.issues} >= {
        "model_receipt_mismatch",
        "unknown_training_mode",
        "unknown_execution",
    }


def test_failed_training_admission_creates_no_output(tmp_path: Path) -> None:
    from gcw.training.core import train

    output = tmp_path / "must-not-exist"
    config = TrainConfig(
        model=_tiny_model("bgt"),
        data=DataConfig(kind="random"),
        output_dir=str(output),
    )
    config.mode = "unsupported"  # type: ignore[assignment]

    with pytest.raises(CompatibilityError):
        train(config)
    assert not output.exists()
