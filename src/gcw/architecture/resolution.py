from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from gcw.provenance import canonical_json_sha256
from gcw.architecture.contracts import (
    COMPOSITION_SCHEMA,
    MODEL_ASSEMBLY_SCHEMA,
    MODEL_CONFIG_CONTRACT,
    TRAIN_CONFIG_CONTRACT,
    CompatibilityIssue,
    CompatibilityReport,
    _config_payload,
)
from gcw.architecture.descriptors import (
    CompositionReceipt,
    ExperimentResolution,
    MemoryDomainDescriptor,
    ModelAssemblyReceipt,
    ModelResolution,
    ProcessorAssembly,
    ResolvedExperiment,
    ResolvedModelAssembly,
)
from gcw.architecture.validation import (
    _experiment_issues,
    _model_issues,
    _unique_issues,
)

if TYPE_CHECKING:
    from gcw.config import ModelConfig, TrainConfig


def resolve_model(config: ModelConfig) -> ModelResolution:
    issues = _model_issues(config)
    report = CompatibilityReport(tuple(issues))
    if not report.compatible:
        return ModelResolution(None, None, report)

    model_config_sha256 = canonical_json_sha256(_config_payload(config))
    processor = _processor_assembly(config)
    domains = _memory_domains(config, processor)
    capabilities = _model_capabilities(config, processor, domains)
    assembly = ResolvedModelAssembly(
        variant=config.variant,
        family=config.family,
        backbone=f"gcw.backbone.{config.family}.v1",
        processor=processor,
        memory_domains=domains,
        generation_policy="gcw.native_greedy_generation.v1",
        checkpoint_abi="gcw.native_safetensors.v1",
        accounting_provider="gcw.symbolic_accounting.v1",
        capabilities=capabilities,
    )
    unsigned = ModelAssemblyReceipt(
        schema=MODEL_ASSEMBLY_SCHEMA,
        model_config_contract=MODEL_CONFIG_CONTRACT,
        model_config_sha256=model_config_sha256,
        backbone_component=assembly.backbone,
        backbone_contract="gcw.native_causal_lm.v1",
        memory_domains=domains,
        processor_component=None if processor is None else processor.component,
        processor_config_sha256=None if processor is None else processor.config_sha256,
        processor_pipeline=() if processor is None else processor.pipeline,
        generation_policy=assembly.generation_policy,
        checkpoint_abi=assembly.checkpoint_abi,
        accounting_abi=assembly.accounting_provider,
        capabilities=capabilities,
        receipt_sha256="",
    )
    receipt = replace(unsigned, receipt_sha256=canonical_json_sha256(unsigned.payload()))
    return ModelResolution(assembly, receipt, report)


def resolve_experiment(
    config: TrainConfig,
    model_receipt: ModelAssemblyReceipt,
) -> ExperimentResolution:
    model_resolution = resolve_model(config.model)
    issues = list(model_resolution.report.issues)
    expected_model_digest = canonical_json_sha256(_config_payload(config.model)) if not issues else None
    if model_receipt.receipt_sha256 != canonical_json_sha256(model_receipt.payload()):
        issues.append(
            CompatibilityIssue(
                "model_receipt_digest",
                "model assembly receipt digest does not authenticate its payload",
                "model_receipt.receipt_sha256",
            )
        )
    if model_resolution.receipt is not None and model_receipt != model_resolution.receipt:
        issues.append(
            CompatibilityIssue(
                "model_receipt_noncanonical",
                "model assembly receipt differs from the receipt resolved from train.model",
                "model_receipt",
            )
        )
    if expected_model_digest is not None and model_receipt.model_config_sha256 != expected_model_digest:
        issues.append(
            CompatibilityIssue(
                "model_receipt_mismatch",
                "model assembly receipt does not describe train.model",
                "model_receipt.model_config_sha256",
            )
        )
    issues.extend(_experiment_issues(config))
    report = CompatibilityReport(tuple(_unique_issues(issues)))
    if not report.compatible or model_resolution.assembly is None:
        return ExperimentResolution(None, None, report)

    model = model_resolution.assembly
    data_protocol = _data_protocol(config)
    objective_protocol = _objective_protocol(config)
    parameter_policy, trainable_ownership = _parameter_policy(config, model)
    schedule = _intervention_schedule(config, model)
    capabilities = tuple(sorted(set(model.capabilities) | _experiment_capabilities(config)))
    experiment = ResolvedExperiment(
        model=model,
        data_protocol=data_protocol,
        objective_protocol=objective_protocol,
        parameter_policy=parameter_policy,
        execution_strategy=config.distributed,
        trainable_ownership=trainable_ownership,
        intervention_schedule=schedule,
        capabilities=capabilities,
    )
    processor = config.model.cache_processor
    recurrence_plan = "none" if processor is None else f"{processor.iteration_mode}:{processor.iterations}"
    if config.model.diffusion_processor is not None:
        flow_plan = f"task_directed_diffusion:{config.model.diffusion_processor.num_steps}"
    elif config.model.flow_processor is not None:
        flow_plan = f"latent_flow:{config.model.flow_processor.num_steps}"
    else:
        flow_plan = "none"
    checkpoint_contracts = ["gcw.native_final.v1"]
    if config.resume is not None or config.mode in {"pretrain", "sft"}:
        checkpoint_contracts.append("gcw.training.resume.v2")
    unsigned = CompositionReceipt(
        schema=COMPOSITION_SCHEMA,
        model_assembly_sha256=model_receipt.receipt_sha256,
        train_config_contract=TRAIN_CONFIG_CONTRACT,
        train_config_sha256=canonical_json_sha256(_config_payload(config)),
        data_protocol=data_protocol,
        objective_protocol=objective_protocol,
        parameter_policy=parameter_policy,
        trainable_ownership=trainable_ownership,
        execution_strategy=config.distributed,
        intervention_schedule=schedule,
        recurrence_plan=recurrence_plan,
        flow_plan=flow_plan,
        checkpoint_contracts=tuple(checkpoint_contracts),
        provenance_contracts=("gcw.source_code_identity.v2", "gcw.composition.v1"),
        capabilities=capabilities,
        receipt_sha256="",
    )
    receipt = replace(unsigned, receipt_sha256=canonical_json_sha256(unsigned.payload()))
    return ExperimentResolution(experiment, receipt, report)


def _processor_assembly(config: ModelConfig) -> ProcessorAssembly | None:
    if config.variant == "tf":
        return None
    processor = config.cache_processor
    if processor is None:
        raise RuntimeError("admitted processed model has no processor configuration")
    components = {
        "bt": "gcw.processor.per_layer.v1",
        "bgt": "gcw.processor.global.v1",
        "icarus": "gcw.processor.per_head_temporal.v1",
        "longshot": "gcw.processor.persistent_workspace.v1",
    }
    writes = ["dense_kv"]
    if processor.linear_state_cache:
        writes.append("qwen35_linear_recurrent")
    return ProcessorAssembly(
        component=components[config.variant],
        config_sha256=canonical_json_sha256(_config_payload(processor)),
        pipeline=("select", "encode", "process", "decode", "writeback"),
        writable_domains=tuple(writes),
        recurrence_mode=f"{processor.iteration_mode}:{processor.iterations}",
        flow_solver_steps=(
            config.diffusion_processor.num_steps
            if config.diffusion_processor is not None
            else 0
            if config.flow_processor is None
            else config.flow_processor.num_steps
        ),
    )


def _memory_domains(
    config: ModelConfig,
    processor: ProcessorAssembly | None,
) -> tuple[MemoryDomainDescriptor, ...]:
    writable = processor is not None
    kv_writes: tuple[str, ...] = ("keys", "values") if writable else ()
    if processor is not None and config.cache_processor is not None and config.cache_processor.update_stream == "v":
        kv_writes = ("values",)
    domains: list[MemoryDomainDescriptor] = [
        MemoryDomainDescriptor(
            domain="dense_kv",
            layout="per_cache_layer_grouped_query",
            positional_frame="absolute_rope",
            addressable=True,
            branchable=True,
            exact_truncate_replay=True,
            differentiable=True,
            allowed_writes=kv_writes,
        )
    ]
    if config.family == "gemma4":
        domains.append(
            MemoryDomainDescriptor(
                domain="sliding_shared_kv",
                layout="mixed_sliding_full_shared",
                positional_frame="layer_specific_rope",
                addressable=True,
                branchable=True,
                exact_truncate_replay=True,
                differentiable=True,
                allowed_writes=kv_writes,
            )
        )
    if (
        config.family in {"qwen3_5", "qwen3_5_moe"}
        and config.layer_types is not None
        and "linear_attention" in config.layer_types
    ):
        domains.extend(
            (
                MemoryDomainDescriptor(
                    domain="qwen35_convolution_history",
                    layout="per_linear_layer_conv_window",
                    positional_frame="causal_history",
                    addressable=False,
                    branchable=True,
                    exact_truncate_replay=False,
                    differentiable=True,
                    allowed_writes=(),
                ),
                MemoryDomainDescriptor(
                    domain="qwen35_linear_recurrent",
                    layout="per_layer_gated_delta_matrix",
                    positional_frame="recurrent",
                    addressable=False,
                    branchable=True,
                    exact_truncate_replay=False,
                    differentiable=True,
                    allowed_writes=(
                        ("additive_learned_perturbation",)
                        if writable and config.cache_processor is not None and config.cache_processor.linear_state_cache
                        else ()
                    ),
                ),
            )
        )
    if config.variant == "longshot":
        domains.append(
            MemoryDomainDescriptor(
                domain="longshot_workspace",
                layout="persistent_selected_cache_heads",
                positional_frame="selected_frontier",
                addressable=True,
                branchable=True,
                exact_truncate_replay=True,
                differentiable=True,
                allowed_writes=("workspace", "frontier_writeback"),
            )
        )
    return tuple(domains)


def _model_capabilities(
    config: ModelConfig,
    processor: ProcessorAssembly | None,
    domains: tuple[MemoryDomainDescriptor, ...],
) -> tuple[str, ...]:
    capabilities = {"causal_lm", "native_generate", "native_generate_fast", "strict_checkpoint"}
    if processor is not None:
        capabilities.update({"cache_intervention", "processor_training"})
    if config.flow_processor is not None:
        capabilities.add("latent_flow")
    if config.diffusion_processor is not None:
        capabilities.add("task_directed_diffusion")
    if config.variant == "longshot":
        capabilities.add("persistent_workspace")
    if any(domain.domain == "qwen35_linear_recurrent" for domain in domains):
        capabilities.add("qwen35_recurrent_state")
    return tuple(sorted(capabilities))


def _data_protocol(config: TrainConfig) -> str:
    return f"gcw.data.{config.data.kind}.v1"


def _objective_protocol(config: TrainConfig) -> str:
    if config.logical_pair_grouping:
        return "gcw.sft.answer_only_ce.adjacent_pair_token_mean.v1"
    if config.mode == "sft":
        return f"gcw.sft.{config.data.sft_loss_scope}_ce.v1"
    return "gcw.pretrain.next_token_ce.v1"


def _parameter_policy(
    config: TrainConfig,
    model: ResolvedModelAssembly,
) -> tuple[str, tuple[str, ...]]:
    if config.trainable == "cache_processor":
        return "processor_only", ("cache_processor",)
    if config.trainable == "backbone":
        return "backbone_only", ("backbone",)
    return "full_model", ("backbone",) if model.processor is None else ("backbone", "cache_processor")


def _intervention_schedule(
    config: TrainConfig,
    model: ResolvedModelAssembly,
) -> tuple[str, ...]:
    if model.processor is None or config.model.cache_processor is None:
        return ()
    processor = config.model.cache_processor
    schedule = [f"cache_every:{processor.process_every}"]
    if processor.iterations > 1:
        schedule.append(f"recurrence:{processor.iteration_mode}:{processor.iterations}")
    if config.model.flow_processor is not None:
        schedule.append(f"latent_flow_steps:{config.model.flow_processor.num_steps}")
    if config.model.diffusion_processor is not None:
        schedule.append(f"task_directed_diffusion_steps:{config.model.diffusion_processor.num_steps}")
    return tuple(schedule)


def _experiment_capabilities(config: TrainConfig) -> set[str]:
    capabilities = {f"data:{config.data.kind}", f"mode:{config.mode}", f"execution:{config.distributed}"}
    if config.resume is not None:
        capabilities.add("exact_resume")
    if config.data.canonical_execution:
        capabilities.add("canonical_evidence")
    if config.logical_pair_grouping:
        capabilities.add("logical_pair_grouping")
    if config.performance.adaptive_batching.enabled:
        capabilities.add("adaptive_adjacent_pair_batching")
    return capabilities
