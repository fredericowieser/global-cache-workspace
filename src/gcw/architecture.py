from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, cast
from gcw.performance.config import PerformanceConfig
from gcw.provenance import canonical_json_sha256

if TYPE_CHECKING:
    from gcw.config import ModelConfig, TrainConfig


MODEL_ASSEMBLY_SCHEMA = "gcw.model_assembly.v1"
COMPOSITION_SCHEMA = "gcw.composition.v1"
MODEL_CONFIG_CONTRACT = "gcw.model_config.v1"
TRAIN_CONFIG_CONTRACT = "gcw.training.core_config.v1"

_SUPPORTED_VARIANTS = ("tf", "bt", "bgt", "icarus", "longshot")
_SUPPORTED_FAMILIES = ("smollm", "llama", "qwen3", "qwen3_5", "qwen3_5_moe", "gemma4")
_SUPPORTED_DATA_KINDS = (
    "random",
    "fineweb",
    "sft",
)
_SUPPORTED_TRAINABLE_POLICIES = ("all", "backbone", "lora", "cache_processor")
_SUPPORTED_CHECKPOINT_FORMATS = ("native_safetensors",)


def _config_payload(value: object) -> object:
    if isinstance(value, PerformanceConfig):
        payload = {field.name: _config_payload(getattr(value, field.name)) for field in fields(value)}
        if value.reference_loss.mode == "legacy":
            payload.pop("reference_loss")
        if not value.longshot_full_history.enabled:
            payload.pop("longshot_full_history")
        if not value.kv_score_sidecar.enabled:
            # Preserve receipts created before this default-off execution policy existed.
            payload.pop("kv_score_sidecar")
        if not value.triton_cache_movement.enabled:
            # Preserve receipts created before this default-off execution policy existed.
            payload.pop("triton_cache_movement")
        if not value.optimized_step.enabled:
            # Preserve receipts created before this default-off execution policy existed.
            payload.pop("optimized_step")
        if value.noisy_diffusion.recurrence_backend == "eager":
            # Preserve receipts created before this default-off execution policy existed.
            payload.pop("noisy_diffusion")
        return payload
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _config_payload(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (list, tuple)):
        return [_config_payload(item) for item in value]
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return {key: _config_payload(item) for key, item in mapping.items()}
    return value


def _has_fields(value: object, *names: str) -> bool:
    return all(hasattr(value, name) for name in names)


_MODEL_CONFIG_VALIDATION_CODES = {
    "diffusion_variant",
    "flow_recurrence",
    "flow_variant",
    "linear_state_encoder",
    "linear_state_family",
    "linear_state_query",
    "linear_state_variant",
    "linear_state_write",
    "workspace_refinement_variant",
    "workspace_refinement_write",
}
_TRAIN_CONFIG_VALIDATION_CODES = {
    "backbone_with_init",
    "canonical_resume_manifest",
    "init_with_resume",
    "pretrain_data",
    "processor_training_without_processor",
    "sft_data",
    "unknown_checkpoint_format",
    "unknown_data_kind",
    "unknown_trainable_policy",
}


def require_model_config(config: ModelConfig) -> None:
    issues = tuple(issue for issue in _model_issues(config) if issue.code in _MODEL_CONFIG_VALIDATION_CODES)
    CompatibilityReport(issues).require()


def require_train_config(config: TrainConfig) -> None:
    issues = tuple(issue for issue in _experiment_issues(config) if issue.code in _TRAIN_CONFIG_VALIDATION_CODES)
    CompatibilityReport(issues).require()


@dataclass(frozen=True)
class CompatibilityIssue:
    code: str
    message: str
    path: str


@dataclass(frozen=True)
class CompatibilityReport:
    issues: tuple[CompatibilityIssue, ...] = ()

    @property
    def compatible(self) -> bool:
        return not self.issues

    def require(self) -> None:
        if self.issues:
            raise CompatibilityError(self)


class CompatibilityError(ValueError):
    def __init__(self, report: CompatibilityReport) -> None:
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


@dataclass(frozen=True)
class MemoryDomainDescriptor:
    domain: str
    layout: str
    positional_frame: str
    addressable: bool
    branchable: bool
    exact_truncate_replay: bool
    differentiable: bool
    allowed_writes: tuple[str, ...]

    def payload(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "layout": self.layout,
            "positional_frame": self.positional_frame,
            "addressable": self.addressable,
            "branchable": self.branchable,
            "exact_truncate_replay": self.exact_truncate_replay,
            "differentiable": self.differentiable,
            "allowed_writes": list(self.allowed_writes),
        }


@dataclass(frozen=True)
class ProcessorAssembly:
    component: str
    config_sha256: str
    pipeline: tuple[str, ...]
    writable_domains: tuple[str, ...]
    recurrence_mode: str
    flow_solver_steps: int

    def payload(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "config_sha256": self.config_sha256,
            "pipeline": list(self.pipeline),
            "writable_domains": list(self.writable_domains),
            "recurrence_mode": self.recurrence_mode,
            "flow_solver_steps": self.flow_solver_steps,
        }


@dataclass(frozen=True)
class ResolvedModelAssembly:
    variant: str
    family: str
    backbone: str
    processor: ProcessorAssembly | None
    memory_domains: tuple[MemoryDomainDescriptor, ...]
    generation_policy: str
    checkpoint_abi: str
    accounting_provider: str
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class ModelAssemblyReceipt:
    schema: str
    model_config_contract: str
    model_config_sha256: str
    backbone_component: str
    backbone_contract: str
    memory_domains: tuple[MemoryDomainDescriptor, ...]
    processor_component: str | None
    processor_config_sha256: str | None
    processor_pipeline: tuple[str, ...]
    generation_policy: str
    checkpoint_abi: str
    accounting_abi: str
    capabilities: tuple[str, ...]
    receipt_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_config_contract": self.model_config_contract,
            "model_config_sha256": self.model_config_sha256,
            "backbone": {
                "component": self.backbone_component,
                "contract": self.backbone_contract,
            },
            "memory_domains": [domain.payload() for domain in self.memory_domains],
            "processor": (
                None
                if self.processor_component is None
                else {
                    "component": self.processor_component,
                    "config_sha256": self.processor_config_sha256,
                    "pipeline": list(self.processor_pipeline),
                }
            ),
            "generation_policy": self.generation_policy,
            "checkpoint_abi": self.checkpoint_abi,
            "accounting_abi": self.accounting_abi,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class ModelResolution:
    assembly: ResolvedModelAssembly | None
    receipt: ModelAssemblyReceipt | None
    report: CompatibilityReport

    def require(self) -> tuple[ResolvedModelAssembly, ModelAssemblyReceipt]:
        self.report.require()
        if self.assembly is None or self.receipt is None:
            raise RuntimeError("compatible model resolution has no assembly receipt")
        return self.assembly, self.receipt


@dataclass(frozen=True)
class ResolvedExperiment:
    model: ResolvedModelAssembly
    data_protocol: str
    objective_protocol: str
    parameter_policy: str
    execution_strategy: str
    trainable_ownership: tuple[str, ...]
    intervention_schedule: tuple[str, ...]
    capabilities: tuple[str, ...]


@dataclass(frozen=True)
class CompositionReceipt:
    schema: str
    model_assembly_sha256: str
    train_config_contract: str
    train_config_sha256: str
    data_protocol: str
    objective_protocol: str
    parameter_policy: str
    trainable_ownership: tuple[str, ...]
    execution_strategy: str
    intervention_schedule: tuple[str, ...]
    recurrence_plan: str
    flow_plan: str
    checkpoint_contracts: tuple[str, ...]
    provenance_contracts: tuple[str, ...]
    capabilities: tuple[str, ...]
    receipt_sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_assembly_sha256": self.model_assembly_sha256,
            "train_config_contract": self.train_config_contract,
            "train_config_sha256": self.train_config_sha256,
            "data_protocol": self.data_protocol,
            "objective_protocol": self.objective_protocol,
            "parameter_policy": self.parameter_policy,
            "trainable_ownership": list(self.trainable_ownership),
            "execution_strategy": self.execution_strategy,
            "intervention_schedule": list(self.intervention_schedule),
            "recurrence_plan": self.recurrence_plan,
            "flow_plan": self.flow_plan,
            "checkpoint_contracts": list(self.checkpoint_contracts),
            "provenance_contracts": list(self.provenance_contracts),
            "capabilities": list(self.capabilities),
        }


def _receipt_metadata(receipt: ModelAssemblyReceipt | CompositionReceipt) -> dict[str, Any]:
    payload = receipt.payload()
    return {
        "payload": payload,
        "receipt_sha256": canonical_json_sha256(payload),
    }


def admission_receipt_metadata(config: TrainConfig) -> dict[str, Any]:
    """Return authenticated model and experiment receipts for runtime artifacts."""
    _, model_receipt = resolve_model(config.model).require()
    _, composition_receipt = resolve_experiment(config, model_receipt).require()
    return {
        "schema": "gcw.admission.v1",
        "model_assembly_receipt": _receipt_metadata(model_receipt),
        "composition_receipt": _receipt_metadata(composition_receipt),
    }


@dataclass(frozen=True)
class ExperimentResolution:
    experiment: ResolvedExperiment | None
    receipt: CompositionReceipt | None
    report: CompatibilityReport

    def require(self) -> tuple[ResolvedExperiment, CompositionReceipt]:
        self.report.require()
        if self.experiment is None or self.receipt is None:
            raise RuntimeError("compatible experiment resolution has no composition receipt")
        return self.experiment, self.receipt


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


def _model_issues(config: ModelConfig) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(CompatibilityIssue(code, message, path))

    if config.variant not in _SUPPORTED_VARIANTS:
        add("unknown_variant", f"unknown model variant: {config.variant}", "model.variant")
    if config.family not in _SUPPORTED_FAMILIES:
        add("unknown_family", f"unknown model family: {config.family}", "model.family")

    processor = config.cache_processor
    if config.variant != "tf" and processor is None:
        add("processor_required", f"{config.variant} requires cache_processor config", "model.cache_processor")
    if processor is not None and not _has_fields(processor, "processor_mode", "update_stream", "iterations"):
        add("processor_config", "cache_processor must be a CacheProcessorConfig", "model.cache_processor")
        processor = None

    flow = config.flow_processor
    if flow is not None and not _has_fields(flow, "num_steps", "time_conditioning"):
        add("flow_config", "flow_processor must be a FlowProcessorConfig", "model.flow_processor")
    if flow is not None and config.variant not in {"bgt", "icarus", "longshot"}:
        add(
            "flow_variant",
            "flow_processor requires variant 'bgt', 'icarus', or 'longshot'",
            "model.flow_processor",
        )
    diffusion = config.diffusion_processor
    if diffusion is not None and not _has_fields(diffusion, "num_steps", "noise_scale"):
        add(
            "diffusion_config",
            "diffusion_processor must be a DiffusionProcessorConfig",
            "model.diffusion_processor",
        )
    if diffusion is not None and config.variant != "bgt":
        add(
            "diffusion_variant",
            "diffusion_processor requires variant 'bgt'",
            "model.diffusion_processor",
        )
    if diffusion is not None and flow is not None:
        add(
            "iterative_processor_conflict",
            "flow_processor and diffusion_processor are mutually exclusive",
            "model.diffusion_processor",
        )

    refinement = config.workspace_flow_refinement
    if refinement is not None and not _has_fields(refinement, "solver", "solver_steps"):
        add(
            "workspace_refinement_config",
            "workspace_flow_refinement must be a WorkspaceFlowRefinementConfig",
            "model.workspace_flow_refinement",
        )
    if refinement is not None and config.variant != "longshot":
        add(
            "workspace_refinement_variant",
            "workspace_flow_refinement requires variant 'longshot'",
            "model.workspace_flow_refinement",
        )

    if processor is not None:
        if refinement is not None and processor.update_stream != "v":
            add(
                "workspace_refinement_write",
                "workspace_flow_refinement requires value-only cache writes",
                "model.cache_processor.update_stream",
            )
        if config.variant == "bt":
            if processor.processor_mode != "standard":
                add(
                    "bt_processor_mode",
                    f"cache_processor.processor_mode='{processor.processor_mode}' is currently BGT-only",
                    "model.cache_processor.processor_mode",
                )
            if processor.score != "attn_l2":
                add(
                    "bt_score",
                    "cache_processor.score='kv_norm' is currently BGT-only",
                    "model.cache_processor.score",
                )
        elif config.variant == "bgt":
            if processor.processor_mode == "workspace_broadcast" and processor.update_stream != "v":
                add(
                    "workspace_broadcast_write",
                    "workspace_broadcast requires cache_processor.update_stream='v'",
                    "model.cache_processor.update_stream",
                )
            if processor.processor_mode == "workspace_broadcast" and processor.score != "kv_norm":
                add(
                    "workspace_broadcast_score",
                    "workspace_broadcast requires cache_processor.score='kv_norm'",
                    "model.cache_processor.score",
                )
            if processor.processor_mode == "head_mask_broadcast" and processor.update_stream != "kv":
                add(
                    "head_mask_write",
                    "head_mask_broadcast requires cache_processor.update_stream='kv'",
                    "model.cache_processor.update_stream",
                )
            if processor.processor_mode == "head_mask_broadcast" and processor.score != "kv_norm":
                add(
                    "head_mask_score",
                    "head_mask_broadcast requires cache_processor.score='kv_norm'",
                    "model.cache_processor.score",
                )
        elif config.variant in {"icarus", "longshot"} and processor.processor_mode != "standard":
            add(
                "processor_mode",
                f"{config.variant.capitalize()} supports only cache_processor.processor_mode='standard'",
                "model.cache_processor.processor_mode",
            )
        if (
            config.variant == "longshot"
            and flow is not None
            and processor.iteration_mode == "internal"
            and processor.iterations > 1
        ):
            add(
                "flow_recurrence",
                "Flow-Longshot does not support internal cache_processor iterations greater than one",
                "model.cache_processor.iterations",
            )
        if processor.linear_state_cache:
            if config.variant != "longshot":
                add(
                    "linear_state_variant",
                    "cache_processor.linear_state_cache is supported only for variant='longshot'",
                    "model.cache_processor.linear_state_cache",
                )
            if config.family not in {"qwen3_5", "qwen3_5_moe"}:
                add(
                    "linear_state_family",
                    "cache_processor.linear_state_cache requires family='qwen3_5' or 'qwen3_5_moe'",
                    "model.family",
                )
            if processor.encoder_method != "query_attention":
                add(
                    "linear_state_encoder",
                    "cache_processor.linear_state_cache requires encoder_method='query_attention'",
                    "model.cache_processor.encoder_method",
                )
            if processor.encoder_query_source != "final_token":
                add(
                    "linear_state_query",
                    "cache_processor.linear_state_cache requires encoder_query_source='final_token'",
                    "model.cache_processor.encoder_query_source",
                )
            if processor.update_stream != "kv":
                add(
                    "linear_state_write",
                    "cache_processor.linear_state_cache requires update_stream='kv'",
                    "model.cache_processor.update_stream",
                )

    if config.variant != "tf" and config.family == "gemma4":
        try:
            config.ensure_uniform_stored_cache_shape(config.variant.capitalize())
        except ValueError as error:
            add("uniform_cache_shape", str(error), "model.layer_types")
    return _unique_issues(issues)


def _experiment_issues(config: TrainConfig) -> list[CompatibilityIssue]:
    issues: list[CompatibilityIssue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(CompatibilityIssue(code, message, path))

    if config.data.kind not in _SUPPORTED_DATA_KINDS:
        add(
            "unknown_data_kind",
            f"unsupported data kind: {config.data.kind}",
            "data.kind",
        )
    if config.trainable not in _SUPPORTED_TRAINABLE_POLICIES:
        add(
            "unknown_trainable_policy",
            f"unsupported trainable policy: {config.trainable}",
            "trainable",
        )
    if config.trainable == "lora" and config.model.lora is None:
        add(
            "lora_training_without_adapters",
            "trainable='lora' requires model.lora",
            "trainable",
        )
    if config.trainable == "cache_processor" and config.model.variant == "tf":
        add(
            "processor_training_without_processor",
            "trainable='cache_processor' requires a cache processor",
            "trainable",
        )
    if config.checkpoint_format not in _SUPPORTED_CHECKPOINT_FORMATS:
        add(
            "unknown_checkpoint_format",
            f"unsupported checkpoint format: {config.checkpoint_format}",
            "checkpoint_format",
        )
    if config.mode not in {"pretrain", "sft"}:
        add("unknown_training_mode", f"unknown training mode: {config.mode}", "mode")
    if config.distributed not in {"none", "ddp", "fsdp"}:
        add("unknown_execution", f"unknown distributed strategy: {config.distributed}", "distributed")
    if config.init_checkpoint is not None and config.backbone_checkpoint is not None:
        add(
            "backbone_with_init",
            "backbone_checkpoint and init_checkpoint are mutually exclusive",
            "init_checkpoint",
        )
    if config.init_checkpoint is not None and config.resume is not None:
        add(
            "init_with_resume",
            "init_checkpoint and resume are mutually exclusive",
            "resume",
        )
    if config.mode == "pretrain" and config.data.kind not in {"fineweb", "random"}:
        add("pretrain_data", "pretrain mode requires fineweb or random data", "data.kind")
    if config.mode == "sft" and config.data.kind != "sft":
        add("sft_data", "sft mode requires data.kind='sft'", "data.kind")
    if config.data.canonical_execution and config.resume is not None and config.resume.expected_manifest_sha256 is None:
        add(
            "canonical_resume_manifest",
            "canonical training resume requires expected manifest sha256",
            "resume.expected_manifest_sha256",
        )

    return _unique_issues(issues)


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


def _unique_issues(issues: list[CompatibilityIssue]) -> list[CompatibilityIssue]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[CompatibilityIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.path)
        if key not in seen:
            unique.append(issue)
            seen.add(key)
    return unique
