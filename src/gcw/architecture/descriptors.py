from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from gcw.provenance import canonical_json_sha256
from gcw.architecture.contracts import CompatibilityReport

if TYPE_CHECKING:
    from gcw.config import TrainConfig


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
            "memory_domains": [
                domain.payload() for domain in self.memory_domains
            ],
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
            raise RuntimeError(
                "compatible model resolution has no assembly receipt"
            )
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


@dataclass(frozen=True)
class ExperimentResolution:
    experiment: ResolvedExperiment | None
    receipt: CompositionReceipt | None
    report: CompatibilityReport

    def require(self) -> tuple[ResolvedExperiment, CompositionReceipt]:
        self.report.require()
        if self.experiment is None or self.receipt is None:
            raise RuntimeError(
                "compatible experiment resolution has no composition receipt"
            )
        return self.experiment, self.receipt


def _receipt_metadata(
    receipt: ModelAssemblyReceipt | CompositionReceipt,
) -> dict[str, Any]:
    payload = receipt.payload()
    return {
        "payload": payload,
        "receipt_sha256": canonical_json_sha256(payload),
    }


def admission_receipt_metadata(
    config: TrainConfig,
    *,
    model_receipt: ModelAssemblyReceipt | None = None,
    composition_receipt: CompositionReceipt | None = None,
) -> dict[str, Any]:
    """Return authenticated model and experiment receipts for runtime artifacts.

    Accepts pre-resolved receipts so the training assembly path can compute them
    once and reuse them (avoiding redundant config serialization + hashing);
    when omitted they are resolved from ``config`` directly.
    """
    from gcw.architecture.resolution import resolve_experiment, resolve_model

    if model_receipt is None:
        _, model_receipt = resolve_model(config.model).require()
    if composition_receipt is None:
        _, composition_receipt = resolve_experiment(
            config, model_receipt
        ).require()
    return {
        "schema": "gcw.admission.v1",
        "model_assembly_receipt": _receipt_metadata(model_receipt),
        "composition_receipt": _receipt_metadata(composition_receipt),
    }
