"""Small CPU smoke test for the installed/source-layout GCW API."""

from __future__ import annotations

import torch

from gcw.architecture import CompatibilityError, resolve_model
from gcw.config import CacheProcessorConfig, FlowProcessorConfig, ModelConfig
from gcw.models import build_model


def _config(variant: str, *, flow: bool = False, iterations: int = 1) -> ModelConfig:
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
            iterations=iterations,
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
        flow_processor=FlowProcessorConfig(num_steps=2) if flow else None,
    )


def _loss(config: ModelConfig, input_ids: torch.Tensor) -> float:
    model = build_model(config)
    model.init_weights()
    loss = model(input_ids, input_ids)
    return float(loss.detach())


def main() -> None:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (2, 8))
    print(f"TF loss={_loss(_config('tf'), input_ids):.6f}")
    print(f"BGT loss={_loss(_config('bgt'), input_ids):.6f}")

    admitted, _ = resolve_model(_config("longshot")).require()
    print(f"Longshot admitted: {admitted.variant}")
    try:
        resolve_model(_config("longshot", flow=True, iterations=2)).require()
    except CompatibilityError as error:
        print(f"Longshot rejected: {error}")


if __name__ == "__main__":
    main()
