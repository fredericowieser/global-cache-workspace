from __future__ import annotations

import argparse
import json
import time

import torch

from gcw.config import CacheProcessorConfig, FlowProcessorConfig, ModelConfig
from gcw.models import LatentFlow, build_model


def smoke_config(variant: str, flow_steps: int) -> ModelConfig:
    return ModelConfig(
        variant=variant,  # type: ignore[arg-type]
        sequence_len=16,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        rotary_style="llama",
        cache_processor=CacheProcessorConfig(
            process_every=8,
            topk=4,
            topk_heads=2,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            encoder_seq_len=8,
        ),
        flow_processor=FlowProcessorConfig(num_steps=flow_steps),
    )


def run_variant(
    variant: str, device: torch.device, flow_steps: int, bf16: bool
) -> dict[str, object]:
    torch.manual_seed(1234)
    config = smoke_config(variant, flow_steps)
    model = build_model(config).to(device)
    model.init_weights()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(
        0, config.vocab_size, (2, config.sequence_len), device=device
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=bf16
    ):
        loss = model(input_ids, input_ids)
    loss.backward()
    flow = next(
        module for module in model.modules() if isinstance(module, LatentFlow)
    )
    if (
        flow.flow_gate.grad is None
        or not torch.isfinite(flow.flow_gate.grad).all()
    ):
        raise RuntimeError(f"{variant} produced an invalid flow-gate gradient")
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    return {
        "variant": variant,
        "architecture": config.architecture_name,
        "device": str(device),
        "flow_steps": flow_steps,
        "loss": float(loss.detach().cpu()),
        "flow_gate_grad": float(flow.flow_gate.grad.detach().float().cpu()),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 1024 / 1024
            if device.type == "cuda"
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one optimisation step for every flow-enabled BGT-family model."
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--flow-steps", type=int, default=2)
    parser.add_argument("--no-bf16", action="store_true")
    args = parser.parse_args()
    if args.flow_steps <= 0:
        raise ValueError("--flow-steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": str(device),
            }
        ),
        flush=True,
    )
    for variant in ("bgt", "icarus", "longshot"):
        result = run_variant(
            variant, device, args.flow_steps, bf16=not args.no_bf16
        )
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
