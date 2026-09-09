"""Pre-flight smoke for the diffusion-longshot probe configs.

Builds the model from a probe config and runs a short generation through the
full backbone+processor path (exercises the diffusion recurrence in
longshot's block). Run on a camped GPU:
  srun --jobid=<id> --overlap -N1 -n1 --gres=gpu:1 \
      .venv/bin/python scripts/experiments/gcw-dev2k/smoke.py \
      configs/experiments/gcw-dev2k/<arm>.yaml
"""

import pathlib
import sys

import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from gcw.config import ModelConfig  # noqa: E402
from gcw.models.model import build_language_model  # noqa: E402


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    payload = yaml.safe_load(path.read_text())["model"]
    cfg = ModelConfig(**payload)
    noise = getattr(cfg.diffusion_processor, "noise_scale", None)
    print(
        f"variant={cfg.variant} diffusion={cfg.diffusion_processor is not None} "
        f"noise={noise} loss_scope=",
        flush=True,
    )
    model = build_language_model(cfg)
    model.to("cuda:0").eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 128), device="cuda:0")
    out = model.generate_fast(
        ids,
        8,
        eos_token_id=151645,
        stop_sequences=[[151645]],
        pad_token_id=151645,
    )
    print(f"generate_fast OK: {tuple(out.shape)}", flush=True)
    print("SMOKE PASS", flush=True)


if __name__ == "__main__":
    main()
