"""Generate the three probe-arm configs from the matrix11 longshot baseline.

Local staging only; rsync the result to the lrc experiment root.
"""

import hashlib
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments" / "gcw-dev2k"
SRC = CONFIG_DIR / "longshot-matrix11.yaml"
DEV2K = "qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k"
ARMS = {
    "longshot": None,
    "diffusion-longshot-t4-sigma0": 0.0,
    "diffusion-longshot-t4-sigma01": 0.1,
}

DIFFUSION_BLOCK = """  diffusion_processor:
    num_steps: 4
    time_conditioning: true
    eta_max: 1.0
    refinement_gate_init: -4.0
    noise_scale: {noise}
    conditioning_scale: 1.0
    inference_mode: deterministic"""


def main() -> None:
    src = SRC.read_text()
    for arm, noise in ARMS.items():
        s = src
        subs = [
            ("sft_loss_scope: answer_only", "sft_loss_scope: full_sequence"),
            ("max_steps: 10902", "max_steps: 2000"),
            ("warmup_steps: 327", "warmup_steps: 60"),
            (
                "experiments/qwen3-06b-instruct-math-full-single-seed-v2-matrix11/configs/experiment.json",
                f"experiments/{DEV2K}/configs/experiment.json",
            ),
            (
                "experiments/qwen3-06b-instruct-math-full-single-seed-v2-matrix11/training-state/longshot",
                f"experiments/{DEV2K}/training-state/{arm}",
            ),
            (
                "name: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-longshot",
                f"name: {DEV2K}-{arm}",
            ),
            (
                "group: qwen3-06b-instruct-math-full-single-seed-v2-matrix11",
                f"group: {DEV2K}",
            ),
            ("  - longshot", f"  - {arm}"),
            ("  arm: longshot", f"  arm: {arm}"),
            (
                "  id: qwen3-06b-instruct-math-full-single-seed-v2-matr-e67813c9296f",
                "  id: qwen3-06b-instruct-math-full-single-seed-v2-matr-"
                + hashlib.sha256(arm.encode()).hexdigest()[:12],
            ),
        ]
        for old, new in subs:
            assert s.count(old) == 1, (
                f"{arm}: {old[:48]!r} count={s.count(old)}"
            )
            s = s.replace(old, new, 1)
        if noise is not None:
            assert s.count("  diffusion_processor: null") == 1
            s = s.replace(
                "  diffusion_processor: null",
                DIFFUSION_BLOCK.format(noise=noise),
                1,
            )
        output = CONFIG_DIR / f"{arm}.yaml"
        output.write_text(s)
        print(f"wrote {output} (diffusion noise={noise})")


if __name__ == "__main__":
    main()
