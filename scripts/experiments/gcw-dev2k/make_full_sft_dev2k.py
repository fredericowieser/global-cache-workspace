"""Derive the dev2k full-SFT (no cache processor) baseline from the parent config."""

import hashlib
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs" / "experiments" / "gcw-dev2k"
SRC = CONFIG_DIR / "parent-full-sft.yaml"
OUT = CONFIG_DIR / "full-sft-dev2k.yaml"

subs = [
    ("sft_loss_scope: answer_only", "sft_loss_scope: full_sequence"),
    ("max_steps: 10902", "max_steps: 2000"),
    ("warmup_steps: 327", "warmup_steps: 60"),
    ("batch_size: 8", "batch_size: 2"),
    ("eval_batches: 128", "eval_batches: 512"),
    (
        "v2-matrix11/configs/experiment.json",
        "v2-matrix11-dev2k/configs/experiment.json",
    ),
    (
        "v2-matrix11/training-state/full-sft",
        "v2-matrix11-dev2k/training-state/full-sft",
    ),
    (
        "name: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-full-sft",
        "name: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k-full-sft",
    ),
    (
        "group: qwen3-06b-instruct-math-full-single-seed-v2-matrix11",
        "group: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k",
    ),
    (
        "id: qwen3-06b-instruct-math-full-single-seed-v2-matr-9ce5a02a7717",
        "id: qwen3-06b-instruct-math-full-single-seed-v2-matr-"
        + hashlib.sha256(b"full-sft-dev2k").hexdigest()[:12],
    ),
    (
        "experiment: qwen3-06b-instruct-math-full-single-seed-v2-matrix11",
        "experiment: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k",
    ),
    (
        "series: qwen3-06b-instruct-math-full-single-seed-v2-matrix11",
        "series: qwen3-06b-instruct-math-full-single-seed-v2-matrix11-dev2k",
    ),
]

s = SRC.read_text()
for old, new in subs:
    assert s.count(old) == 1, f"missing: {old[:60]!r} count={s.count(old)}"
    s = s.replace(old, new, 1)
OUT.write_text(s)
print(f"wrote {OUT}")
