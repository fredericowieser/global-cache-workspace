from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from gcw.bottleneck_specs import BOTTLENECK_SPECS, bottleneck_spec
from gcw.checkpoint import load_checkpoint
from gcw.convert import convert_bottleneck, convert_hf

PUBLIC_ROOT = Path("/mnt/jfs-01/shared/gcw")

LEGACY_FINAL_MODELS: dict[str, dict[str, str]] = {
    "llama-3.2-1b": {
        "bt": "/home/f50056118/bottleneck_llm/paper_runs/bt/llama-3.2-1b/seed0/train/OpenMathInstruct-2/bt/Llama3.2-1B/version-4/final_model",
        "bgt": "/home/f50056118/bottleneck_llm/paper_runs/bgt/llama-3.2-1b/seed0/train/OpenMathInstruct-2/bgt/Llama3.2-1B/version-2/final_model",
    },
    "llama-3.2-3b": {
        "bt": "/home/f50056118/bottleneck_llm/paper_runs/bt/llama-3.2-3b/seed0/train/OpenMathInstruct-2/bt/Llama3.2-3B/version-1/final_model",
        "bgt": "/home/f50056118/bottleneck_llm/paper_runs/bgt/llama-3.2-3b/seed0/train/OpenMathInstruct-2/bgt/Llama3.2-3B/version-1/final_model",
    },
    "llama-3.1-8b": {
        "bt": "/home/f50056118/bottleneck_llm/paper_runs/bt/llama-3.1-8b/seed0/train/OpenMathInstruct-2/bt/Llama3.1-8B/version-1/final_model",
        "bgt": "/home/f50056118/bottleneck_llm/paper_runs/bgt/llama-3.1-8b/seed0/train/OpenMathInstruct-2/bgt/Llama3.1-8B/version-1/final_model",
    },
    "qwen3-0.6b": {
        "bt": "/home/f50056118/bottleneck_llm/paper_runs/bt/qwen3-0.6b/seed0/train/OpenMathInstruct-2/bt/Qwen3-0.6B/version-2/final_model",
        "bgt": "/home/f50056118/bottleneck_llm/paper_runs/bgt/qwen3-0.6b/seed0/train/OpenMathInstruct-2/bgt/Qwen3-0.6B/version-4/final_model",
    },
}

VERIFIED_LLAMA32_1B_BGT_RESULTS = {
    "json": "results/llama32_1b_bgt_paper7_bosfix_llfix.json",
    "md": "results/llama32_1b_bgt_paper7_bosfix_llfix.md",
    "csv": "results/llama32_1b_bgt_paper7_bosfix_llfix.csv",
}


@dataclass(frozen=True)
class ArtifactRecord:
    kind: Literal["legacy_final", "sft_backbone"]
    backbone: str
    variant: str
    display_name: str
    source_path: str
    tokenizer_source: str
    output_path: str
    model_sha256: str
    bytes: int
    dtype: str
    sequence_len: int
    validation: str


def legacy_artifacts(
    root: Path = PUBLIC_ROOT,
) -> list[tuple[str, str, Path, Path]]:
    artifacts = []
    for backbone, variants in LEGACY_FINAL_MODELS.items():
        spec = bottleneck_spec(backbone)
        for variant, source in variants.items():
            out = (
                root
                / "checkpoints"
                / "bottleneck_llm"
                / spec.artifact_slug
                / variant
                / "final"
            )
            artifacts.append((backbone, variant, Path(source), out))
    return artifacts


def sft_artifacts(root: Path = PUBLIC_ROOT) -> list[tuple[str, Path, Path]]:
    artifacts = []
    for backbone, spec in BOTTLENECK_SPECS.items():
        out = root / "checkpoints" / "sft_backbones" / spec.artifact_slug / "tf"
        artifacts.append((backbone, Path(spec.sft_path), out))
    return artifacts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Publish GCW-native Bottleneck LLM artifacts to shared storage."
    )
    parser.add_argument("--root", type=Path, default=PUBLIC_ROOT)
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-results", action="store_true")
    args = parser.parse_args(argv)

    records = publish(
        root=args.root,
        dtype=args.dtype,
        force=args.force,
        copy_results=not args.skip_results,
    )
    write_manifest(args.root, records)
    make_world_readable(args.root)
    print(f"Published {len(records)} artifacts under {args.root}")


def publish(
    root: Path = PUBLIC_ROOT,
    dtype: str = "auto",
    force: bool = False,
    copy_results: bool = True,
) -> list[ArtifactRecord]:
    records = []
    for backbone, source, out in sft_artifacts(root):
        spec = bottleneck_spec(backbone)
        _ensure_source(source)
        if force or not (out / "model.safetensors").exists():
            _reset_output(out)
            convert_hf(
                str(source),
                out,
                family=spec.family,
                variant="tf",
                dtype=_hf_dtype(dtype),
            )
        records.append(
            _record("sft_backbone", backbone, "tf", source, spec.sft_path, out)
        )

    for backbone, variant, source, out in legacy_artifacts(root):
        spec = bottleneck_spec(backbone)
        _ensure_source(source)
        if force or not (out / "model.safetensors").exists():
            _reset_output(out)
            convert_bottleneck(
                source,
                out,
                variant=variant,
                dtype=dtype,
                source=spec.source,
                tokenizer_source=str(out),
                backbone=backbone,
                sequence_len=8192,
                legacy_generate=True,
            )
        _copy_tokenizer_files(source, out)
        _set_checkpoint_source(out, str(out))
        records.append(
            _record(
                "legacy_final", backbone, variant, source, spec.sft_path, out
            )
        )

    if copy_results:
        copy_verified_results(root)
    return records


def write_manifest(root: Path, records: list[ArtifactRecord]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "root": str(root),
        "artifact_count": len(records),
        "legacy_final_count": sum(
            record.kind == "legacy_final" for record in records
        ),
        "sft_backbone_count": sum(
            record.kind == "sft_backbone" for record in records
        ),
        "artifacts": [asdict(record) for record in records],
    }
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (root / "MANIFEST.md").write_text(
        _manifest_markdown(records), encoding="utf-8"
    )


def copy_verified_results(root: Path) -> None:
    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for src in VERIFIED_LLAMA32_1B_BGT_RESULTS.values():
        path = Path(src)
        if path.exists():
            shutil.copy2(path, results_dir / path.name)


def make_world_readable(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
        for name in dirnames:
            os.chmod(Path(dirpath) / name, 0o755)
        for name in filenames:
            os.chmod(Path(dirpath) / name, 0o644)


def _record(
    kind: Literal["legacy_final", "sft_backbone"],
    backbone: str,
    variant: str,
    source: Path,
    tokenizer_source: str,
    out: Path,
) -> ArtifactRecord:
    model, cfg, state = load_checkpoint(out)
    del model
    spec = bottleneck_spec(backbone)
    return ArtifactRecord(
        kind=kind,
        backbone=backbone,
        variant=variant,
        display_name=spec.display_name,
        source_path=str(source),
        tokenizer_source=(
            str(out) if kind == "legacy_final" else tokenizer_source
        ),
        output_path=str(out),
        model_sha256=_sha256(out / "model.safetensors"),
        bytes=(out / "model.safetensors").stat().st_size,
        dtype=str(state.get("dtype", "unknown")),
        sequence_len=cfg.sequence_len,
        validation="strict_load_ok",
    )


def _manifest_markdown(records: list[ArtifactRecord]) -> str:
    lines = [
        "# GCW Shared Checkpoint Manifest",
        "",
        "| kind | backbone | variant | path | tokenizer | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.kind} | {record.display_name} | {record.variant} | `{record.output_path}` | "
            f"`{record.tokenizer_source}` | `{record.model_sha256[:12]}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _ensure_source(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _copy_tokenizer_files(source: Path, out: Path) -> None:
    source_dir = source if source.is_dir() else source.parent
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, out / name)


def _set_checkpoint_source(checkpoint: Path, source: str) -> None:
    path = checkpoint / "train_state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    state["source"] = source
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _reset_output(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _hf_dtype(dtype: str) -> str:
    return "bfloat16" if dtype == "auto" else dtype


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
