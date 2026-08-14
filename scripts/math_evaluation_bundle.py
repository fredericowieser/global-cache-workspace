from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from typing import Any


REQUIRED_FILES = ("config.json", "model.safetensors", "train_state.json")
SCHEMA = "gcw.evaluation_checkpoint_bundle.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_commit(repository: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()


def _file_identity(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _validate_source_checkpoint(checkpoint: Path, expected_step: int) -> dict[str, Any]:
    for name in REQUIRED_FILES:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(checkpoint / name)
    state = json.loads((checkpoint / "train_state.json").read_text())
    if int(state.get("step", -1)) != expected_step:
        raise ValueError(f"checkpoint step {state.get('step')} does not equal {expected_step}")
    original_manifest = checkpoint / "manifest.json"
    if original_manifest.is_file():
        payload = json.loads(original_manifest.read_text())
        if not payload.get("complete") or int(payload.get("step", -1)) != expected_step:
            raise ValueError("source checkpoint manifest is incomplete or has the wrong step")
        declared = payload.get("files", {})
        for name in REQUIRED_FILES:
            if declared.get(name) != _sha256(checkpoint / name):
                raise ValueError(f"source checkpoint manifest hash mismatch for {name}")
    return state


def create_bundle(
    checkpoint: Path,
    destination: Path,
    *,
    arm: str,
    expected_step: int,
    repository: Path,
    training_config: Path | None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    destination = destination.resolve()
    state = _validate_source_checkpoint(checkpoint, expected_step)
    if destination.exists():
        return verify_bundle(destination, expected_arm=arm, expected_step=expected_step)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for name in REQUIRED_FILES:
            shutil.copyfile(checkpoint / name, temporary / name)
        files = {name: _file_identity(temporary / name) for name in REQUIRED_FILES}
        original_manifest = checkpoint / "manifest.json"
        payload: dict[str, Any] = {
            "schema": SCHEMA,
            "arm": arm,
            "step": expected_step,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by_host": socket.gethostname(),
            "source_commit": _source_commit(repository),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_manifest_sha256": (
                _sha256(original_manifest) if original_manifest.is_file() else None
            ),
            "training_config_sha256": (
                _sha256(training_config.resolve()) if training_config is not None else None
            ),
            "files": files,
            "checkpoint_state": {
                "dtype": state.get("dtype"),
                "processor_enabled": state.get("processor_enabled"),
            },
        }
        encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        (temporary / "bundle-manifest.json").write_bytes(encoded)
        ready = {
            "schema": "gcw.evaluation_checkpoint_ready.v1",
            "arm": arm,
            "step": expected_step,
            "bundle_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        (temporary / "READY.json").write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return verify_bundle(destination, expected_arm=arm, expected_step=expected_step)


def verify_bundle(
    bundle: Path,
    *,
    expected_arm: str | None = None,
    expected_step: int | None = None,
) -> dict[str, Any]:
    bundle = bundle.resolve()
    manifest_path = bundle / "bundle-manifest.json"
    ready_path = bundle / "READY.json"
    if not manifest_path.is_file() or not ready_path.is_file():
        raise FileNotFoundError(f"bundle is not ready: {bundle}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    ready = json.loads(ready_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("unknown evaluation bundle schema")
    if ready.get("bundle_manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
        raise ValueError("READY manifest digest mismatch")
    if expected_arm is not None and manifest.get("arm") != expected_arm:
        raise ValueError(f"expected arm {expected_arm}, got {manifest.get('arm')}")
    if expected_step is not None and int(manifest.get("step", -1)) != expected_step:
        raise ValueError(f"expected step {expected_step}, got {manifest.get('step')}")
    if set(manifest.get("files", {})) != set(REQUIRED_FILES):
        raise ValueError("evaluation bundle has the wrong file set")
    for name, expected in manifest["files"].items():
        path = bundle / name
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"evaluation bundle size mismatch for {name}")
        if _sha256(path) != expected["sha256"]:
            raise ValueError(f"evaluation bundle hash mismatch for {name}")
    state = json.loads((bundle / "train_state.json").read_text())
    if int(state.get("step", -1)) != int(manifest["step"]):
        raise ValueError("evaluation bundle train state has the wrong step")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create or verify an authenticated evaluation checkpoint bundle")
    subparsers = parser.add_subparsers(dest="action", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--checkpoint", required=True)
    create.add_argument("--destination", required=True)
    create.add_argument("--arm", required=True)
    create.add_argument("--expected-step", type=int, required=True)
    create.add_argument("--training-config")
    create.add_argument("--execute", action="store_true")
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--expected-arm")
    verify.add_argument("--expected-step", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repository = Path(__file__).resolve().parents[1]
    if args.action == "create":
        plan = {
            "action": "create",
            "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
            "destination": str(Path(args.destination).expanduser().resolve()),
            "arm": args.arm,
            "expected_step": args.expected_step,
            "source_commit": _source_commit(repository),
        }
        if not args.execute:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return
        result = create_bundle(
            Path(args.checkpoint).expanduser(),
            Path(args.destination).expanduser(),
            arm=args.arm,
            expected_step=args.expected_step,
            repository=repository,
            training_config=(Path(args.training_config).expanduser() if args.training_config else None),
        )
    else:
        result = verify_bundle(
            Path(args.bundle).expanduser(),
            expected_arm=args.expected_arm,
            expected_step=args.expected_step,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
