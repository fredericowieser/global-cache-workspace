from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Sequence

from gcw.config import load_train_config
from gcw.eval import ProcessBranchEvalCondition, run_task_groups
from scripts.evaluate_process_branch_records import (
    _atomic_write_json_new,
    authenticate_answer_only_artifact,
    authenticate_canonical_checkpoint,
    load_canonical_protocol,
    load_canonical_protocol_bytes,
)


FULL_GENERATION_TASKS = (
    "gsm8k_openmath",
    "svamp",
    "gsm8k_hard",
    "logiqa_opsd_generate",
    "agieval_gaokao_mathqa_opsd_generate",
)


ANSWER_ONLY_PROTOCOL_SCHEMA = "gcw.answer_only_eval.protocol.v1"
ANSWER_ONLY_RECEIPT_SCHEMA = "gcw.answer_only_eval.dry_run_receipt.v1"
ANSWER_ONLY_CONDITIONS = ["OFF", "INIT", "K1", "K2", "K4", "K8"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_answer_only_protocol(path: Path) -> dict:
    return load_canonical_protocol(path)


def finalize_answer_only_protocol(protocol_path: Path) -> dict:
    protocol = load_canonical_protocol_bytes(protocol_path.read_bytes())
    finalized = copy.deepcopy(protocol)
    for arm_name, arm in finalized["arms"].items():
        if arm["checkpoint_sha256"] is not None or arm["checkpoint_manifest_sha256"] is not None:
            raise ValueError(f"answer-only base protocol arm {arm_name} is already finalized")
        config_path = Path(arm["train_config"])
        config_bytes = config_path.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest() != arm["config_sha256"]:
            raise ValueError(f"answer-only arm {arm_name} train config sha256 mismatch")
        checkpoint = Path(arm["checkpoint"])
        manifest_path = checkpoint / "manifest.json"
        model_path = checkpoint / "model.safetensors"
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        authenticate_canonical_checkpoint(
            checkpoint,
            expected_manifest_sha256=manifest_sha256,
            expected_model_sha256=model_sha256,
        )
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
            raise ValueError(
                f"answer-only arm {arm_name} checkpoint manifest changed during finalization"
            )
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != model_sha256:
            raise ValueError(
                f"answer-only arm {arm_name} checkpoint model changed during finalization"
            )
        arm["checkpoint_manifest_sha256"] = manifest_sha256
        arm["checkpoint_sha256"] = model_sha256
    encoded = json.dumps(
        finalized,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    load_canonical_protocol_bytes(encoded)
    return finalized


def build_answer_only_dry_run_receipt(protocol_path: Path) -> dict:
    protocol_bytes = protocol_path.read_bytes()
    protocol = load_canonical_protocol_bytes(protocol_bytes)
    protocol_argument = os.path.relpath(protocol_path.resolve(), Path.cwd())
    commands = []
    dataset = protocol["dataset"]
    data_path = Path(dataset["path"])
    dataset_blockers: list[str] = []
    if not data_path.is_file():
        dataset_blockers.append("dataset_not_visible_on_current_node")
    else:
        try:
            data_bytes = data_path.read_bytes()
            authenticate_answer_only_artifact(
                data_bytes,
                expected_artifact_sha256=dataset["sha256"],
                expected_source_ids_sha256=dataset["ordered_source_ids_sha256"],
            )
        except (OSError, ValueError):
            dataset_blockers.append("dataset_authentication_failed")
    for arm_name, arm in protocol["arms"].items():
        if not isinstance(arm, dict):
            raise ValueError(f"answer-only arm {arm_name} must be an object")
        train_config = arm.get("train_config")
        checkpoint = arm.get("checkpoint")
        if not isinstance(train_config, str) or not train_config or not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"answer-only arm {arm_name} paths are invalid")
        arm_blockers = []
        train_config_path = Path(train_config)
        if not train_config_path.is_file():
            arm_blockers.append("train_config_not_visible")
        else:
            expected_config_sha256 = arm.get("config_sha256")
            if not isinstance(expected_config_sha256, str) or _sha256(train_config_path) != expected_config_sha256:
                arm_blockers.append("train_config_sha256_mismatch")
            else:
                try:
                    train = load_train_config(train_config_path)
                    process = train.process_branch
                    frozen_values = (
                        process.expected_train_eligible_source_ids_sha256,
                        process.expected_heldout_eligible_source_ids_sha256,
                        process.expected_initial_backbone_sha256,
                        process.expected_tokenizer_sha256,
                    ) if process is not None else ()
                    if (
                        process is None
                        or not process.require_frozen_identities
                        or any(value is None for value in frozen_values)
                    ):
                        arm_blockers.append("training_identities_not_frozen")
                except (OSError, TypeError, ValueError):
                    arm_blockers.append("train_config_validation_failed")
        checkpoint_path = Path(checkpoint)
        checkpoint_sha256 = arm.get("checkpoint_sha256")
        checkpoint_manifest_sha256 = arm.get("checkpoint_manifest_sha256")
        if not isinstance(checkpoint_sha256, str):
            arm_blockers.append("checkpoint_sha256_not_frozen")
        if not isinstance(checkpoint_manifest_sha256, str):
            arm_blockers.append("checkpoint_manifest_sha256_not_frozen")
        if not checkpoint_path.is_dir():
            arm_blockers.append("checkpoint_not_visible")
        elif isinstance(checkpoint_sha256, str) and isinstance(checkpoint_manifest_sha256, str):
            try:
                authenticate_canonical_checkpoint(
                    checkpoint_path,
                    expected_manifest_sha256=checkpoint_manifest_sha256,
                    expected_model_sha256=checkpoint_sha256,
                )
            except (OSError, ValueError):
                arm_blockers.append("checkpoint_authentication_failed")
        arm_blockers.extend(dataset_blockers)
        for split_name, split in protocol["splits"].items():
            blockers = ["dry_run_only", *arm_blockers]
            if not split["execution_allowed"]:
                blockers.append("protocol_forbids_execution")
            output_stem = f"results/answer_only_precursor/{arm_name}/{split_name}"
            argv = [
                ".venv/bin/python",
                "scripts/evaluate_process_branch_records.py",
                "canonical",
                "--protocol",
                protocol_argument,
                "--arm",
                arm_name,
                "--split",
                split_name,
                "--bundle-out",
                f"{output_stem}.bundle",
            ]
            commands.append(
                {
                    "arm": arm_name,
                    "split": split_name,
                    "skip_rows": split["skip_rows"],
                    "take_rows": split["take_rows"],
                    "execute": False,
                    "blockers": blockers,
                    "argv": argv,
                }
            )
    return {
        "schema": ANSWER_ONLY_RECEIPT_SCHEMA,
        "protocol": protocol_path.name,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "evaluator_version": protocol["evaluator_version"],
        "conditions": protocol["conditions"],
        "commands": commands,
        "final_rows_evaluated": False,
        "accelerator_launch_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    argument_list = list(argv) if argv is not None else sys.argv[1:]
    if "--finalize-answer-only-protocol" in argument_list:
        parser = argparse.ArgumentParser(
            description="Freeze authenticated checkpoint identities into an answer-only protocol."
        )
        parser.add_argument("--finalize-answer-only-protocol", required=True, type=Path)
        parser.add_argument("--execution-protocol", required=True, type=Path)
        args = parser.parse_args(argument_list)
        finalized = finalize_answer_only_protocol(args.finalize_answer_only_protocol)
        _atomic_write_json_new(args.execution_protocol, finalized)
        print(json.dumps(finalized, sort_keys=True))
        return 0
    if "--answer-only-protocol" in argument_list:
        parser = argparse.ArgumentParser(description="Publish an answer-only OFF/INIT/K evaluation dry-run receipt.")
        parser.add_argument("--answer-only-protocol", required=True, type=Path)
        parser.add_argument("--dry-run-receipt", required=True, type=Path)
        args = parser.parse_args(argument_list)
        receipt = build_answer_only_dry_run_receipt(args.answer_only_protocol)
        _atomic_write_json_new(args.dry_run_receipt, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0

    parser = argparse.ArgumentParser(description="Run the complete process-branch generation evaluation suite.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cycles", required=True, type=int)
    parser.add_argument("--prompt-suffix", required=True)
    parser.add_argument("--answer-prefix", default="\n\n")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argument_list)

    if args.out.exists():
        raise FileExistsError(args.out)
    if args.cycles < 0:
        raise ValueError("cycles must be non-negative")
    partial_dir = args.out.parent / f".{args.out.stem}.partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GCW_EVAL_PARTIAL_DIR"] = str(partial_dir)
    os.environ["GCW_EVAL_PROGRESS_PATH"] = str(partial_dir / "progress.json")

    condition = ProcessBranchEvalCondition(
        cycles=args.cycles,
        prompt_suffix=args.prompt_suffix,
        answer_prefix=args.answer_prefix,
    )
    results = run_task_groups(
        args.checkpoint,
        [([task], 0) for task in FULL_GENERATION_TASKS],
        limit=None,
        batch_size=1,
        process_branch_condition=condition,
    )
    payload = {
        "schema": "gcw.process_branch.full_generation_suite.v1",
        "checkpoint": str(args.checkpoint),
        "cycles": args.cycles,
        "prompt_suffix": args.prompt_suffix,
        "answer_prefix": args.answer_prefix,
        "tasks": list(FULL_GENERATION_TASKS),
        "limit": None,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "results"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
