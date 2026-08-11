from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).parents[1]
CANARY = ROOT / "scripts/cloud/run_answer_only_precursor_canary_h200.sh"
FULL = ROOT / "scripts/cloud/run_answer_only_precursor_full_h200.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(script: str, name: str) -> str:
    match = re.search(rf'^{name}="([^"]+)"$', script, flags=re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_answer_only_cloud_runners_have_valid_shell_syntax() -> None:
    subprocess.run(["bash", "-n", str(CANARY), str(FULL)], check=True)


def test_answer_only_canary_is_isolated_and_fail_closed() -> None:
    script = CANARY.read_text()
    config = ROOT / _assignment(script, "CANARY_CONFIG")

    assert _assignment(script, "CANARY_CONFIG_SHA256") == _sha256(config)
    manifest = ROOT / _assignment(script, "INPUT_MANIFEST")
    assert _assignment(script, "INPUT_MANIFEST_SHA256") == _sha256(manifest)
    assert 'for role, expected in manifest["files"].items()' in script
    assert "max_optimizer_steps: 1" in config.read_text()
    assert "qwen3_06b_answer_only_k1_6000_seed24.yaml" not in script
    assert "qwen3_06b_answer_only_random_depth_6000_seed24.yaml" not in script
    assert "evaluate_process_branch_records.py" not in script
    assert "sudo -n systemctl poweroff" in script
    assert "refusing canary relaunch" in script
    assert "gcw.answer_only_precursor.canary_success.v1" in script


def test_answer_only_full_runner_requires_canary_and_never_evaluates_final_split() -> None:
    script = FULL.read_text()

    assert ': "${CANARY_EVIDENCE:?CANARY_EVIDENCE is required}"' in script
    assert ': "${CANARY_EVIDENCE_SHA256:?CANARY_EVIDENCE_SHA256 is required}"' in script
    assert "--split final" not in script
    assert "final_rows_evaluated\": False" in script
    assert "sudo -n systemctl poweroff" in script
    assert script.index("wrong canary evidence kind") < script.index(
        'python scripts/train.py --config "$P0_CONFIG"'
    )
    assert script.index('python scripts/train.py --config "$P0_CONFIG"') < script.index(
        'python scripts/train.py --config "$P1_CONFIG"'
    )
    assert script.index('python scripts/train.py --config "$P1_CONFIG"') < script.index(
        "--finalize-answer-only-protocol"
    )
    assert script.index("--finalize-answer-only-protocol") < script.index(
        "--arm p0 --split heldout"
    )
    assert script.index("--arm p0 --split heldout") < script.index(
        "--arm p1 --split heldout"
    )


def test_answer_only_full_runner_embeds_current_reviewed_input_hashes() -> None:
    script = FULL.read_text()
    bindings = {
        "P0_CONFIG": "P0_CONFIG_SHA256",
        "P1_CONFIG": "P1_CONFIG_SHA256",
        "BASE_PROTOCOL": "BASE_PROTOCOL_SHA256",
        "DRY_RECEIPT": "DRY_RECEIPT_SHA256",
        "INPUT_MANIFEST": "INPUT_MANIFEST_SHA256",
    }
    for path_name, digest_name in bindings.items():
        path = ROOT / _assignment(script, path_name)
        assert _assignment(script, digest_name) == _sha256(path)
