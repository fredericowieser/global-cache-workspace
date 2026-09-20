from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

import yaml

from gcw.config import EvalConfig
from gcw.evaluation.korbench_scoring import extract_answer, is_correct

CATEGORIES = ("operation", "logic", "cipher", "puzzle", "counterfactual")
SOURCES = json.loads(
    Path(__file__).with_name("korbench_sources.json").read_text()
)
REVISION = SOURCES["revision"]
PROTOCOL = "gcw.korbench.zero_shot.v1"
DEFAULT_DATA = Path("data/eval-data/korbench")


def prepare_dataset(output: Path, source_dir: Path | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"KOR-Bench destination already exists: {output}")
    sources = {}
    for name, expected in SOURCES["files"].items():
        if source_dir is None:
            url = f"https://raw.githubusercontent.com/KOR-Bench/KOR-Bench/{REVISION}/{name}"
            with urlopen(url, timeout=60) as response:
                payload = response.read()
        else:
            payload = (Path(source_dir) / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"KOR-Bench source hash differs: {name}")
        sources[name] = payload.decode("utf-8")
    templates = yaml.safe_load(sources["config/prompt/zero-shot.yaml"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".korbench-", dir=output.parent
    ) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        files = {}
        for category in CATEGORIES:
            rules = [
                json.loads(line)
                for line in sources[f"data/{category}/rule.jsonl"].splitlines()
                if line.strip()
            ]
            samples = [
                json.loads(line)
                for line in sources[
                    f"data/{category}/sample.jsonl"
                ].splitlines()
                if line.strip()
            ]
            rule_map = {row["idx"]: row for row in rules}
            if len(rules) != 25 or set(rule_map) != {
                str(i) for i in range(1, 26)
            }:
                raise ValueError(
                    f"KOR-Bench rule population differs: {category}"
                )
            if len(samples) != 250 or {row["idx"] for row in samples} != {
                str(i) for i in range(1, 251)
            }:
                raise ValueError(
                    f"KOR-Bench question population differs: {category}"
                )
            if Counter(row["rule_id"] for row in samples) != Counter(
                {key: 10 for key in rule_map}
            ):
                raise ValueError(
                    f"KOR-Bench rule/question joins differ: {category}"
                )
            rows = []
            for sample in samples:
                rule = rule_map[sample["rule_id"]]
                rows.append(
                    {
                        "id": f"{category}:{sample['idx']}",
                        "idx": sample["idx"],
                        "category": category,
                        "source_category": sample.get("category", ""),
                        "rule_id": sample["rule_id"],
                        "rule_content": rule["rule_content"],
                        "question": sample["question"],
                        "prompt": templates[f"{category}_prompt_format"][
                            0
                        ].format(rule["rule_content"], sample["question"]),
                        "answer": sample["answer"],
                        "real_life_answer": sample.get("real_life_answer", ""),
                        "title": rule.get("title", ""),
                        "tag": rule.get("tag", ""),
                        "source_revision": REVISION,
                    }
                )
            payload = "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ).encode("utf-8")
            digest = hashlib.sha256(payload).hexdigest()
            if (
                SOURCES.get("prepared")
                and digest != SOURCES["prepared"][category]
            ):
                raise ValueError(f"KOR-Bench preparation changed: {category}")
            name = f"{category}.jsonl"
            (staging / name).write_bytes(payload)
            files[category] = {
                "file": name,
                "count": len(rows),
                "sha256": digest,
            }
        manifest = {
            "protocol": PROTOCOL,
            "source": "https://github.com/KOR-Bench/KOR-Bench",
            "revision": REVISION,
            "source_sha256": SOURCES["files"],
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        staging.rename(output)
    return manifest


def load_dataset(category: str, **metadata):
    from datasets import Dataset, DatasetDict

    if category not in CATEGORIES:
        raise ValueError(f"Unknown KOR-Bench category: {category}")
    if category == "operation":
        try:
            antlr_version = version("antlr4-python3-runtime")
        except PackageNotFoundError as exc:
            raise ImportError(
                "KOR-Bench requires the eval extra (antlr4-python3-runtime==4.11.1)"
            ) from exc
        if antlr_version != "4.11.1":
            raise ImportError(
                "KOR-Bench requires antlr4-python3-runtime==4.11.1"
            )
    directory = Path(os.environ.get("GCW_KORBENCH_DATA", str(DEFAULT_DATA)))
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Prepare KOR-Bench first: python scripts/prepare_korbench.py"
        ) from exc
    if (
        manifest.get("revision") != REVISION
        or manifest.get("protocol") != PROTOCOL
        or manifest.get("source_sha256") != SOURCES["files"]
    ):
        raise ValueError(
            "KOR-Bench manifest does not match the pinned protocol"
        )
    payload = (directory / f"{category}.jsonl").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = SOURCES["prepared"][category]
    receipt = manifest["files"][category]
    if digest != expected or receipt != {
        "file": f"{category}.jsonl",
        "count": 250,
        "sha256": expected,
    }:
        raise ValueError(f"Prepared KOR-Bench data hash differs: {category}")
    if (
        metadata.get("source_revision", REVISION) != REVISION
        or metadata.get("data_sha256", expected) != expected
    ):
        raise ValueError("KOR-Bench task metadata differs from pinned data")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    return DatasetDict({"test": Dataset.from_list(rows)})


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    response = results[0]
    category = doc["category"]
    extracted = extract_answer(
        response, "puzzle" if category == "puzzle" else "basic"
    )
    metrics = {
        "accuracy": float(
            is_correct(
                response, doc["answer"], category, doc["rule_id"], doc["idx"]
            )
        ),
        "parse_failure": float(
            extracted is None or not extracted.strip("[] \t\r\n")
        ),
    }
    if category == "counterfactual":
        metrics["real_life_answer_rate"] = float(
            is_correct(
                response,
                doc["real_life_answer"],
                category,
                doc["rule_id"],
                doc["idx"],
            )
        )
    return metrics


def annotate_results(results: dict) -> None:
    for task, samples in results.get("samples", {}).items():
        if not task.startswith("korbench_"):
            continue
        for sample in samples:
            doc = sample["doc"]
            sample["extracted_answer"] = extract_answer(
                sample["filtered_resps"][0],
                "puzzle" if doc["category"] == "puzzle" else "basic",
            )


def evaluation_protocol(config: EvalConfig) -> dict:
    task_root = Path("configs/tasks/korbench")
    return {
        "name": PROTOCOL,
        "source_revision": REVISION,
        "prepared_sha256": SOURCES["prepared"],
        "scorer_sha256": hashlib.sha256(
            Path(__file__).with_name("korbench_scoring.py").read_bytes()
        ).hexdigest(),
        "adapter_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "max_new_tokens": config.default_max_new_tokens,
        "mode": "zero-shot",
        "task_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [*task_root.glob("*.yaml"), task_root / "utils.py"]
            )
        },
        "scoring_dependencies": {
            package: version(package)
            for package in ("sympy", "antlr4-python3-runtime", "lm-eval")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned KOR-Bench zero-shot evaluation set"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Read a local upstream snapshot instead of downloading",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(args.out, args.source_dir), indent=2, sort_keys=True
        )
    )
