"""Audit saved KOR-Bench responses and calculate paired correctness overlap."""

import csv
import hashlib
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PAPER = ROOT / "src/Global-Cache-Workspace-Theory-ICLR-2027-/appendices"
LOCAL = ROOT / "slurm_out/korbench-v3-20260920/results"
REMOTE = Path("/mnt/jfs-01/shared/gcw/martin_nb/results")
CATEGORIES = ("operation", "logic", "cipher", "puzzle", "counterfactual")
MODELS = {
    "Llama 3.2-1B": "llama32_1b",
    "Llama 3.2-3B": "llama32_3b",
    "Qwen3.5-2B": "qwen35_2b",
    "Qwen3.5-4B": "qwen35_4b",
    "Qwen3.5-27B": "qwen35_27b",
}
METHODS = {
    "Pretrained instruct": "backbone",
    "SFT": "sft",
    r"\methodSFT{}": "sft",
    "LoRA": "lora",
    r"\methodBT{}": "bt",
    r"\methodBGT{}": "bgt",
    r"\methodLongshot{}": "longshot",
    r"\methodDiffusionLongshot{}": "diffusion_longshot",
}


def source_path(regime, model, method, mode):
    if method == "backbone":
        regime = "math"
    if regime != "math":
        if model == "qwen35_4b":
            folder = REMOTE / (
                "reasoning-gym-4b-maths-init-20260922"
                if regime == "sequential"
                else "reasoning-gym-4b-train-eval-20260922T231023Z"
            )
            pattern = (
                f"{model}_{method}_seed24_step2000_korbench_thinking-off.json"
            )
        else:
            folder = (
                ROOT
                / "slurm_out"
                / (
                    "reasoning-gym-128k-math-init"
                    if regime == "sequential"
                    else "reasoning-gym-bf16-train-eval-20260922T212943Z"
                )
                / "results"
            )
            pattern = f"{model}_{method}_seed24_korbench_thinking-off.json"
    elif model == "qwen35_27b":
        if method in ("bt", "bgt"):
            folder = REMOTE / "qwen27b-bt-bgt-korbench-20260921T111531Z"
        elif mode == "off":
            return (
                REMOTE
                / "qwen27b-korbench-recovery-20260920/completed"
                / f"{method}_korbench_thinking-off.completed.json"
            )
        elif method in ("longshot", "diffusion_longshot"):
            folder = REMOTE / "qwen27b-korbench-thinking-on-v3-20260921T000735Z"
        else:
            folder = (
                ROOT / "data/analysis/local/korbench-overlap-20260924/sources"
            )
        pattern = f"{method}_korbench_thinking-{mode}.json"
    else:
        folder = LOCAL
        token = (
            "pretrained"
            if method == "backbone"
            else ("diffusion" if method == "diffusion_longshot" else method)
        )
        pattern = f"{model}*{token}*thinking-{mode}.json"
    matches = list(folder.glob(pattern))
    assert len(matches) == 1, (folder, pattern, matches)
    return matches[0]


def paper_rows():
    rows = []
    for filename in ("D_Extended_Results.tex", "E_Diffusion_Archive.tex"):
        for block in re.findall(
            r"\\begin\{table\}(.*?)\\end\{table\}",
            (PAPER / filename).read_text(),
            re.S,
        ):
            label = re.search(r"\\label\{([^}]+)\}", block)[1]
            if "korbench" not in label:
                continue
            regime = (
                "sequential"
                if "sequential" in label
                else ("reasoning" if "reasoning-gym" in label else "math")
            )
            for line in block.splitlines():
                if line.startswith(r"\multicolumn") and r"\textit{" in line:
                    model = MODELS[re.search(r"\\textit\{([^}]+)\}", line)[1]]
                if not any(line.startswith(m + " &") for m in METHODS):
                    continue
                cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
                for i, mode in enumerate(
                    ("off", "on") if regime == "math" else ("off",)
                ):
                    accuracy = [c.split("/")[i] for c in cells[1:7]]
                    if accuracy[-1].startswith("\\"):
                        continue
                    rows.append(
                        dict(
                            appendix=filename,
                            table=label,
                            regime=regime,
                            model=model,
                            method=METHODS[cells[0]],
                            mode=mode,
                            accuracy=[float(v) for v in accuracy],
                        )
                    )
    return rows


def load_run(path):
    raw = path.read_bytes()
    payload = json.loads(raw)
    result = payload["results"] if "campaign_manifest" in payload else payload
    protocol = result["gcw_runtime"]["korbench"]["benchmark"]
    rescore = protocol["name"] != "gcw.korbench.zero_shot.v3"
    from gcw.evaluation.korbench import (
        process_results,
        scoring_response,
        extract_answer,
        scoring_protocol,
    )

    effective_protocol = scoring_protocol()
    assert protocol["scorer_sha256"] == effective_protocol["scorer_sha256"]
    if not rescore:
        for field in ("response_preprocessing", "answer_selection"):
            assert protocol[field] == effective_protocol[field]
    samples = {}
    changes = 0
    for task, records in result["samples"].items():
        assert len(records) == 250, (path, task, len(records))
        for record in records:
            doc = record["doc"]
            key = (task, doc["id"])
            assert key not in samples
            assert (
                record["filtered_resps"] and len(record["filtered_resps"]) == 1
            )
            scores = (
                process_results(doc, record["filtered_resps"])
                if rescore
                else record
            )
            if rescore:
                scores["extracted_answer"] = extract_answer(
                    scoring_response(record["filtered_resps"][0]),
                    "puzzle" if doc["category"] == "puzzle" else "basic",
                )
            changes += scores["accuracy"] != record["accuracy"]
            assert scores["accuracy"] in (0, 1)
            prompt = record["arguments"][0][0]
            normalized_prompt = re.sub(
                r"Today Date: [^\n]+", "Today Date: <DATE>", prompt
            )
            samples[key] = dict(
                normalized_prompt_hash=hashlib.sha256(
                    normalized_prompt.encode()
                ).hexdigest(),
                correct=int(scores["accuracy"]),
                parse_failure=int(scores["parse_failure"]),
                doc_hash=hashlib.sha256(
                    json.dumps(doc, sort_keys=True).encode()
                ).hexdigest(),
                prompt_hash=record["prompt_hash"],
                response_hash=hashlib.sha256(
                    record["filtered_resps"][0].encode()
                ).hexdigest(),
                extracted_answer=scores["extracted_answer"],
                rule_id=doc["rule_id"],
            )
    assert len(samples) == 1250
    provenance = dict(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        rescored=rescore,
        changed_scores=changes,
        protocol={k: v for k, v in protocol.items() if k != "task_configs"},
        generation_protocol=result["gcw_protocol"],
        effective_scoring_protocol=effective_protocol,
    )
    return samples, provenance


def contingency(pairs):
    counts = Counter((b["correct"], m["correct"]) for b, m in pairs)
    cc, cw, wc, ww = (counts[k] for k in ((1, 1), (1, 0), (0, 1), (0, 0)))
    n = cc + cw + wc + ww
    return dict(
        n=n,
        both_correct=cc,
        lost=cw,
        gained=wc,
        both_wrong=ww,
        backbone_correct=cc + cw,
        backbone_wrong=wc + ww,
        retained_pct=100 * cc / (cc + cw) if cc + cw else None,
        recovered_pct=100 * wc / (wc + ww) if wc + ww else None,
        changed_correctness_pct=100 * (cw + wc) / n,
        accuracy_change_pp=100 * (wc - cw) / n,
        method_accuracy_pct=100 * (cc + wc) / n,
        backbone_accuracy_pct=100 * (cc + cw) / n,
    )


def main():
    table_rows = paper_rows()
    runs = {}
    sources = {}
    for row in table_rows:
        path = source_path(
            row["regime"], row["model"], row["method"], row["mode"]
        )
        if str(path) not in runs:
            print("Loading", path.name, flush=True)
            runs[str(path)], sources[str(path)] = load_run(path)
        samples = runs[str(path)]
        computed = [
            100
            * sum(
                s["correct"]
                for (task, _), s in samples.items()
                if task == "korbench_" + category
            )
            / 250
            for category in CATEGORIES
        ]
        computed.append(
            100 * sum(s["correct"] for s in samples.values()) / 1250
        )
        assert all(
            abs(x - y) < 0.005 for x, y in zip(computed, row["accuracy"])
        ), (row, computed)
        row["source"] = str(path)
    output, category_rows, paired_rows = [], [], []
    seen = set()
    for row in table_rows:
        identity = tuple(row[k] for k in ("regime", "model", "method", "mode"))
        if identity in seen:
            continue
        seen.add(identity)
        baseline = str(
            source_path(row["regime"], row["model"], "backbone", row["mode"])
        )
        base, method = runs[baseline], runs[row["source"]]
        assert base.keys() == method.keys()
        bp, mp = sources[baseline], sources[row["source"]]
        for k in ("enable_thinking", "apply_chat_template"):
            assert bp["generation_protocol"][k] == mp["generation_protocol"][k]
        for k in (
            "max_new_tokens",
            "source_revision",
            "prepared_sha256",
            "scorer_sha256",
        ):
            assert bp["protocol"][k] == mp["protocol"][k], (identity, k)
        for key in base:
            assert base[key]["doc_hash"] == method[key]["doc_hash"], (
                identity,
                key,
            )
            assert (
                base[key]["normalized_prompt_hash"]
                == method[key]["normalized_prompt_hash"]
            ), (identity, key, "prompt difference beyond date")
            paired_rows.append(
                dict(
                    zip(("regime", "model", "method", "mode"), identity),
                    task=key[0],
                    source_id=key[1],
                    backbone_correct=base[key]["correct"],
                    method_correct=method[key]["correct"],
                )
            )
        summary = {k: row[k] for k in ("regime", "model", "method", "mode")}
        summary.update(contingency([(base[k], method[k]) for k in base]))
        summary.update(
            prompt_mismatches=sum(
                base[k]["prompt_hash"] != method[k]["prompt_hash"] for k in base
            ),
            identical_response_pct=100
            * sum(
                base[k]["response_hash"] == method[k]["response_hash"]
                for k in base
            )
            / 1250,
            identical_extracted_answer_pct=100
            * sum(
                base[k]["extracted_answer"] == method[k]["extracted_answer"]
                and not base[k]["parse_failure"]
                and not method[k]["parse_failure"]
                for k in base
            )
            / 1250,
            lost_parse_failures=sum(
                base[k]["correct"]
                and not method[k]["correct"]
                and method[k]["parse_failure"]
                for k in base
            ),
        )
        output.append(summary)
        for category in CATEGORIES:
            pairs = [
                (base[k], method[k])
                for k in base
                if k[0] == "korbench_" + category
            ]
            category_rows.append(
                {k: row[k] for k in ("regime", "model", "method", "mode")}
                | dict(category=category)
                | contingency(pairs)
            )
    for name, rows in (
        ("summary.csv", output),
        ("categories.csv", category_rows),
        ("paired_correctness.csv.gz", paired_rows),
    ):
        stream = (
            io.TextIOWrapper(
                gzip.GzipFile(filename=str(OUT / name), mode="wb", mtime=0)
            )
            if name.endswith(".gz")
            else (OUT / name).open("w")
        )
        with stream as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (OUT / "provenance.json").write_text(
        json.dumps(dict(sources=sources, table_rows=table_rows), indent=2)
        + "\n"
    )
    print(
        f"Validated {len(table_rows)} table entries from {len(runs)} unique evaluations; {len(output)} unique comparisons."
    )


if __name__ == "__main__":
    main()
