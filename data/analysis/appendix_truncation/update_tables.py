"""Add pooled generation-cap percentages from the audited appendix sources."""

import json
import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[2] / "src/Global-Cache-Workspace-Theory-ICLR-2027-/appendices"
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
evidence = json.loads((OUT / "provenance.json").read_text(encoding="utf-8"))
values = {}
for row in evidence["table_rows"]:
    metrics = [
        evidence["sources"][task["source"]]["runtime"][task["task"]]
        for task in row["tasks"]
    ]
    for field in (
        "generated_responses", "response_tokens_total", "generation_cap_hits"
    ):
        assert sum(task[field] for task in metrics) == row[field]
    rate = 100 * row["generation_cap_hits"] / row["generated_responses"]
    assert round(rate, 2) == row["truncation_pct"]
    key = tuple(row[k] for k in ("table", "model", "method", "mode"))
    assert key not in values
    values[key] = row

seen = set()
updates = {}
for filename in ("C_Extended_Results.tex", "E_Diffusion_Archive.tex"):
    path = PAPER / filename
    text = path.read_text(encoding="utf-8")
    for match in list(re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S))[::-1]:
        block = match[0]
        label = re.search(r"\\label\{([^}]+)\}", block)[1]
        if not any(key[0] == label for key in values):
            continue
        existing = r"Trunc. (\%)" in block
        lines = []
        for line in block.splitlines():
            model_match = re.search(r"\\text(?:bf|it)\{([^}]+)\}", line)
            if line.startswith(r"\multicolumn") and model_match:
                model = MODELS[model_match[1]]
            if line.startswith("Method &"):
                columns = [c.strip() for c in line[:-2].split("&")]
            if any(line.startswith(method + " &") for method in METHODS):
                cells = [c.strip() for c in line[:-2].split("&")]
                entries = []
                for i, mean in enumerate(cells[columns.index("Mean/resp.")].split("/")):
                    key = (label, model, METHODS[cells[0]], ("off", "on")[i])
                    if mean.startswith("\\"):
                        assert key not in values
                        entries.append(r"\textemdash")
                        continue
                    row = values[key]
                    seen.add(key)
                    n = row["generated_responses"]
                    tokens = row["response_tokens_total"]
                    assert f"{tokens / n:.1f}" == mean, key
                    total = cells[columns.index("Total (M)")].split("/")[i]
                    assert f"{tokens / 1e6:.3f}" == total, key
                    entries.append(f"{row['truncation_pct']:.2f}")
                value = "/".join(entries)
                if existing:
                    assert cells[-1] == value, (key, cells[-1], value)
                else:
                    line = line.rstrip()[:-2].rstrip() + " & " + value + r" \\"
            lines.append(line)
        if existing:
            continue
        block = "\n".join(lines)
        old_count = len(columns)
        block = re.sub(r"(\\begin\{tabular\}\{[^\n]+)(\})", r"\1c\2", block)
        block = block.replace(rf"\multicolumn{{{old_count}}}", rf"\multicolumn{{{old_count + 1}}}")
        block = block.replace(r"\multicolumn{2}{c}{Generation tokens}", r"\multicolumn{3}{c}{Generation}")
        block = block.replace(rf"\cmidrule(l){{{old_count - 1}-{old_count}}}", rf"\cmidrule(l){{{old_count - 1}-{old_count + 1}}}")
        block = block.replace(r"Total (M) \\", r"Total (M) & Trunc. (\%) \\")
        caption_end = block.index("\n\\label")
        assert block[caption_end - 1] == "}"
        definition = " Trunc. reports the percentage of responses stopped by the generation cap, pooled over the same tasks as token statistics."
        block = block[:caption_end - 1] + definition + block[caption_end - 1:]
        text = text[:match.start()] + block + text[match.end():]
    updates[path] = text
assert seen == values.keys(), values.keys() - seen
for path, text in updates.items():
    if text != path.read_text(encoding="utf-8"):
        path.write_text(text, encoding="utf-8", newline="\n")
print(f"Verified {len(seen)} table/mode truncation entries.")
