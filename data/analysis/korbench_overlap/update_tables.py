"""Insert audited conditional correctness percentages into the six appendix tables."""

import csv
import re

from analyze import METHODS, MODELS, OUT, PAPER

values = {
    tuple(row[k] for k in ("regime", "model", "method", "mode")): row
    for row in csv.DictReader((OUT / "summary.csv").open())
}
for filename in ("C_Extended_Results.tex", "E_Diffusion_Archive.tex"):
    path = PAPER / filename
    text = path.read_text()
    for match in list(
        re.finditer(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
    )[::-1]:
        block = match[0]
        label = re.search(r"\\label\{([^}]+)\}", block)[1]
        if "korbench" not in label:
            continue
        regime = (
            "sequential"
            if "sequential" in label
            else ("reasoning" if "reasoning-gym" in label else "math")
        )
        if "Retained & Recovered" in block:
            continue
        lines = []
        for line in block.splitlines():
            if line.startswith(r"\multicolumn") and r"\textit{" in line:
                model = MODELS[re.search(r"\\textit\{([^}]+)\}", line)[1]]
            if any(line.startswith(method + " &") for method in METHODS):
                cells = [c.strip() for c in line.rstrip()[:-2].split("&")]
                additions = []
                for metric in ("retained_pct", "recovered_pct"):
                    entries = []
                    for mode in ("off", "on") if regime == "math" else ("off",):
                        row = values.get(
                            (regime, model, METHODS[cells[0]], mode)
                        )
                        entries.append(
                            f"{float(row[metric]):.2f}"
                            if row
                            else r"\textemdash"
                        )
                    additions.append("/".join(entries))
                line = " & ".join(cells[:7] + additions + cells[7:]) + r" \\"
            lines.append(line)
        block = "\n".join(lines)
        block = block.replace(
            r"\begin{tabular}{lccccc>{\bfseries}ccc}",
            r"\begin{tabular}{lccccc>{\bfseries}ccccc}",
        )
        block = block.replace(r"\multicolumn{9}", r"\multicolumn{11}")
        block = block.replace(
            r"& \multicolumn{2}{c}{Generation tokens}",
            r"& \multicolumn{2}{c}{Backbone overlap (\%)} & \multicolumn{2}{c}{Generation tokens}",
        )
        block = block.replace(
            r"\cmidrule(l){8-9}", r"\cmidrule(l){8-9}\cmidrule(l){10-11}"
        )
        block = block.replace(
            "Avg. & Mean/resp.", "Avg. & Retained & Recovered & Mean/resp."
        )
        definition = r" Retained is the percentage of the matched pretrained backbone\'s correct answers that remain correct; Recovered is the percentage of its incorrect answers that become correct, pairing all 1,250 questions within the same thinking mode."
        definition = definition.replace("backbone\\'s", "backbone's")
        end_caption = block.index("\n\\label")
        assert block[end_caption - 1] == "}"
        block = block[: end_caption - 1] + definition + block[end_caption - 1 :]
        text = text[: match.start()] + block + text[match.end() :]
    path.write_text(text)
