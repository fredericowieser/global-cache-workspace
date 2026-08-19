from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_extract_paper_table():
    path = Path("scripts/extract_paper_table.py")
    spec = importlib.util.spec_from_file_location("extract_paper_table", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_paper_table_uses_martin_metric_mapping():
    module = _load_extract_paper_table()
    row = module._extract_row(
        {
            "results": {
                "gsm8k_openmath": {"exact_match,flexible": 0.1},
                "svamp": {"exact_match,flexible": 0.2},
                "gsm8k_hard": {"exact_match,flexible": 0.3},
                "theoremqa": {"exact_match,none": 0.4},
                "logiqa": {"acc,none": 0.01, "acc_norm,none": 0.5},
                "agieval_gaokao_mathqa": {"acc,none": 0.02, "acc_norm,none": 0.6},
            }
        }
    )

    assert row == {
        "GSM8K": 10.0,
        "SVAMP": 20.0,
        "GSM-Hd": 30.0,
        "ThmQA": 40.0,
        "LogiQA": 50.0,
        "Gaokao": 60.0,
        "AVG": 35.0,
    }


def test_extract_paper_table_averages_only_present_tasks():
    module = _load_extract_paper_table()
    row = module._extract_row(
        {
            "results": {
                "gsm8k_openmath": {"exact_match,flexible": 0.1},
                "svamp": {"exact_match,flexible": 0.2},
            }
        }
    )

    assert row["ThmQA"] is None
    assert row["AVG"] == 15.0
