from scripts.extract_paper_table import _extract_row, _target_for_row


def test_extract_paper_table_uses_paper_column_metrics():
    payload = {
        "results": {
            "gsm8k_openmath": {
                "exact_match,flexible": 0.2,
                "exact_match,boxed": 0.1,
            },
            "svamp": {"exact_match,flexible": 0.3},
            "gsm8k_hard": {
                "exact_match,flexible": 0.4,
                "exact_match,boxed": 0.1,
            },
            "theoremqa": {"exact_match,none": 0.5},
            "logiqa": {"acc,none": 0.6, "acc_norm,none": 0.7},
            "agieval_gaokao_mathqa": {"acc_norm,none": 0.8},
        }
    }

    row = _extract_row(payload)

    assert row["GSM8K"] == 20.0
    assert row["SVAMP"] == 30.0
    assert row["GSM-Hd"] == 40.0
    assert row["ThmQA"] == 50.0
    assert row["LogiQA"] == 70.0
    assert row["Gaokao"] == 80.0
    assert row["AVG"] == 48.333333333333336


def test_target_average_matches_present_columns():
    target = _target_for_row(
        {
            "GSM8K": 35.33,
            "SVAMP": 45.33,
            "GSM-Hd": 8.11,
            "ThmQA": None,
            "LogiQA": 29.03,
            "Gaokao": 25.64,
            "AVG": 28.69,
        }
    )

    assert target["AVG"] == (35.60 + 46.70 + 7.70 + 29.50 + 25.40) / 5
