from scripts.run_math_campaign_eval import main


def test_campaign_eval_help(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_math_campaign_eval.py", "--help"])
    try:
        main()
    except SystemExit as error:
        assert error.code == 0
