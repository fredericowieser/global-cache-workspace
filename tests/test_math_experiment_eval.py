from scripts.run_math_experiment_eval import main


def test_experiment_eval_help(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_math_experiment_eval.py", "--help"])
    try:
        main()
    except SystemExit as error:
        assert error.code == 0
