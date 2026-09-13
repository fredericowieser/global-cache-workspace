import json

from scripts import run_math_experiment_eval as evaluation


def test_experiment_eval_help(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_math_experiment_eval.py", "--help"])
    try:
        evaluation.main()
    except SystemExit as error:
        assert error.code == 0


def test_ablation_reuses_one_loaded_target_for_boolean_conditions(
    tmp_path, monkeypatch
):
    loaded = (object(), object())
    loads = []
    calls = []

    def load(**kwargs):
        loads.append(kwargs)
        return loaded

    def run_all(**kwargs):
        calls.append(kwargs)
        return {"enable_thinking": kwargs["config"].enable_thinking}

    monkeypatch.setattr(evaluation, "_load_eval_target", load)
    monkeypatch.setattr(
        evaluation, "_distributed_eval_state", lambda: (None, 0, 1)
    )
    monkeypatch.setattr(evaluation, "_eval_device", lambda _rank: "cpu")
    monkeypatch.setattr(evaluation, "run_all", run_all)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_math_experiment_eval.py",
            "--hf-model",
            "model",
            "--out-dir",
            str(tmp_path / "evaluation"),
        ],
    )

    evaluation.main()

    assert len(loads) == 1
    assert [call["config"].enable_thinking for call in calls] == [False, True]
    assert all(call["loaded_target"] is loaded for call in calls)
    output = tmp_path / "evaluation"
    assert (output / "thinking-disabled" / "result.json").is_file()
    assert (output / "thinking-enabled" / "result.json").is_file()
    assert set(json.loads((output / "summary.json").read_text())) == {
        "thinking-disabled",
        "thinking-enabled",
    }
