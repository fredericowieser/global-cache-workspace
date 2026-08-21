from scripts.select_math_experiment_lrs import select


def test_lr_selection_uses_final_held_out_loss(tmp_path):
    for rate, loss in (("0.0001", 0.8), ("0.0002", 0.7)):
        path = tmp_path / "bgt" / rate
        path.mkdir(parents=True)
        (path / "train.log").write_text(
            f"step 0/8 val_loss=1.0000\nstep 8/8 loss=0.5 val_loss={loss}\n"
        )

    payload = select(tmp_path)

    assert payload["selected"] == {"bgt": 0.0002}


def test_launch_script_help(monkeypatch):
    from scripts.build_math_experiment_launch import main

    monkeypatch.setattr(
        "sys.argv", ["build_math_experiment_launch.py", "--help"]
    )
    try:
        main()
    except SystemExit as error:
        assert error.code == 0
