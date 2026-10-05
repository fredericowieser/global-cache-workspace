import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.plot_cache_update_metrics import load_metrics, plot_metrics


def test_equal_episode_weighting_and_missing_steps(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = []
    for example, value, applications in [(0, 1.0, 1), (1, 9.0, 100)]:
        rows.append(
            {
                "schema": "gcw.cache_update_metrics.v1",
                "task": "example",
                "example_id": str(example),
                "steps": [
                    {
                        "external_iteration": 1,
                        "solver_step": step,
                        "metrics": {
                            "kv": {
                                "min": value,
                                "mean": value,
                                "max": value,
                                "std": 0.0,
                                "applications": applications,
                            }
                        },
                    }
                    for step in ([1, 2] if example == 0 else [1])
                ],
            }
        )
    rows.append({**rows[0], "example_id": "empty", "steps": []})
    path.write_text("\n".join(map(json.dumps, rows)))
    samples, counts, empty = load_metrics(path)
    assert counts == {"example": 3}
    assert empty == 1
    fig = plot_metrics(samples, counts, empty, "Test")
    try:
        assert len(fig.axes) == 4
        np.testing.assert_allclose(fig.axes[1].lines[0].get_xdata(), [1, 2])
        np.testing.assert_allclose(fig.axes[1].lines[0].get_ydata(), [5, 1])
        fig.savefig(tmp_path / "plot.png")
        assert (tmp_path / "plot.png").stat().st_size > 0
    finally:
        plt.close(fig)
    with pytest.raises(ValueError, match="tasks not found"):
        load_metrics(path, ["missing"])
    path.write_text("\n".join(map(json.dumps, [rows[0], rows[0]])))
    with pytest.raises(ValueError, match="duplicate episode"):
        load_metrics(path)


def write_v2_metrics(path, *, direction_only=False):
    steps = []
    for iteration in (1, 2):
        for step in (1, 2, 3):
            metrics = {}
            if not direction_only:
                metrics["value"] = {
                    "min": 1.0,
                    "mean": 2.0,
                    "max": 3.0,
                    "std": 1.0,
                    "applications": 2,
                    "heads": 2,
                }
            if (iteration, step) in {(1, 2), (2, 3)}:
                metrics["value_direction_change"] = {
                    "min": 0.0,
                    "mean": 1.0,
                    "max": 2.0,
                    "std": 1.0,
                    "applications": 1,
                    "heads": 2,
                    "valid_head_observations": 2,
                }
            steps.append(
                {
                    "external_iteration": iteration,
                    "solver_step": step,
                    "metrics": metrics,
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema": "gcw.cache_update_metrics.v2",
                "task": "example",
                "example_id": "0",
                "steps": steps,
            }
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "quantity, panels", [("both", 8), ("magnitude", 4), ("direction", 4)]
)
def test_v2_mixed_figures_filtering_and_missing_directions(
    tmp_path, quantity, panels
):
    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    samples, counts, empty = load_metrics(path, ["example"])
    fig = plot_metrics(samples, counts, empty, "Test", quantity)
    try:
        assert len(fig.axes) == panels
        if quantity != "magnitude":
            axis = fig.axes[-3]
            assert axis.get_ylabel() == "Cosine distance (unitless)"
            assert len(axis.lines) == 2
            x = axis.lines[0].get_xdata().tolist()
            for iteration, line in enumerate(axis.lines, 1):
                values = line.get_ydata()
                observed_step = 2 if iteration == 1 else 3
                assert values[x.index(observed_step)] == 1
                assert sum(np.isnan(values)) == len(x) - 1
        if quantity != "direction":
            assert "norm" in fig.axes[0].get_ylabel()
    finally:
        plt.close(fig)


def test_unavailable_quantity_and_default_direction_only_figure(tmp_path):
    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path, direction_only=True)
    samples, counts, empty = load_metrics(path)
    fig = plot_metrics(samples, counts, empty, "Test")
    assert len(fig.axes) == 4
    plt.close(fig)
    with pytest.raises(ValueError, match="no magnitude metrics"):
        plot_metrics(samples, counts, empty, "Test", "magnitude")
    path.write_text(
        path.read_text().replace(
            '"gcw.cache_update_metrics.v2"', '"unsupported"'
        )
    )
    with pytest.raises(ValueError, match="unsupported metrics schema"):
        load_metrics(path)


@pytest.mark.parametrize(
    "quantity, suffix",
    [(None, ".png"), ("magnitude", ".png"), ("direction", ".direction.png")],
)
def test_cli_default_output_paths(tmp_path, monkeypatch, quantity, suffix):
    from scripts.plot_cache_update_metrics import main

    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    args = ["plot_cache_update_metrics.py", str(path)]
    if quantity:
        args.extend(["--quantity", quantity])
    monkeypatch.setattr("sys.argv", args)
    main()
    assert path.with_suffix(suffix).stat().st_size > 0


def test_cli_v1_unavailable_direction_error(tmp_path, monkeypatch, capsys):
    from scripts.plot_cache_update_metrics import main

    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    row = json.loads(path.read_text())
    row["schema"] = "gcw.cache_update_metrics.v1"
    for step in row["steps"]:
        step["metrics"] = {"kv": step["metrics"]["value"]}
    path.write_text(json.dumps(row))
    monkeypatch.setattr(
        "sys.argv",
        ["plot_cache_update_metrics.py", str(path), "--quantity", "direction"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "no direction metrics" in capsys.readouterr().err
    assert not path.with_suffix(".direction.png").exists()


@pytest.mark.parametrize(
    "phase,mean", [("combined", 2), ("prefill", 1), ("decoding", 3)]
)
def test_v3_selects_phase(tmp_path, phase, mean):
    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    row = json.loads(path.read_text())
    row["schema"] = "gcw.cache_update_metrics.v3"
    row["phase_steps"] = {}
    for name, value in [("prefill", 1), ("decoding", 3)]:
        steps = json.loads(json.dumps(row["steps"]))
        for step in steps:
            step["metrics"]["value"]["mean"] = value
        row["phase_steps"][name] = steps
    empty = {
        **row,
        "example_id": "empty",
        "steps": [],
        "phase_steps": {"prefill": [], "decoding": []},
    }
    path.write_text("\n".join(map(json.dumps, [row, empty])))
    samples, counts, empty_count = load_metrics(path, phase=phase)
    assert samples[("value", 1, 1)][0][1] == mean
    assert counts == {"example": 2}
    assert empty_count == 1
    row["phase_steps"]["decoding"] = []
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="no update metrics"):
        load_metrics(path, phase="decoding")


@pytest.mark.parametrize("schema", ["v1", "v2"])
@pytest.mark.parametrize("phase", ["prefill", "decoding"])
def test_historical_phase_selection_is_rejected(tmp_path, schema, phase):
    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    row = json.loads(path.read_text())
    row["schema"] = f"gcw.cache_update_metrics.{schema}"
    path.write_text(json.dumps(row))
    with pytest.raises(
        ValueError, match="phase-specific metrics require a v3 sidecar"
    ):
        load_metrics(path, phase=phase)


@pytest.mark.parametrize("phase", ["prefill", "decoding"])
@pytest.mark.parametrize("quantity", ["both", "direction"])
def test_phase_cli_output_and_title(tmp_path, monkeypatch, phase, quantity):
    from scripts import plot_cache_update_metrics as module

    path = tmp_path / "metrics.jsonl"
    write_v2_metrics(path)
    row = json.loads(path.read_text())
    row["schema"] = "gcw.cache_update_metrics.v3"
    row["phase_steps"] = {"prefill": row["steps"], "decoding": row["steps"]}
    path.write_text(json.dumps(row))
    plot = module.plot_metrics
    titles = []

    def record_title(samples, counts, empty, title, quantity):
        titles.append(title)
        return plot(samples, counts, empty, title, quantity)

    monkeypatch.setattr(module, "plot_metrics", record_title)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_cache_update_metrics.py",
            str(path),
            "--phase",
            phase,
            "--quantity",
            quantity,
        ],
    )
    module.main()
    assert titles == [f"Diffusion update diagnostics ({phase})"]
    suffix = (
        f".{phase}" + (".direction" if quantity == "direction" else "") + ".png"
    )
    assert path.with_suffix(suffix).stat().st_size > 0
