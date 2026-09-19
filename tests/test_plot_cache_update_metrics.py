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
