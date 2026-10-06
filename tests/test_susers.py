import contextlib
import io
import json
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/cluster/susers"


class SusersTests(unittest.TestCase):
    def run_susers(self, jobs):
        output = io.StringIO()
        with patch(
            "subprocess.check_output", return_value=json.dumps({"jobs": jobs})
        ) as command, contextlib.redirect_stdout(output):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        command.assert_called_once_with(["squeue", "--json"], text=True)
        return [line.split() for line in output.getvalue().splitlines()]

    def test_prefers_aggregate_tres_without_double_counting(self):
        self.assertEqual(
            self.run_susers([{
                "job_state": ["RUNNING"],
                "user_name": "martin",
                "tres_alloc_str": "cpu=44,gres/gpu=2,gres/gpu:b300=2",
                "gres_detail": ["gpu:b300:2(IDX:1-2)"],
            }]),
            [["USER", "GPUS"], ["martin", "2"], ["TOTAL", "2"]],
        )

    def test_runpod_fallback_and_per_user_aggregation(self):
        jobs = [
            {
                "job_state": ["RUNNING"],
                "user_name": user,
                "tres_alloc_str": "cpu=22,mem=167G,node=1,billing=22",
                "gres_detail": [detail],
            }
            for user, detail in [
                ("teresa", "gpu:b300:1(IDX:0)"),
                ("martin", "gpu:b300:2(IDX:1-2)"),
                ("martin", "gpu:1(IDX:3)"),
            ]
        ]
        self.assertEqual(
            self.run_susers(jobs),
            [["USER", "GPUS"], ["martin", "3"], ["teresa", "1"],
             ["TOTAL", "4"]],
        )

    def test_sums_node_details_and_gpu_types_not_device_indices(self):
        self.assertEqual(
            self.run_susers([{
                "job_state": ["RUNNING"],
                "user_name": "martin",
                "tres_alloc_str": None,
                "gres_detail": [
                    "gpu:a100:2(IDX:0,3),gpu:h100:1(IDX:5)",
                    "gpu:2(IDX:6-7),nic:1(IDX:0)",
                ],
            }]),
            [["USER", "GPUS"], ["martin", "5"], ["TOTAL", "5"]],
        )

    def test_ignores_inactive_jobs_and_missing_gpu_allocations(self):
        self.assertEqual(
            self.run_susers([
                {"job_state": ["PENDING"], "user_name": "queued",
                 "tres_per_node": "gres/gpu:8"},
                {"job_state": ["COMPLETED"], "user_name": "finished",
                 "tres_alloc_str": "gres/gpu=8"},
                {"job_state": ["RUNNING"], "user_name": "cpu",
                 "gres_detail": None},
                {"job_state": ["RUNNING"], "user_name": "zero",
                 "tres_alloc_str": "gres/gpu=0",
                 "gres_detail": ["gpu:b300:2(IDX:0-1)"]},
            ]),
            [["USER", "GPUS"], ["TOTAL", "0"]],
        )


if __name__ == "__main__":
    unittest.main()
