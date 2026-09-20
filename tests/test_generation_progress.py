import types

import pytest
import torch

from gcw.config import EvalConfig
from gcw.evaluation.lm_eval_adapter import GCWEvalLM
from gcw.evaluation.progress import GenerationProgress
from gcw.evaluation.update_metrics import CacheUpdateMetrics


@pytest.mark.parametrize("reference", [False, True])
def test_generation_progress_tracks_completed_batches_and_oom_retries(
    tmp_path, monkeypatch, reference
):
    from gcw.evaluation import lm_eval_adapter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GCW_EVAL_REFERENCE_GENERATE", "1" if reference else "0")
    trackers = []

    class Tracker(GenerationProgress):
        def __init__(self, keys):
            super().__init__(keys)
            trackers.append(self)

    monkeypatch.setattr(lm_eval_adapter, "GenerationProgress", Tracker)
    observed = []

    class Model:
        def get_device(self):
            return torch.device("cpu")

        def generate(self, ids, **kwargs):
            observed.append(trackers[-1].completed)
            if ids.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            suffix = torch.tensor([[8, 9, 0]]).repeat(ids.shape[0], 1)
            return torch.cat((ids, suffix), dim=1)

        generate_fast = generate

    lm = GCWEvalLM.__new__(GCWEvalLM)
    lm.model = Model()
    lm.tokenizer = types.SimpleNamespace(
        eot_id=0,
        eos_token_id=0,
        encode=lambda text, **kwargs: [4] * len(text),
        decode=lambda tokens: " ".join(map(str, tokens)),
    )
    lm.eval_config = EvalConfig(batch_size="auto", max_batch_size=4)
    lm.update_metrics = CacheUpdateMetrics(lm.model, None)
    lm.max_length = 32
    lm.batch_size = "auto"
    lm._auto_batch_sizes = {"generation": 4}
    lm.add_special_tokens_for_generate = False
    requests = [
        types.SimpleNamespace(
            args=("aa" if i < 4 else "b", {"max_gen_toks": 3}),
            task_name="korbench_logic" if i < 4 else "korbench_cipher",
            doc_id=i,
            idx=0,
        )
        for i in range(6)
    ]
    assert lm.generate_until(requests) == ["8 9"] * 6
    assert trackers[-1].completed == trackers[-1].total == 6
    assert trackers[-1].tokens == 12
    assert observed == ([0, 1, 2, 3, 4, 5] if reference else [0, 0, 2, 4])
    assert list(tmp_path.iterdir()) == []


def test_generation_progress_logs_periodically(monkeypatch, capsys):
    from gcw.evaluation import progress as module

    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setenv("RANK", "1")
    progress = GenerationProgress([("a", str(i), "0") for i in range(3)])
    output = capsys.readouterr().out
    assert "0/3 requests" in output
    assert "rank 1" in output
    progress.advance([("a", "0", "0")], 10)
    assert capsys.readouterr().out == ""
    now[0] += 61
    progress.advance([("a", "1", "0")], 20)
    assert "2/3 requests, 30 generated tokens" in capsys.readouterr().out
    progress.advance([("a", "2", "0")], 30)
    assert "3/3 requests, 60 generated tokens" in capsys.readouterr().out
