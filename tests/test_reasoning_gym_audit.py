import numpy as np
import pytest

from gcw.data._staging import dataset_staging
from scripts.audit_reasoning_gym import truncation_statistics


def test_truncation_audit_distinguishes_eos_loss_and_prompt_only():
    counts = np.asarray([(100, 413, 0), (600, 100, 1), (200, 312, 1)])
    report = truncation_statistics(counts, 512)
    assert report["truncated_rows"] == 2
    assert report["prompt_only_rows"] == 1
    assert report["eos_only_truncated_rows"] == 1
    assert report["removed_total_tokens_truncated_rows"]["mean"] == 94.5
    assert report["removed_total_tokens_truncated_rows"]["std"] == 93.5
    assert report["removed_answer_tokens_truncated_rows"]["total"] == 101
    assert report["lost_answer_token_percent"] == pytest.approx(100 * 101 / 825)
    complete = truncation_statistics(counts, 1024)
    assert complete["truncated_rows"] == 0
    assert complete["removed_total_tokens_truncated_rows"]["mean"] is None


def test_staging_cleans_failed_build_and_keeps_published_data(tmp_path):
    with pytest.raises(RuntimeError, match="failed build"):
        with dataset_staging(tmp_path) as temporary:
            (temporary / "partial.jsonl").write_text("incomplete")
            raise RuntimeError("failed build")
    assert not temporary.exists()
    with dataset_staging(tmp_path) as temporary:
        staging = temporary / "dataset"
        staging.mkdir()
        (staging / "train.jsonl").write_text("complete")
        staging.rename(tmp_path / "published")
    assert not temporary.exists()
    assert (tmp_path / "published" / "train.jsonl").read_text() == "complete"
