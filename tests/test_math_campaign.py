from __future__ import annotations

from argparse import Namespace
import json

from scripts import prepare_math_campaign as campaign


class Tokenizer:
    eot_id = 0

    @staticmethod
    def encode(text):
        return list(range(len(text.split())))

    @staticmethod
    def apply_chat_template(messages, **_kwargs):
        return list(range(len(messages[0]["content"].split()) + 2))

    @staticmethod
    def identity_sha256():
        return "1" * 64


def test_campaign_builder_filters_overlap_length_and_shuffles_deterministically(tmp_path, monkeypatch):
    source = tmp_path / "openmath.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"prompt": prompt, "answer": answer})
            for prompt, answer in (("keep one", "a"), ("benchmark", "b"), ("too long prompt", "a b c d"))
        )
        + "\n"
    )
    monkeypatch.setattr(campaign, "get_tokenizer", lambda _name: Tokenizer())
    monkeypatch.setattr(campaign, "_load_benchmark_hashes", lambda: ({campaign._prompt_hash("benchmark")}, {"x": 1}))
    monkeypatch.setattr(
        campaign,
        "_verified_openr1_rows",
        lambda *_args: iter((("reasoning row", "<think>x</think> 2", "r1"),)),
    )
    args = Namespace(
        out_dir=str(tmp_path / "out"),
        tokenizer="tokenizer",
        openmath=str(source),
        sequence_length=8,
        caps=[4, 6, 8],
        seed=24,
        exclude_jsonl=[],
        openr1_repo="repo",
        openr1_config="all",
        openr1_revision="revision",
    )

    manifest = campaign.build(args)
    rows = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]

    assert manifest["rows"] == 2
    assert manifest["source_counts"] == {"openmath": 1, "openr1": 1}
    assert manifest["rejection_counts"]["openmath:benchmark_overlap"] == 1
    assert manifest["rejection_counts"]["openmath:over_context"] == 1
    assert [row["campaign_order"] for row in rows] == [0, 1]
    assert {row["enable_thinking"] for row in rows} == {False, True}
    assert len(manifest["dataset_sha256"]) == 64
