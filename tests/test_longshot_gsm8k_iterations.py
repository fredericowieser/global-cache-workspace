from __future__ import annotations

from types import SimpleNamespace
import math

from gcw.analysis.longshot_gsm8k_iterations import (
    GenerationRow,
    Gsm8kExample,
    IterationExampleRow,
    _iteration_values,
    answers_match,
    extract_flexible,
    format_prompt,
    merge_shard_outputs,
    select_shard,
    summarize_rows,
    target_answer,
    write_outputs,
)


def test_gsm8k_answer_extraction_matches_flexible_last_number():
    assert target_answer("We compute 18 + 24. #### 42") == "42"
    assert extract_flexible("therefore the answer is \\boxed{1,234}.") == "1234"
    assert extract_flexible("therefore \\boxed{42}. Ignore this later distractor 999.") == "42"
    assert extract_flexible("scratchpad text #### 42.0") == "42.0"
    assert extract_flexible("The answer is -7. Earlier number was 10.") == "-7"
    assert extract_flexible("first 10 then -7") == "-7"
    assert answers_match("1,234", "1234")
    assert answers_match("42.0", "42")
    assert not answers_match("41", "42")


def test_legacy_prompt_format_falls_back_to_question_answer():
    tokenizer = SimpleNamespace(chat_template=None)

    assert format_prompt(tokenizer, "What is 2+2?", prompt_style="legacy") == "Question: What is 2+2?\nAnswer:"


def test_sparse_iteration_values_are_parsed():
    args = SimpleNamespace(iteration_list="0,1,2,4,8", max_iterations=10)

    assert _iteration_values(args) == {0, 1, 2, 4, 8}


def test_iteration_summary_collapses_splits_and_reports_accuracy():
    detail_rows = [
        IterationExampleRow(0, 0, 0, -2.0, -0.5, 1.0, 5.0, 4),
        IterationExampleRow(0, 0, 1, -4.0, -1.0, 1.0, 5.0, 4),
        IterationExampleRow(0, 1, 0, 2.0, 0.5, 2.0, 5.0, 3),
        IterationExampleRow(0, 1, 1, 4.0, 1.0, 2.0, 5.0, 3),
    ]
    generations = [
        GenerationRow(0, 0, "42", "42", True, "42"),
        GenerationRow(0, 1, "7", "8", False, "8"),
    ]

    rows = summarize_rows(detail_rows, generations, bootstrap_samples=0, seed=0)

    assert len(rows) == 1
    row = rows[0]
    assert row.iteration == 0
    assert row.correct == 1
    assert row.n_examples == 2
    assert row.accuracy == 0.5
    assert math.isclose(row.mean_synergy_raw, 0.0)
    assert math.isclose(row.mean_synergy_norm, 0.0)
    assert row.n_tokens == 7


def test_shards_cover_examples_once():
    examples = [Gsm8kExample(i, f"q{i}", f"#### {i}", str(i)) for i in range(10)]

    shards = [select_shard(examples, num_shards=4, shard_index=i) for i in range(4)]

    covered = sorted(example.index for shard in shards for example in shard)
    assert covered == list(range(10))
    assert [example.index for example in shards[0]] == [0, 4, 8]


def test_merge_shard_outputs_recomputes_summary(tmp_path):
    shard_a = tmp_path / "shard_0"
    shard_b = tmp_path / "shard_1"
    out = tmp_path / "combined"
    write_outputs(
        shard_a,
        [IterationExampleRow(0, 0, 0, -2.0, -0.5, 1.0, 5.0, 4)],
        [GenerationRow(0, 0, "42", "42", True, "42")],
        bootstrap_samples=0,
        seed=0,
    )
    write_outputs(
        shard_b,
        [IterationExampleRow(0, 1, 0, 2.0, 0.5, 2.0, 5.0, 3)],
        [GenerationRow(0, 1, "7", "8", False, "8")],
        bootstrap_samples=0,
        seed=0,
    )

    merge_shard_outputs([shard_a, shard_b], out, bootstrap_samples=0, seed=0)

    summary = (out / "summary.md").read_text()
    assert "Raw synergy, nats/token [95% CI]" in summary
    assert "| 0 | 50.00 | 0.0000 [0.0000, 0.0000] | 1.5000 | 2 |" in summary
