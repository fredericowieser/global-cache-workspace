from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from gcw.data import reasoning_gym as preparation
from gcw.data import reasoning_gym_traces as traces
from gcw.data._vendor.reasoning_gym.algorithmic import (
    manipulate_matrix,
    string_manipulation,
    string_splitting,
)
from gcw.data._vendor.reasoning_gym.cognition.color_cube_rotation import (
    Color,
    Cube,
    Side,
)
from gcw.data.sft import iter_jsonl, sft_batch


class ByteChatTokenizer:
    eot_id = 256

    def encode(self, text):
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return self.encode("<user>" + messages[0]["content"] + "<assistant>")

    def identity_sha256(self):
        return "0123456789abcdef" * 4


def test_string_priority_uses_prompt_number_and_discards_repeated_state():
    dataset = traces.create_dataset("string_manipulation", seed=24, size=1)
    dataset.rules = [
        ("Change ba to ab.", lambda s: ("ab", 17) if s == "ba" else (s, 0)),
        ("Change ab to ba.", lambda s: ("ba", 4) if s == "ab" else (s, 0)),
    ]
    entry = {
        "question": string_manipulation.QUESTION_TEMPLATE.format(
            string="ab", rules="1. Change ba to ab.\n2. Change ab to ba."
        ),
        "answer": "ba",
        "metadata": {"solution": "do not use this", "states": ["wrong"]},
    }
    trace = traces.render_trace("string_manipulation", dataset, entry)
    assert trace.final_answer == "ba"
    assert trace.steps == 1
    assert trace.termination == "repeated_state_discarded"
    assert "Rule 2 is the first applicable rule" in trace.solution
    assert "Rule 1, the first applicable rule" in trace.solution
    assert "Rule 17" not in trace.solution


def test_nonterminating_string_generation_has_a_hard_bound():
    dataset = traces.create_dataset("string_manipulation", seed=24, size=1)
    with pytest.raises(traces.AdmissionError, match="transition_limit"):
        dataset._get_all_transforms("a", [("append", lambda s: (s + "a", 1))])


@pytest.mark.parametrize(
    "initial,limit,answer,steps,termination",
    [
        ((1, 1, 1), 20, "0 0 0 2 0 2", 5, "no_applicable_rule"),
        ((2, 2, 0), 2, "0 2 0 4 2 0", 2, "iteration_limit"),
        ((0, 0, 0), 20, "0 0 0 0 0 0", 0, "no_applicable_rule"),
    ],
)
def test_dismantling_priority_and_iteration_limits(
    initial, limit, answer, steps, termination
):
    dataset = traces.create_dataset("string_splitting", seed=24, size=1)
    entry = {
        "question": string_splitting.QUESTION_TEMPLATE.format(
            A_machine=initial[0],
            B_machine=initial[1],
            C_machine=initial[2],
            max_iterations=limit,
        ),
        "answer": answer,
        "metadata": {},
    }
    trace = traces.render_trace("string_splitting", dataset, entry)
    assert trace.final_answer == answer
    assert trace.steps == steps
    assert trace.termination == termination
    if initial == (1, 1, 1):
        assert "rule 5 (X + Y -> Z) 2 time(s)" in trace.solution


@pytest.mark.parametrize(
    "program,answer,steps",
    [("A# #B B# #B", "#B A#", 2), ("B# A# #A", "B#", 1)],
)
def test_ab_leftmost_pair_and_cancellation(program, answer, steps):
    dataset = traces.create_dataset("ab", seed=24, size=1)
    entry = {
        "question": f"Now, consider the following program:\n\n{program}\n\nReturn the final state.",
        "answer": answer,
        "metadata": {},
    }
    trace = traces.render_trace("ab", dataset, entry)
    assert trace.steps == steps
    assert trace.final_answer == answer


def test_empty_ab_answers_are_explicitly_excluded():
    dataset = traces.create_dataset("ab", seed=24, size=1)
    entry = {
        "question": "Now, consider the following program:\n\nA# #A\n\nReturn the final state.",
        "answer": "",
        "metadata": {},
    }
    with pytest.raises(traces.AdmissionError, match="empty_final_answer"):
        traces.render_trace("ab", dataset, entry)


@pytest.mark.parametrize(
    "rotation,expected",
    [
        ("front", ["blue", "green", "orange", "yellow", "red", "white"]),
        ("right", ["green", "orange", "blue", "red", "white", "yellow"]),
        ("back", ["white", "green", "red", "yellow", "orange", "blue"]),
        ("left", ["yellow", "red", "blue", "orange", "white", "green"]),
        ("bottom", ["orange", "green", "white", "yellow", "blue", "red"]),
    ],
)
def test_cube_rotations_against_hand_labelled_faces(rotation, expected):
    dataset = traces.create_dataset("color_cube_rotation", seed=24, size=1)
    initial = dict(
        zip(
            Side,
            [
                Color.RED,
                Color.GREEN,
                Color.BLUE,
                Color.YELLOW,
                Color.WHITE,
                Color.ORANGE,
            ],
        )
    )
    cube = Cube(initial.copy())
    dataset._rotate_to_top(cube, Side(rotation))
    assert [cube.colors[side].value for side in Side] == expected
    repetitions = 2 if rotation == "bottom" else 4
    for _ in range(repetitions - 1):
        dataset._rotate_to_top(cube, Side(rotation))
    assert cube.colors == initial


@pytest.mark.parametrize(
    "method,args,expected",
    [
        ("_rot90", (), [[4, 1], [5, 2], [6, 3]]),
        ("_rot180", (), [[6, 5, 4], [3, 2, 1]]),
        ("_rot270", (), [[3, 6], [2, 5], [1, 4]]),
        ("_hmirror", (), [[4, 5, 6], [1, 2, 3]]),
        ("_vmirror", (), [[3, 2, 1], [6, 5, 4]]),
        ("_dmirror", (), [[1, 4], [2, 5], [3, 6]]),
        ("_cmirror", (), [[6, 3], [5, 2], [4, 1]]),
        ("_map", (2, 9), [[1, 9, 3], [4, 5, 6]]),
        ("_zero_divisible", (2,), [[1, 0, 3], [0, 5, 0]]),
        ("_crop", (1, 2, 2, 3), [[2, 3], [5, 6]]),
        ("_remove_every_nth_row", (2,), [[1, 2, 3]]),
        ("_remove_every_nth_col", (2,), [[1, 3], [4, 6]]),
    ],
)
def test_matrix_operations_against_hand_computed_results(
    method, args, expected
):
    dataset = traces.create_dataset("manipulate_matrix", seed=24, size=1)
    assert getattr(dataset, method)([[1, 2, 3], [4, 5, 6]], *args) == expected


def test_matrix_trace_renders_intermediate_matrices():
    dataset = traces.create_dataset("manipulate_matrix", seed=24, size=1)
    operations = [
        {
            "transform": "rotate",
            "degrees": "90",
            "instruction": "- Rotate the matrix 90 degrees",
        },
        {
            "transform": "map",
            "from": 5,
            "to": 0,
            "instruction": "- Map each occurrence of 5 to 0",
        },
    ]
    entry = {
        "question": manipulate_matrix.QUESTION_TEMPLATE.format(
            matrix="1 2 3\n4 5 6",
            operations="\n".join(op["instruction"] for op in operations),
        ),
        "answer": "4 1\n0 2\n6 3",
        "metadata": {"operations": operations, "solution": "wrong"},
    }
    trace = traces.render_trace("manipulate_matrix", dataset, entry)
    assert "4 1\n5 2\n6 3" in trace.solution
    assert trace.final_answer == "4 1\n0 2\n6 3"
    assert trace.steps == 2


def test_gcd_trace_uses_euclidean_divisions_and_reduces_all_numbers():
    dataset = traces.create_dataset("gcd", seed=24, size=1)
    entry = {
        "question": "Find the Greatest Common Divisor (GCD) of these numbers: 84, 30, 18. Give only the GCD as your final answer.",
        "answer": "6",
        "metadata": {"result": 999},
    }
    trace = traces.render_trace("gcd", dataset, entry)
    assert "84 = 2 * 30 + 24." in trace.solution
    assert "30 = 1 * 24 + 6." in trace.solution
    assert "18 = 3 * 6 + 0." in trace.solution
    assert trace.final_answer == "6"
    assert "Give only" not in trace.prompt


@pytest.mark.parametrize("task", list(traces.TASKS))
def test_seeded_reconstruction_does_not_use_hidden_solutions(task):
    dataset = traces.create_dataset(task, seed=123456, size=200)
    accepted = 0
    for index in range(200):
        entry = deepcopy(dataset[index])
        for field in ("solution", "states", "result"):
            entry["metadata"][field] = "hidden result deliberately corrupted"
        try:
            trace = traces.render_trace(task, dataset, entry)
        except traces.AdmissionError as exc:
            assert str(exc) == "empty_final_answer"
            continue
        assert trace.final_answer == entry["answer"]
        assert trace.solution
        assert (
            traces.extract_final_answer(
                "<answer>" + trace.final_answer + "</answer>"
            )
            == trace.final_answer
        )
        accepted += 1
    assert accepted > 190


def test_reconstruction_mismatch_is_fatal():
    dataset = traces.create_dataset("gcd", seed=24, size=1)
    entry = dataset[0]
    entry["answer"] = "incorrect"
    with pytest.raises(ValueError, match="disagrees"):
        traces.render_trace("gcd", dataset, entry)


def test_preparation_is_reproducible_balanced_disjoint_and_sft_compatible(
    tmp_path,
):
    tokenizer = ByteChatTokenizer()
    kwargs = dict(
        train_rows=60,
        validation_rows=12,
        test_rows=12,
        sequence_length=12000,
        batch_size=16,
    )
    first = preparation.prepare_dataset(tmp_path / "a", tokenizer, **kwargs)
    second = preparation.prepare_dataset(
        tmp_path / "b", tokenizer, workers=2, **kwargs
    )
    assert first == second
    prompts, problems, seeds = set(), set(), set()
    for split, count in [("train", 60), ("validation", 12), ("test", 12)]:
        path = tmp_path / "a" / f"{split}.jsonl"
        assert path.read_bytes() == (tmp_path / "b" / path.name).read_bytes()
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == first["files"][split]["sha256"]
        )
        rows = list(iter_jsonl(path))
        assert len(rows) == count
        for row in rows:
            assert row["prompt_sha256"] not in prompts
            assert row["problem_sha256"] not in problems
            assert row["effective_seed"] not in seeds
            prompts.add(row["prompt_sha256"])
            problems.add(row["problem_sha256"])
            seeds.add(row["effective_seed"])
            assert row["trace_steps"] >= 2
            assert (
                traces.extract_final_answer(row["answer"])
                == row["final_answer"]
            )
            x, y = sft_batch(
                [row], tokenizer, max_length=12000, chat_template=True
            )
            assert x.shape[1] == row["sequence_tokens"]
            assert (y != -100).sum().item() == row["target_tokens"]
            assert x[0, -1].item() == tokenizer.eot_id
        for task in traces.TASKS:
            report = first["tasks"][task][split]
            assert report["retained"] == count // 6
            assert report["attempted"] == report["retained"] + sum(
                report["rejections"].values()
            )
    with pytest.raises(FileExistsError):
        preparation.prepare_dataset(tmp_path / "a", tokenizer, **kwargs)


def test_overlong_dataset_fails_without_publishing_partial_data(tmp_path):
    output = tmp_path / "too-short"
    with pytest.raises(RuntimeError, match="undersized"):
        preparation.prepare_dataset(
            output,
            ByteChatTokenizer(),
            train_rows=6,
            validation_rows=0,
            test_rows=0,
            sequence_length=1,
            attempts_per_row=2,
        )
    assert not output.exists()
    failure = json.loads(output.with_name("too-short.failed.json").read_text())
    assert any(
        task["train"]["rejections"].get("over_context")
        for task in failure["report"]["tasks"].values()
    )


def test_same_input_with_different_cube_wording_is_deduplicated(
    tmp_path, monkeypatch
):
    original = preparation._generate_batch

    def duplicate_jobs(job):
        rows, rejected = original(job)
        if rows:
            for row in rows:
                row["problem_sha256"] = job[0]
        return rows, rejected

    monkeypatch.setattr(preparation, "_generate_batch", duplicate_jobs)
    with pytest.raises(RuntimeError, match="undersized"):
        preparation.prepare_dataset(
            tmp_path / "duplicates",
            ByteChatTokenizer(),
            train_rows=12,
            validation_rows=0,
            test_rows=0,
            sequence_length=12000,
            attempts_per_row=3,
        )
    failure = json.loads((tmp_path / "duplicates.failed.json").read_text())
    assert any(
        task["train"]["rejections"].get("duplicate_prompt_or_input")
        for task in failure["report"]["tasks"].values()
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("work\n<answer>1 2\n3 4</answer>", "1 2\n3 4"),
        ("<answer>a</answer><answer>b</answer>", None),
        ("no labelled answer", None),
    ],
)
def test_answer_extraction(text, expected):
    assert traces.extract_final_answer(text) == expected


def test_bundled_sources_match_recorded_hashes():
    sources = json.loads((preparation.VENDOR / "SOURCES.json").read_text())
    assert sources["revision"] == "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
    for name, hashes in sources["files"].items():
        assert (
            hashlib.sha256((preparation.VENDOR / name).read_bytes()).hexdigest()
            == hashes["adapted_sha256"]
        )


@pytest.mark.parametrize(
    "suffix,rows,cap", [("", 700032, 4096), ("_128k", 128000, 1536)]
)
def test_training_templates_share_data_backbone_and_exposure(suffix, rows, cap):
    from gcw.architecture import resolve_experiment, resolve_model
    from gcw.config import load_train_config
    from gcw.params import processor_params

    configs = {
        name: load_train_config(
            Path("configs/train/reasoning_gym") / f"{name}{suffix}.yaml"
        )
        for name in ("full_sft", "lora", "longshot")
    }
    assert len({c.backbone_checkpoint for c in configs.values()}) == 1
    assert len({c.data.dataset for c in configs.values()}) == 1
    assert len({c.data.tokenizer_source for c in configs.values()}) == 1
    for config in configs.values():
        _, model_receipt = resolve_model(config.model).require()
        resolve_experiment(config, model_receipt).require()
        assert config.max_steps * config.global_batch_rows == rows
        assert config.sequence_len == cap
        assert config.data.sft_take_rows == rows
        assert config.batch_size * config.eval_batches == 768
        assert config.data.sft_chat_template
        assert not config.data.sft_enable_thinking
        assert config.data.sft_loss_scope == "answer_only"
        assert (
            config.data.validation_dataset
            == "data/train-data/reasoning-gym/validation.jsonl"
        )
        expected_directory = "reasoning-gym-128k" if suffix else "reasoning-gym"
        assert (
            config.data.dataset
            == f"data/train-data/{expected_directory}/train.jsonl"
        )
        assert config.data.prompt_suffix == config.data.response_prefix == ""
    assert configs["full_sft"].trainable == "backbone"
    assert configs["lora"].trainable == "lora"
    assert configs["longshot"].trainable == "cache_processor"
    assert configs["longshot"].freeze_backbone
    assert processor_params(configs["longshot"].model) == 22131841


def test_pinned_tokenizer_batched_and_scalar_counts_match_sft():
    from gcw.data.tokenizers import get_tokenizer

    source = Path("artifacts/tokenizers/Qwen3.5-2B")
    if not (source / "tokenizer.json").is_file():
        pytest.skip(
            "Download the pinned local tokenizer using the Reasoning Gym docs"
        )
    tokenizer = get_tokenizer("hf:" + source.as_posix())
    assert (
        tokenizer.identity_sha256()
        == "5de890504409a3c030c22818ca38ba0469abe07935719b95139e2ef10319b02c"
    )
    rows = [
        json.loads(line)
        for line in Path("data/train-data/reasoning-gym-examples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    class ScalarTokenizer:
        encode = tokenizer.encode
        apply_chat_template = tokenizer.apply_chat_template

    lengths = preparation._lengths(tokenizer, rows)
    assert lengths == preparation._lengths(ScalarTokenizer(), rows)
    for row, (prompt, target) in zip(rows, lengths, strict=True):
        assert (prompt, target) == (row["prompt_tokens"], row["target_tokens"])
        x, y = sft_batch([row], tokenizer, max_length=4096, chat_template=True)
        assert x.shape[1] == prompt + target
        assert (y != -100).sum().item() == target
        assert x[0, -1].item() == tokenizer.eot_id


@pytest.mark.parametrize("transitions", [255, 256, 257])
def test_string_transition_bound_includes_256(transitions):
    dataset = traces.create_dataset("string_manipulation", seed=24, size=1)
    dataset.rules = [
        (
            "Remove the first character.",
            lambda s: (s[1:], 1) if len(s) > 1 else (s, 0),
        )
    ]
    initial = "a" * (transitions + 1)
    entry = {
        "question": string_manipulation.QUESTION_TEMPLATE.format(
            string=initial, rules="1. Remove the first character."
        ),
        "answer": "a",
        "metadata": {},
    }
    if transitions > traces.MAX_TRANSITIONS:
        with pytest.raises(traces.AdmissionError, match="transition_limit"):
            dataset._get_all_transforms(initial, dataset.rules)
        with pytest.raises(traces.AdmissionError, match="transition_limit"):
            traces.render_trace("string_manipulation", dataset, entry)
    else:
        assert (
            len(dataset._get_all_transforms(initial, dataset.rules))
            == transitions + 1
        )
        assert (
            traces.render_trace("string_manipulation", dataset, entry).steps
            == transitions
        )


def test_vendor_pin_covers_support_files_and_manifest():
    payload = (preparation.VENDOR / "SOURCES.json").read_bytes()
    assert (
        hashlib.sha256(payload).hexdigest() == preparation.VENDOR_SOURCES_SHA256
    )
    sources = json.loads(payload)
    for name, expected in sources["supporting_files"].items():
        assert (
            hashlib.sha256((preparation.VENDOR / name).read_bytes()).hexdigest()
            == expected
        )
    actual = {
        p.relative_to(preparation.VENDOR).as_posix()
        for p in preparation.VENDOR.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    }
    assert actual == (
        set(sources["files"])
        | set(sources["supporting_files"])
        | {"SOURCES.json"}
    ) - {"../__init__.py"}
