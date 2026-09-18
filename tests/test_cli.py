import json

import pytest

from gcw import cli


def test_chat_eval_parser_leaves_thinking_unspecified():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "chat",
            "--hf-model",
            "model",
            "--tasks",
            "mmlu",
            "gsm8k_openmath",
        ]
    )
    assert args.suite == "chat"
    assert args.tasks == ["mmlu", "gsm8k_openmath"]
    assert args.enable_thinking is None
    assert args.max_new_tokens is None


def test_chat_eval_parser_can_enable_thinking():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "chat",
            "--hf-model",
            "model",
            "--tasks",
            "mmlu",
            "--enable-thinking",
        ]
    )
    assert args.enable_thinking is True


@pytest.mark.parametrize("suite", ["all", "math", "general", "tasks"])
def test_standard_eval_parser_exposes_boolean_thinking_config(suite):
    base = ["eval", suite, "--hf-model", "model"]
    if suite == "tasks":
        base.extend(["--tasks", "piqa"])
    disabled = cli.build_parser().parse_args(base)
    enabled = cli.build_parser().parse_args(
        [*base, "--apply-chat-template", "--enable-thinking"]
    )
    assert disabled.apply_chat_template is False
    assert disabled.enable_thinking is None
    assert enabled.apply_chat_template is True
    assert enabled.enable_thinking is True


@pytest.mark.parametrize(
    "removed",
    [
        ["--prompt-mode", "plain"],
        ["--mode", "thinking_off"],
        ["--modes", "thinking_on"],
    ],
)
def test_eval_parser_rejects_removed_thinking_interfaces(removed):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            ["eval", "math", "--hf-model", "model", *removed]
        )


def test_specialized_eval_parsers_use_only_boolean_thinking_flags():
    parser = cli.build_parser()
    gsm = parser.parse_args(
        [
            "eval",
            "gsm8k-reproduction",
            "--model-path",
            "model",
            "--output-dir",
            "out",
        ]
    )
    assert gsm.enable_thinking is None
    answer_nll = parser.parse_args(
        [
            "eval",
            "answer-nll",
            "--checkpoint",
            "checkpoint",
            "--expected-manifest-sha256",
            "digest",
            "--data",
            "data",
            "--expected-data-sha256",
            "digest",
            "--expected-count",
            "1",
            "--out",
            "out",
        ]
    )
    assert answer_nll.apply_chat_template is None
    assert answer_nll.enable_thinking is None
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "eval",
                "gsm8k-reproduction",
                "--model-path",
                "model",
                "--output-dir",
                "out",
                "--modes",
                "thinking_on",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "eval",
                "chat",
                "--hf-model",
                "model",
                "--tasks",
                "piqa",
                "--mode",
                "thinking_off",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "eval",
                "answer-nll",
                "--checkpoint",
                "checkpoint",
                "--expected-manifest-sha256",
                "digest",
                "--data",
                "data",
                "--expected-data-sha256",
                "digest",
                "--expected-count",
                "1",
                "--out",
                "out",
                "--chat-template",
            ]
        )


def test_paper7_eval_parsers_expose_shared_options():
    for suite in ("paper7", "paper7-legacy"):
        args = cli.build_parser().parse_args(
            [
                "eval",
                suite,
                "--checkpoint",
                "checkpoint-3000",
                "--batch-size",
                "16",
                "--generation-process-every",
                "1",
                "--head-selection-method",
                "random",
            ]
        )
        assert args.suite == suite
        assert args.checkpoint == "checkpoint-3000"
        assert args.batch_size == 16
        assert args.max_batch_size == 64
        assert args.generation_process_every == 1
        assert args.head_selection_method == "random"
        assert args.enable_thinking is None
        assert args.apply_chat_template is True
        assert args.dtype == "auto"
        assert not hasattr(args, "prompt_mode")
        assert not hasattr(args, "processor")


@pytest.mark.parametrize(
    ("suite", "target"),
    [
        ("paper7", ["--checkpoint", "checkpoint"]),
        ("paper7-legacy", ["--checkpoint", "checkpoint"]),
        ("all", ["--hf-model", "model"]),
        ("math", ["--hf-model", "model"]),
        ("general", ["--hf-model", "model"]),
        ("tasks", ["--hf-model", "model", "--tasks", "piqa"]),
    ],
)
def test_lm_eval_parsers_accept_automatic_batching(suite, target):
    args = cli.build_parser().parse_args(
        [
            "eval",
            suite,
            *target,
            "--batch-size",
            "auto",
            "--max-batch-size",
            "32",
        ]
    )

    assert args.batch_size == "auto"
    assert args.max_batch_size == 32


def test_chat_eval_parser_rejects_automatic_batching():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(
            [
                "eval",
                "chat",
                "--hf-model",
                "model",
                "--tasks",
                "piqa",
                "--batch-size",
                "auto",
            ]
        )


def test_legacy_paper7_eval_parser_can_enable_thinking():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "paper7-legacy",
            "--checkpoint",
            "checkpoint-3000",
            "--enable-thinking",
        ]
    )

    assert args.enable_thinking is True


def test_eval_parser_accepts_explicit_boolean_values():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "paper7-legacy",
            "--checkpoint",
            "checkpoint-3000",
            "--apply-chat-template",
            "false",
            "--enable-thinking",
            "false",
        ]
    )
    assert args.apply_chat_template is False
    assert args.enable_thinking is False


def test_thinking_falls_back_to_checkpoint_training_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {"training_fingerprint": {"data": {"sft_enable_thinking": True}}}
        )
    )

    assert cli._resolve_enable_thinking(
        None, checkpoint=checkpoint, apply_chat_template=True
    )
    assert not cli._resolve_enable_thinking(
        None, checkpoint=checkpoint, apply_chat_template=False
    )
    assert not cli._resolve_enable_thinking(
        False, checkpoint=checkpoint, apply_chat_template=True
    )
    assert not cli._resolve_enable_thinking(
        None, checkpoint=tmp_path / "raw-backbone", apply_chat_template=True
    )


def test_thinking_falls_back_to_published_run_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(json.dumps({"complete": True}))
    (checkpoint / "run_config.json").write_text(
        json.dumps({"data": {"sft_enable_thinking": True}})
    )

    assert cli._resolve_enable_thinking(
        None, checkpoint=checkpoint, apply_chat_template=True
    )


def test_thinking_fallback_rejects_nonboolean_training_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {"training_fingerprint": {"data": {"sft_enable_thinking": "false"}}}
        )
    )

    with pytest.raises(ValueError, match="must be boolean"):
        cli._resolve_enable_thinking(
            None, checkpoint=checkpoint, apply_chat_template=True
        )


def test_legacy_paper7_eval_parser_rejects_nonpositive_generation_cadence():
    for value in ("0", "-1"):
        with pytest.raises(SystemExit) as exc_info:
            cli.build_parser().parse_args(
                [
                    "eval",
                    "paper7-legacy",
                    "--checkpoint",
                    "checkpoint-3000",
                    "--generation-process-every",
                    value,
                ]
            )
        assert exc_info.value.code == 2


def test_generation_cadence_is_available_to_all_generative_eval_parsers():
    standard = cli.build_parser().parse_args(
        [
            "eval",
            "math",
            "--checkpoint",
            "checkpoint-3000",
            "--generation-process-every",
            "2",
        ]
    )
    chat = cli.build_parser().parse_args(
        [
            "eval",
            "chat",
            "--checkpoint",
            "checkpoint-3000",
            "--tokenizer-path",
            "tokenizer",
            "--tasks",
            "gsm8k_openmath",
            "--generation-process-every",
            "2",
        ]
    )

    assert standard.generation_process_every == 2
    assert chat.generation_process_every == 2


def test_convert_hf_forwards_processor_variants(monkeypatch):
    calls = []

    def fake_convert_hf(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cli, "convert_hf", fake_convert_hf)

    for variant in ("icarus", "longshot"):
        cli.main(
            [
                "convert",
                "hf",
                "--family",
                "llama",
                "--model",
                "tiny",
                "--variant",
                variant,
                "--out",
                "out",
            ]
        )

    assert calls == [
        (
            ("tiny", "out"),
            {"family": "llama", "variant": "icarus", "dtype": "bfloat16"},
        ),
        (
            ("tiny", "out"),
            {"family": "llama", "variant": "longshot", "dtype": "bfloat16"},
        ),
    ]


def test_convert_hf_accepts_qwen35_family(monkeypatch):
    calls = []

    def fake_convert_hf(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cli, "convert_hf", fake_convert_hf)

    cli.main(
        [
            "convert",
            "hf",
            "--family",
            "qwen3_5",
            "--model",
            "tiny",
            "--out",
            "out",
        ]
    )

    assert calls == [
        (
            ("tiny", "out"),
            {"family": "qwen3_5", "variant": "tf", "dtype": "bfloat16"},
        )
    ]


def test_convert_hf_accepts_gemma4_family(monkeypatch):
    calls = []

    def fake_convert_hf(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cli, "convert_hf", fake_convert_hf)

    cli.main(
        [
            "convert",
            "hf",
            "--family",
            "gemma4",
            "--model",
            "tiny",
            "--out",
            "out",
        ]
    )

    assert calls == [
        (
            ("tiny", "out"),
            {"family": "gemma4", "variant": "tf", "dtype": "bfloat16"},
        )
    ]


@pytest.mark.parametrize(
    "model,enabled",
    [
        ({"variant": "longshot", "diffusion_processor": {}, "cache_processor": {}}, True),
        ({"variant": "longshot", "diffusion_processor": None, "cache_processor": {}}, False),
        ({"variant": "bgt", "diffusion_processor": {}, "cache_processor": {}}, False),
        ({"model_type": "qwen3_5"}, False),
    ],
)
def test_diffusion_metrics_temporary_default(tmp_path, model, enabled):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model": model}))
    out = tmp_path / "eval.json"
    parser = cli.build_parser()
    command = ["eval", "paper7", "--checkpoint", str(checkpoint), "--out", str(out)]
    args = parser.parse_args(command)
    expected = str(tmp_path / "eval.cache-update-metrics.jsonl") if enabled else None
    assert cli._resolve_cache_update_metrics(args) == expected
    args = parser.parse_args([*command, "--cache-update-metrics", "custom.jsonl"])
    assert cli._resolve_cache_update_metrics(args) == "custom.jsonl"


def test_diffusion_metrics_default_requires_processor_and_destination(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model": {
        "variant": "longshot", "diffusion_processor": {}, "cache_processor": {},
    }}))
    args = cli.build_parser().parse_args([
        "eval", "paper7", "--checkpoint", str(tmp_path), "--out", str(tmp_path / "eval.json"),
    ])
    args.processor = "off"
    assert cli._resolve_cache_update_metrics(args) is None
    args.processor = "on"
    assert cli._resolve_cache_update_metrics(args).endswith(".cache-update-metrics.jsonl")
    args.out = None
    assert cli._resolve_cache_update_metrics(args) is None
