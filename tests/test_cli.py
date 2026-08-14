from gcw import cli


def test_chat_eval_parser_requires_explicit_mode_tasks_and_target():
    args = cli.build_parser().parse_args(
        [
            "eval",
            "chat",
            "--hf-model",
            "model",
            "--tasks",
            "mmlu",
            "gsm8k_openmath",
            "--mode",
            "thinking_on",
        ]
    )
    assert args.suite == "chat"
    assert args.tasks == ["mmlu", "gsm8k_openmath"]
    assert args.mode == "thinking_on"
    assert args.max_new_tokens == 2048


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

    assert calls == [(("tiny", "out"), {"family": "qwen3_5", "variant": "tf", "dtype": "bfloat16"})]


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

    assert calls == [(("tiny", "out"), {"family": "gemma4", "variant": "tf", "dtype": "bfloat16"})]
