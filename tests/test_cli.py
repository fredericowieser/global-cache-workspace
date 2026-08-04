from gcw import cli


def test_eval_opsd_kl_audit_loads_config_and_runs(monkeypatch, capsys):
    marker = object()
    calls = []
    monkeypatch.setattr(cli, "load_opsd_matched_audit_config", lambda path: (marker, path))

    def fake_run(config):
        calls.append(config)
        return {"kind": "matched-audit", "examples": 32}

    monkeypatch.setattr(cli, "run_matched_opsd_audit", fake_run)

    cli.main(["eval", "opsd-kl-audit", "--config", "audit.yaml"])

    assert calls == [(marker, "audit.yaml")]
    assert '"examples": 32' in capsys.readouterr().out


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
