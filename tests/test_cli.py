from gcw import cli


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
