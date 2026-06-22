from gcw import cli


def test_convert_hf_forwards_icarus_variant(monkeypatch):
    calls = []

    def fake_convert_hf(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(cli, "convert_hf", fake_convert_hf)

    cli.main(
        [
            "convert",
            "hf",
            "--family",
            "llama",
            "--model",
            "tiny",
            "--variant",
            "icarus",
            "--out",
            "out",
        ]
    )

    assert calls == [
        (
            ("tiny", "out"),
            {"family": "llama", "variant": "icarus", "dtype": "bfloat16"},
        )
    ]
