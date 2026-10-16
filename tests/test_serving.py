import json
import os
import select
import subprocess
import sys
import threading
import warnings
from http.client import HTTPConnection, RemoteDisconnected
from pathlib import Path
from typing import get_args

import pytest
import torch
from gcw.checkpoint import save_checkpoint
from gcw.config.types import Family, Variant
from gcw.serving import (
    NativeService,
    RequestError,
    ServiceConfig,
    ServiceHTTPServer,
    tokenizer_directory_sha256,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**overrides):
    return ServiceConfig(
        {
            "model": "gcw-test",
            "port": 0,
            "max_output_tokens": 8,
            "checkpoint": {"kind": "random", "sequence_len": 128},
            **overrides,
        }
    )


def request(text="hi", **overrides):
    return {
        "model": "gcw-test",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 3,
        **overrides,
    }


@pytest.fixture
def service():
    return NativeService(config())


@pytest.fixture
def server(service):
    with ServiceHTTPServer(service) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()


def http(server, method, path, payload=None, body=None, timeout=10):
    connection = HTTPConnection(*server.server_address, timeout=timeout)
    if payload is not None:
        body = json.dumps(payload).encode()
    try:
        connection.request(
            method,
            path,
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        assert response.getheader("Content-Type") == "application/json"
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@pytest.mark.parametrize(
    "duplicate",
    [
        b'"temperature":1,"temperature":0',
        b'"messages":[{"role":"tool","role":"user","content":"hi"}]',
    ],
)
def test_http_rejects_duplicate_fields_before_generation(
    server, monkeypatch, duplicate
):
    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate JSON must not reach native generation")

    monkeypatch.setattr(server.service.model, "generate", forbidden)
    body = (
        b'{"model":"gcw-test","messages":[{"role":"user","content":"hi"}],"max_tokens":1,'
        + duplicate
        + b"}"
    )
    status, payload = http(server, "POST", "/v1/chat/completions", body=body)
    assert status == 400 and payload["error"]


def test_config_rejects_duplicate_fields(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"model":"first","model":"second","checkpoint":{"kind":"random"}}'
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        ServiceConfig.from_file(path)


def test_http_lifecycle_and_native_generation(server):
    status, health = http(server, "GET", "/health")
    assert status == 200 and health["status"] == "ok"
    assert health["gcw"]["device"] == "cpu"
    assert health["gcw"]["tokenizer"]["chat_capability"] is False
    assert health["gcw"]["cache"]["physical_cache_bytes"] is None
    status, models = http(server, "GET", "/v1/models")
    assert status == 200
    assert models["data"][0]["id"] == "gcw-test"
    status, completion = http(server, "POST", "/v1/chat/completions", request())
    assert status == 200
    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"]["role"] == "assistant"
    assert isinstance(completion["choices"][0]["message"]["content"], str)
    assert completion["choices"][0]["finish_reason"] == "length"
    prompt_tokens = len(b"user: hi\nassistant:")
    assert completion["usage"] == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 3,
        "total_tokens": prompt_tokens + 3,
    }


def test_discovery_during_generation_and_serial_native_calls(
    server, monkeypatch
):
    service = server.service
    entered = threading.Event()
    release = threading.Event()
    second_waiting = threading.Event()
    second_entered = threading.Event()
    original_lock = service.lock
    calls = []
    results, errors = [], []

    class ObservedLock:
        def __enter__(self):
            if entered.is_set():
                second_waiting.set()
            return original_lock.__enter__()

        def __exit__(self, *args):
            return original_lock.__exit__(*args)

    def generate(input_ids, max_tokens, **kwargs):
        calls.append(input_ids.tolist())
        assert original_lock.locked()
        if len(calls) == 1:
            entered.set()
            assert release.wait(timeout=10), "held generation was not released"
        else:
            second_entered.set()
        return torch.cat(
            (input_ids, input_ids.new_full((1, max_tokens), ord("A"))), dim=1
        )

    def complete(text):
        try:
            results.append(
                http(server, "POST", "/v1/chat/completions", request(text))
            )
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(service, "lock", ObservedLock())
    monkeypatch.setattr(service.model, "generate", generate)
    clients = [
        threading.Thread(target=complete, args=(text,), daemon=True)
        for text in ("first", "second")
    ]
    try:
        clients[0].start()
        assert entered.wait(timeout=5)
        clients[1].start()
        assert second_waiting.wait(
            timeout=5
        ), "second handler never reached lock"
        for path in ("/health", "/v1/models"):
            status, payload = http(server, "GET", path, timeout=1)
            assert status == 200
            if path == "/health":
                assert payload["status"] == "ok"
            else:
                assert payload["data"][0]["id"] == service.config.model
        assert not release.is_set()
        assert not second_entered.is_set()
        assert len(calls) == 1
    finally:
        release.set()
        for client in clients:
            if client.ident is not None:
                client.join(timeout=15)
                assert not client.is_alive()
    assert not errors
    assert second_entered.is_set()
    assert calls == [
        [service.tokenizer.encode(request(text)["messages"])]
        for text in ("first", "second")
    ]
    assert len(results) == 2
    assert all(status == 200 for status, _ in results)
    assert all(
        result["choices"][0]["message"]["content"] == "AAA"
        and result["usage"]["completion_tokens"] == 3
        for _, result in results
    )


def test_http_requests_keep_native_calls_and_fresh_caches(server, monkeypatch):
    service = server.service
    original_generate = service.model.generate
    original_forward = service.model.forward_chunk
    calls, caches = [], []

    def generate(input_ids, max_tokens, **kwargs):
        calls.append((input_ids.tolist(), max_tokens, kwargs))
        return original_generate(input_ids, max_tokens, **kwargs)

    def forward(input_ids, start_pos, cache, **kwargs):
        if start_pos == 0:
            assert not cache.keys
            caches.append(cache)
        return original_forward(input_ids, start_pos, cache, **kwargs)

    monkeypatch.setattr(service.model, "generate", generate)
    monkeypatch.setattr(service.model, "forward_chunk", forward)
    results = []
    for text in ("hi", "different", "hi"):
        status, result = http(
            server, "POST", "/v1/chat/completions", request(text)
        )
        assert status == 200
        results.append(result)
    assert calls == [
        (
            [service.tokenizer.encode(request(text)["messages"])],
            3,
            {"eos_token_id": None},
        )
        for text in ("hi", "different", "hi")
    ]
    assert len(caches) == len({id(cache) for cache in caches}) == 3
    assert results[0]["choices"] == results[2]["choices"]
    assert results[0]["usage"] == results[2]["usage"]


def test_handler_bound_and_close_joins_owned_daemon_threads(
    service, monkeypatch
):
    server = ServiceHTTPServer(service)
    release = threading.Event()
    all_entered = threading.Event()
    joining = threading.Event()
    closed = threading.Event()
    owned = []
    guard = threading.Lock()
    original = server.finish_request

    def held_handler(*args):
        with guard:
            owned.append(threading.current_thread())
            if len(owned) == server.max_workers:
                all_entered.set()
        assert release.wait(timeout=10), "held handlers were not released"
        original(*args)

    def close():
        try:
            server.server_close()
        finally:
            closed.set()

    monkeypatch.setattr(server, "finish_request", held_handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    connections = []
    serving.start()
    try:
        for _ in range(server.max_workers):
            connection = HTTPConnection(*server.server_address, timeout=5)
            connections.append(connection)
            connection.request("GET", "/health")
        assert all_entered.wait(timeout=5)
        assert len(owned) == len(server._workers) == server.max_workers == 4
        assert all(worker.daemon for worker in owned)
        with pytest.raises(
            (RemoteDisconnected, ConnectionResetError, BrokenPipeError)
        ):
            http(server, "GET", "/health", timeout=1)
        assert len(owned) == server.max_workers

        worker = owned[0]
        original_join = worker.join

        def observe_join(*args, **kwargs):
            joining.set()
            return original_join(*args, **kwargs)

        monkeypatch.setattr(worker, "join", observe_join)
        server.shutdown()
        serving.join(timeout=5)
        assert not serving.is_alive()
        closer.start()
        assert joining.wait(timeout=5), "server_close did not join its handlers"
        assert not closed.is_set()
    finally:
        release.set()
        server.shutdown()
        serving.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=15)
            assert not closer.is_alive()
        else:
            server.server_close()
        for connection in connections:
            try:
                response = connection.getresponse()
                assert response.status == 200
                response.read()
            finally:
                connection.close()
    assert closed.is_set()
    assert not server._workers
    assert all(not worker.is_alive() for worker in owned)
    server.server_close()


@pytest.mark.parametrize("family", get_args(Family))
def test_all_native_families_use_factory(family):
    if family in {"qwen3_5", "qwen3_5_moe"}:
        pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    service = NativeService(
        config(checkpoint={"kind": "random", "family": family})
    )
    assert service.model_config.family == family
    assert sum(p.numel() for p in service.model.parameters()) < 1_000_000
    assert service.complete(request())["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize(
    "family, variant",
    [("smollm", variant) for variant in get_args(Variant)]
    + [("qwen3_5", "longshot"), ("gemma4", "longshot")],
)
def test_native_variants_and_fresh_cache(family, variant, monkeypatch):
    service = NativeService(
        config(
            checkpoint={"kind": "random", "family": family, "variant": variant}
        )
    )
    backbone = getattr(service.model, "backbone", service.model)
    original = backbone.forward_chunk
    caches = []

    def observe(input_ids, start_pos, cache, **kwargs):
        if start_pos == 0:
            assert not cache.keys
            caches.append(cache)
        return original(input_ids, start_pos, cache, **kwargs)

    monkeypatch.setattr(backbone, "forward_chunk", observe)
    first = service.complete(request())
    service.complete(request("different"))
    again = service.complete(request())
    assert first["choices"] == again["choices"]
    assert first["usage"] == again["usage"]
    assert len(caches) == 3
    assert len({id(cache) for cache in caches}) == 3


@pytest.mark.parametrize("sequence_len", [16, 256, 4096, 4097, 32768, 65536])
def test_random_sequence_len_admission(sequence_len):
    cfg = config(checkpoint={"kind": "random", "sequence_len": sequence_len})
    assert cfg.checkpoint["sequence_len"] == sequence_len


@pytest.mark.parametrize(
    "sequence_len", [15, 65537, True, 256.0, "32768", None]
)
def test_random_sequence_len_rejection(sequence_len):
    with pytest.raises(ValueError, match="sequence_len"):
        config(checkpoint={"kind": "random", "sequence_len": sequence_len})


def test_random_default_sequence_len():
    service = NativeService(config(checkpoint={"kind": "random"}))
    assert service.model_config.sequence_len == 256
    assert service.metadata["context"]["max_total_tokens"] == 256


def test_large_context_keeps_tiny_bgt_and_short_generation():
    service = NativeService(
        config(
            checkpoint={
                "kind": "random",
                "variant": "bgt",
                "sequence_len": 32768,
            }
        )
    )
    assert service.model_config.hidden_size == 32
    assert service.model_config.num_layers == 2
    assert sum(p.numel() for p in service.model.parameters()) < 1_000_000
    assert service.metadata["context"] == {
        "max_prompt_tokens": 32768,
        "max_total_tokens": 32768,
    }
    assert service.complete(request())["usage"]["completion_tokens"] == 3


def test_text_roles_and_greedy_options(service):
    messages = [
        {"role": role, "content": "x"}
        for role in ("system", "user", "assistant", "user")
    ]
    result = service.complete(
        request(messages=messages, temperature=0, top_p=1, stream=False, n=1)
    )
    assert result["usage"]["prompt_tokens"] == len(
        service.tokenizer.encode(messages)
    )


def test_native_failure_is_json_and_service_survives(server, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("internal checkpoint detail")

    monkeypatch.setattr(server.service.model, "generate", broken)
    for _ in range(server.max_workers + 1):
        status, result = http(server, "POST", "/v1/chat/completions", request())
        assert status == 500
        assert result["error"]["message"] == "native generation failed"
    assert http(server, "GET", "/health")[0] == 200


@pytest.mark.parametrize(
    "family, variant",
    [
        ("smollm", "tf"),
        ("smollm", "bgt"),
        ("qwen3_5", "longshot"),
        ("gemma4", "longshot"),
    ],
)
def test_cpu_does_not_probe_or_seed_cuda(monkeypatch, family, variant):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU serving must not probe or seed CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    service = NativeService(
        config(
            checkpoint={"kind": "random", "family": family, "variant": variant}
        )
    )
    assert service.complete(request())["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize(
    "family, variant",
    [("smollm", "bgt"), ("qwen3_5", "longshot"), ("gemma4", "longshot")],
)
def test_rng_weights_and_failure_isolation(monkeypatch, family, variant):
    rng = torch.random.get_rng_state().clone()
    service = NativeService(
        config(
            checkpoint={"kind": "random", "family": family, "variant": variant}
        )
    )
    assert torch.equal(torch.random.get_rng_state(), rng)
    weights = {
        name: tensor.clone()
        for name, tensor in service.model.state_dict().items()
    }
    first = service.complete(request())
    assert torch.equal(torch.random.get_rng_state(), rng)
    original = service.model.backbone.forward_chunk
    calls = 0

    def fail_after_cache_update(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            raise RuntimeError("controlled failure after cache update")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(
            service.model.backbone, "forward_chunk", fail_after_cache_update
        )
        with pytest.raises(RuntimeError, match="controlled failure"):
            service.complete(request())
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert service.complete(request())["choices"] == first["choices"]
    assert torch.equal(torch.random.get_rng_state(), rng)
    for name, tensor in service.model.state_dict().items():
        torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)


def test_native_eos_usage(service):
    messages = request()["messages"]
    ids = service.tokenizer.encode(messages)
    output = service.model.generate(torch.tensor([ids]), 1)
    service.tokenizer.eos_token_id = output[0, -1].item()
    result = service.complete(request())
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 1


def test_reject_before_native_can_truncate(service, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native generate must not be called for oversized input")

    monkeypatch.setattr(service.model, "generate", forbidden)
    with pytest.raises(RequestError, match="context limit"):
        service.complete(request("x" * 128))
    with pytest.raises(RequestError, match="context limit"):
        service.complete(request("x" * 112))


def test_exact_context_budget_is_preserved_and_overflow_rejected(
    service, monkeypatch
):
    original = service.model.generate
    inputs = []

    def observe(input_ids, max_new_tokens, **kwargs):
        inputs.append(input_ids.tolist()[0])
        return original(input_ids, max_new_tokens, **kwargs)

    monkeypatch.setattr(service.model, "generate", observe)
    framing = len(service.tokenizer.encode(request("")["messages"]))
    text = "x" * (service.model_config.sequence_len - framing - 3)
    payload = request(text)
    result = service.complete(payload)
    assert inputs == [service.tokenizer.encode(payload["messages"])]
    assert result["usage"]["total_tokens"] == service.model_config.sequence_len
    with pytest.raises(RequestError, match="context limit"):
        service.complete(request(text + "x"))
    assert len(inputs) == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        request(temperature=0.1),
        request(top_p=0.9),
        request(stream=True),
        request(stream=0),
        request(n=2),
        request(n=True),
        request(max_tokens=True),
        request(max_tokens=9),
        request(max_tokens=0),
        request(seed=1),
        request(tools=[]),
        request(tool_choice="none"),
        request(stop=["x"]),
        request(messages=[{"role": "tool", "content": "result"}]),
        request(messages=[{"role": "user", "content": []}]),
        request(
            messages=[{"role": "assistant", "content": "", "tool_calls": []}]
        ),
        request(messages=[]),
        request(messages=[None]),
    ],
)
def test_schema_errors(server, payload):
    status, response = http(server, "POST", "/v1/chat/completions", payload)
    assert status == 400
    assert response["error"]["type"] == "invalid_request_error"


def test_routes_limits_and_invalid_json(server):
    assert http(server, "GET", "/missing")[0] == 404
    assert http(server, "POST", "/missing", {})[0] == 404
    assert http(server, "DELETE", "/health")[0] == 501
    assert (
        http(server, "POST", "/v1/chat/completions", request(model="unknown"))[
            0
        ]
        == 404
    )
    assert http(server, "POST", "/v1/chat/completions", body=b"{")[0] == 400
    assert http(server, "POST", "/v1/chat/completions", body=b"\xff")[0] == 400
    assert (
        http(server, "POST", "/v1/chat/completions", body=b"x" * 65537)[0]
        == 413
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},
        {"host": "example.org"},
        {"port": True},
        {"device": "cuda"},
        {"max_body_bytes": 1048577},
        {"checkpoint": {"kind": "random", "sequence_len": 100000}},
        {"checkpoint": {"kind": "hf", "path": "/tmp/anything"}},
        {"checkpoint": {"kind": "gcw", "path": "/tmp/anything"}},
        {"unknown": True},
    ],
)
def test_config_rejects_unsafe_or_unknown_fields(overrides, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError):
        config(**overrides)


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"Content-Length": "0"}, 415),
        ({"Content-Type": "application/json"}, 411),
        ({"Content-Type": "application/json", "Content-Length": "-1"}, 411),
        (
            {
                "Content-Type": "application/json",
                "Transfer-Encoding": "chunked",
            },
            400,
        ),
    ],
)
def test_http_framing_errors(server, headers, expected):
    connection = HTTPConnection(*server.server_address, timeout=10)
    try:
        connection.putrequest("POST", "/v1/chat/completions")
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == expected
        assert json.loads(response.read())["error"]
    finally:
        connection.close()


def local_tokenizer(path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(
        WordLevel({f"t{i}": i for i in range(256)}, unk_token="t0")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="t0", eos_token="t1"
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ': ' + m['content'] + '\\n' }}{% endfor %}"
        "{% if add_generation_prompt %}assistant:{% endif %}"
    )
    tokenizer.save_pretrained(path)
    return {
        "path": str(path),
        "revision": "local-test-fixture-v1",
        "sha256": tokenizer_directory_sha256(path),
    }


@pytest.mark.parametrize("processor_only", [False, True])
@pytest.mark.parametrize(
    "family, variant",
    [("smollm", "bgt"), ("qwen3_5", "longshot"), ("gemma4", "longshot")],
)
def test_native_checkpoint_reload_and_verified_local_tokenizer(
    tmp_path, processor_only, family, variant
):
    source = NativeService(
        config(
            checkpoint={"kind": "random", "family": family, "variant": variant}
        )
    )
    checkpoint = tmp_path / "checkpoint"
    state = {"seed": 0, "step": 5}
    if processor_only:
        backbone = tmp_path / "backbone"
        save_checkpoint(
            backbone, source.model.backbone, source.model.backbone.config
        )
        state["backbone_checkpoint"] = str(backbone)
    else:
        state["backbone_checkpoint"] = str(tmp_path / "stale-unused-backbone")
    save_checkpoint(
        checkpoint,
        source.model,
        source.model_config,
        state,
        cache_processor_only=processor_only,
    )
    tokenizer = local_tokenizer(tmp_path / "tokenizer")
    cfg = config(
        checkpoint={"kind": "gcw", "path": str(checkpoint)}, tokenizer=tokenizer
    )
    rng = torch.random.get_rng_state().clone()
    loaded = NativeService(cfg)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(p.dtype == torch.float32 for p in loaded.model.parameters())
    assert loaded.model_config == source.model_config
    for name, tensor in source.model.state_dict().items():
        torch.testing.assert_close(
            loaded.model.state_dict()[name], tensor, rtol=0, atol=0
        )
    assert loaded.complete(request())["usage"]["completion_tokens"] >= 1
    assert loaded.metadata["checkpoint"]["files"]["model.safetensors"]
    recorded = loaded.metadata["checkpoint"]["recorded_backbone_checkpoint"]
    assert recorded["reference_may_be_unused"] is True
    assert bool(recorded["model_sha256"]) == processor_only
    (Path(tokenizer["path"]) / "added.txt").write_text("changed")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        NativeService(cfg)


def test_cli_starts_with_absolute_config_and_stops(tmp_path):
    path = tmp_path / "backend.json"
    path.write_text(
        json.dumps(
            {"model": "gcw-test", "port": 0, "checkpoint": {"kind": "random"}}
        )
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.Popen(
        [sys.executable, "-m", "gcw.serving", "--config", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert select.select([process.stdout], [], [], 30)[
            0
        ], "CLI did not become ready"
        ready = json.loads(process.stdout.readline())
        assert ready["status"] == "ready"
        connection = HTTPConnection(ready["host"], ready["port"], timeout=10)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["status"] == "ok"
        finally:
            connection.close()
    finally:
        process.terminate()
        process.communicate(timeout=10)


@pytest.mark.parametrize("family", ["qwen3_5", "gemma4"])
def test_longshot_pilot_config_and_native_short_generation(family, monkeypatch):
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "serving"
        / f"{family}_longshot.json"
    )
    monkeypatch.setenv("SLURM_JOB_ID", "mock-admission-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cfg = ServiceConfig.from_file(path)
    assert cfg.device == "cuda:0"
    assert cfg.seed == 0 and cfg.max_output_tokens == 128
    assert cfg.checkpoint == {
        "kind": "random",
        "family": family,
        "variant": "longshot",
        "sequence_len": 32768,
    }
    # Only the CPU copy is constructed; admission above never touches CUDA.
    cfg.device = "cpu"
    service = NativeService(cfg)
    from gcw.models.cache_processors.longshot import GlobalCacheProcessor

    assert isinstance(service.model.cache_processor, GlobalCacheProcessor)
    assert service.model_config.hidden_size == 32
    assert service.model_config.num_layers == 2
    assert all(
        p.dtype == torch.float32 and p.device.type == "cpu"
        for p in service.model.parameters()
    )
    payload = request(model=cfg.model)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = service.complete(payload)
    assert any(
        "padding/truncating the selected cache history" in str(w.message)
        for w in observed
    )
    assert result["usage"]["prompt_tokens"] == len(
        service.tokenizer.encode(payload["messages"])
    )
    assert result["usage"]["completion_tokens"] == 3
    assert result["gcw"]["checkpoint"] == {
        "kind": "random",
        "diagnostic_only": True,
        "pretrained_weights": False,
    }
    assert result["gcw"]["cache"]["selected_history"][
        "distinct_from_prompt_overflow"
    ]
    assert result["gcw"]["context"]["max_total_tokens"] == 32768
    assert result["choices"] == service.complete(payload)["choices"]
    with pytest.raises(RequestError, match="context limit"):
        service.complete(request("x" * 32768, model=cfg.model))


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
@pytest.mark.parametrize(
    "job, visible",
    [(None, "0"), ("1", None), ("1", "-1"), (" ", "0"), ("1", " ")],
)
def test_cuda_gate_does_not_probe_gpu(monkeypatch, device, job, visible):
    for key, value in (
        ("SLURM_JOB_ID", job),
        ("CUDA_VISIBLE_DEVICES", visible),
    ):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    def forbidden(*args, **kwargs):
        pytest.fail("CUDA gate must reject before touching GPU")

    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    with pytest.raises(ValueError, match="Slurm GPU allocation"):
        config(device=device)
    cfg = config()
    cfg.device = device
    with pytest.raises(ValueError, match="Slurm GPU allocation"):
        NativeService(cfg)


@pytest.mark.parametrize(
    "device, resolved", [("cuda", "cuda:0"), ("cuda:1", "cuda:1")]
)
def test_allocated_cuda_selects_explicit_current_device_mock_only(
    monkeypatch, device, resolved
):
    monkeypatch.setenv("SLURM_JOB_ID", "mock-only")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    selected, moves = [], []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda value: selected.append(str(value))
    )

    def move(model, *, device, dtype):
        moves.append((str(device), dtype))
        return model

    monkeypatch.setattr(torch.nn.Module, "to", move)
    service = NativeService(config(device=device))
    assert selected == [resolved]
    assert moves == [(resolved, torch.float32)]
    assert service.metadata["resolved_device"] == resolved


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_rejected(server, tmp_path, constant):
    body = (
        '{"model":"gcw-test","messages":[{"role":"user","content":"hi"}],"temperature":'
        + constant
        + "}"
    ).encode()
    assert http(server, "POST", "/v1/chat/completions", body=body)[0] == 400
    path = tmp_path / "nonfinite.json"
    path.write_text(
        '{"model":"gcw-test","checkpoint":{"kind":"random"},"seed":'
        + constant
        + "}"
    )
    with pytest.raises(ValueError, match="non-finite JSON number"):
        ServiceConfig.from_file(path)


def test_config_path_must_be_absolute():
    with pytest.raises(ValueError, match="absolute"):
        ServiceConfig.from_file("relative.json")
