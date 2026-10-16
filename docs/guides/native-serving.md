# Native generation service

`python -m gcw.serving` runs one native GCW model behind a loopback-only HTTP
adapter with concurrent metadata requests and serialized native generation.
It uses `build_model` or `load_checkpoint`, then native `generate`: no HF
generation fallback, weight conversion, training, downloads, or paid APIs.
This is a local research adapter, not a production server.

## Diagnostic fixture

Use an existing GCW CPU environment (or `PYTHONPATH=src`). Save an absolute JSON
config, then run `python -m gcw.serving --config /absolute/backend.json`:

```json
{
  "model": "gcw-cpu-diagnostic",
  "checkpoint": {"kind": "random", "family": "smollm", "variant": "bgt"},
  "device": "cpu",
  "seed": 0,
  "host": "127.0.0.1",
  "port": 8000,
  "max_body_bytes": 65536,
  "max_output_tokens": 64
}
```

Startup prints JSON `status: ready` after loading and binding. Port zero chooses
an ephemeral port, reported there; Ctrl-C closes the listener and joins owned
request threads, waiting for in-flight work. Nothing installs, schedules, or
restarts a service.

Random fixtures have 256 byte tokens, two layers, hidden size 32, intermediate
size 64, four attention heads, two KV heads, and head dimension eight. Processors
are similarly tiny. Framing is `role: content\n` followed by `assistant:`; UTF-8
bytes encode the prompt, and decoding replaces invalid UTF-8. Metadata labels
these fixtures `diagnostic_only` and `chat_capability: false`: **plumbing, not
language capability**. Larger `sequence_len` values admit larger prompts without
changing model dimensions or proving quality, memory fit, or full-length speed.

Families use native factory/capability admission: `smollm`, `llama`, `qwen3`,
`qwen3_5`, `qwen3_5_moe`, `gemma4`; variants: `tf`, `bt`, `bgt`, `icarus`,
`longshot`. Unsupported compositions fail at startup. Qwen3.5 fixtures have one
linear and one full-attention layer; MoE has four experts, two active per token.

## Native checkpoint and tokenizer

Replace `checkpoint` and add `tokenizer`:

```json
{
  "model": "my-gcw-checkpoint",
  "checkpoint": {"kind": "gcw", "path": "/absolute/checkpoint"},
  "tokenizer": {
    "path": "/absolute/tokenizer-only-directory",
    "revision": "upstream-commit-or-local-export-label",
    "sha256": "REPLACE_WITH_64_CHARACTER_LOWERCASE_DIRECTORY_SHA256"
  }
}
```

The native loader reads `config.json`, `model.safetensors`, and `train_state.json`
with normal admission/shape checks; weights use float32, including on CPU.
Processor-only checkpoints use `train_state.backbone_checkpoint`. Prefer an
absolute local directory: relative references retain native current-working-
directory semantics. Complete checkpoints need not use their recorded backbone
reference. Metadata calls it `recorded_backbone_checkpoint`, marks
`reference_may_be_unused: true`, and records its weight hash when available
(otherwise `null`), without claiming an unused reference was loaded.

Checkpoints require a local HF tokenizer with a chat template and Transformers
(the existing `convert`/`eval` extra). Loading uses `local_files_only=True` and
`trust_remote_code=False`; byte fallback is forbidden. Token IDs must fit the
checkpoint vocabulary. Chat rendering uses the saved template,
`add_generation_prompt=True`, and `enable_thinking=False`; no role/template is
silently dropped. A template does not prove model chat capability.

Compute the required directory hash in the GCW environment:

```bash
python -c 'from pathlib import Path; from gcw.serving import tokenizer_directory_sha256; print(tokenizer_directory_sha256(Path("/absolute/tokenizer-only-directory")))'
```

This hashes UTF-8 JSON of sorted `[relative_posix_path, file_sha256]` pairs for
every file recursively (`ensure_ascii=True`, separators `(',', ':')`). Symlink
targets contribute file contents. Keep weights outside this tokenizer-only
directory; any file change fails startup. Keep all artifacts immutable during
loading/serving. The revision label is required provenance, not verified upstream
ancestry; the hash anchors bytes, not an authenticated training-resume receipt.
It is neither an HF commit hash nor the evaluation harness's tokenizer hash.

## Config contract

Unknown/duplicate keys fail closed, including nested objects and HTTP requests.
Config/tokenizer paths are absolute. `model` and `checkpoint` are required;
`tokenizer` is required exactly for `kind: gcw`.

| Field | Default / constraint |
|---|---|
| `model` | Nonempty served alias, not an upstream model identifier |
| `device` | `cpu`; explicit `cuda` / `cuda:N` requires a Slurm GPU allocation |
| `checkpoint.kind` | `random` or `gcw` |
| Random `family`, `variant`, `sequence_len` | `smollm`, `tf`, `256`; length integer 16–65536 |
| GCW `path` | Local checkpoint directory; no random-only fields |
| `tokenizer.path`, `revision`, `sha256` | Local directory, nonempty label, lowercase directory SHA256 |
| `seed` | `0`; integer 0–2³²−1, used at initialization and reset per request |
| `host` | Only `127.0.0.1` |
| `port` | `8000`; integer 0–65535 |
| `max_body_bytes` | `65536`; integer 1–1048576, independent of token budget |
| `max_output_tokens` | `64`; integer 1–4096 |

### Gated CUDA diagnostics

`configs/serving/{qwen3_5,gemma4}_longshot.json` selects tiny native Longshot,
random seed 0, float32, `cuda:0`, 32768 logical tokens, a 128-token output cap,
and a one-MiB body cap. Aliases include `random-seed0-diagnostic`; no pretrained
weights load. Compare with the same rendered CL smoke prompt and explicit
`max_tokens: 128`; default request budget remains 32.

Launch only inside an explicitly owned Slurm allocation. `SLURM_JOB_ID` and
nonempty `CUDA_VISIBLE_DEVICES` (not `-1`) must exist before startup. Admission
checks these without a GPU probe, rechecks before construction, then checks CUDA
availability and selects the configured current device. `cuda` means visible
ordinal zero; `cuda:N` is allocation-remapped, not a physical GPU index. Health
reports `resolved_device`. The service never obtains allocations.

Run frozen source with `PYTHONPATH=/absolute/frozen-gcw/src`,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1`.
The minimal import closure is Torch, safetensors, PyYAML, and Transformers with
its normal dependencies. Qwen3.5 uses Transformers' native Torch gated-delta
fallback; `fla` and `causal_conv1d` are optional. Omit unrelated audio/vision
packages to avoid import side effects. Do not overlay the full GCW GPU extra
(Torch <2.11) on the candidate RunPod Torch 2.13 runtime; frozen `PYTHONPATH`
source does not require installing GCW there. CPU tests do not qualify that
stack or any CUDA hardware.

**Preserve native Longshot warnings.** These fixtures' linear time encoder
expects eight selected positions and warns about `padding/truncating the selected
cache history` otherwise. This is internal history fitting, not service prompt
truncation. Health reports it separately. Qwen3.5 recurrent linear-attention state
remains native (`linear_state_cache: false`); Gemma4 keeps native sliding/full
attention with uniform tiny stored-cache dimensions. No custom algorithm or
warning filter is installed.

## HTTP contract

- `GET /health`: `{"status":"ok","gcw":{...}}` after loading.
- `GET /v1/models`: OpenAI-style list with one served model.
- `POST /v1/chat/completions`: one non-streaming greedy completion.

```json
{
  "model": "gcw-cpu-diagnostic",
  "messages": [{"role": "user", "content": "Hello"}],
  "temperature": 0,
  "top_p": 1,
  "max_tokens": 8,
  "stream": false,
  "n": 1
}
```

`messages` must be nonempty, with text `content` and only `system`, `user`, or
`assistant` roles. Defaults: temperature 0, top_p 1, n 1, stream false, max_tokens
`min(32, max_output_tokens)`. Other sampling values, tools/calls/roles, stop
strings, request seeds, logprobs, penalties, multimodal input, and unknown fields
are rejected. `max_completion_tokens` is not an alias. Templates may restrict
role ordering. This text contract does **not** support the Pi tool bridge.

Before native generation, rendered prompt plus requested output must fit
`ModelConfig.sequence_len`; the service never truncates prompts. Each request
gets fresh native cache state, inference mode, and the configured RNG seed.
Only the generated suffix is decoded; local-tokenizer EOS ends generation.
No batching, persistent sessions, cross-request cache reuse, or weight updates.

Responses contain standard completion fields, `finish_reason` (`stop` for EOS,
otherwise `length`), and token usage for the rendered prompt and generated suffix
(including EOS). Completions, model entries, and health include `gcw` provenance:
checkpoint hashes/config, tokenizer identity, seed, dtype, context, capabilities.
Physical cache tokens/bytes remain `null` (unmeasured); logical limits are not a
measured memory bound or evidence of checkpoint quality.

Bodies require one `Content-Length` and `Content-Type: application/json`.
Chunked encoding is unsupported. Standard-library header limits and ten-second
socket-read timeouts apply; connections close after each response. UTF-8 JSON
must contain no duplicate members or non-finite numbers (`NaN`, `±Infinity`).
Errors use `{"error":{"message":"...","type":"invalid_request_error",
"param":null,"code":"400"}}`: invalid request 400, unknown route/model 404,
body too large 413, content type 415, missing length 411, read timeout 408.
Unexpected native failures return 500 (`type: server_error`); unsupported HTTP
methods return JSON 501. No authentication, TLS, CORS, streaming, cancellation,
or execution deadlines. Use a trusted local machine, never an external proxy.

The server owns at most four active daemon request handlers. Health and model
discovery can complete while generation runs; the existing model lock still
serializes tokenization, RNG access, native generation, and decoding. There is no
parallel native generation, batching, or inference retry. When all four handlers
are occupied (including requests waiting for the model lock or reading bodies),
new connections close without an HTTP response. Metadata availability is not
guaranteed under saturation. This small bound assumes trusted private loopback
clients, not a public-service security boundary or parallel-generation
qualification. Socket-read timeouts do not cancel native work. Server cleanup
joins owned handlers; embedding callers must stop `serve_forever` with `shutdown`
before calling `server_close` from another thread.

## CPU validation and limits

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_serving.py
```

Use the existing `convert`/`eval` extra for the full suite. Generic optional-family
checks may skip missing imports; explicit Qwen3.5 Longshot regressions require
the Transformers fallback. Tests cover tiny native families/variants, HTTP/JSON
admission and lifecycle, metadata during held generation, serialized generation,
bounded handlers and joined cleanup, exact-fit/overflow context gates, fresh
caches and unchanged native call arguments over HTTP, RNG and weight
preservation/recovery after failed generation, CPU copies of CUDA configs
with native warnings, mocked device selection and allocation rejection before
CUDA access, and full/processor-only reloads using a locally created tokenizer.
They download nothing. Full-length CPU performance, real trained checkpoint/
upstream tokenizer compatibility, CUDA fit, and large-model parity remain
**unqualified**; synthetic reloads establish none of these.
