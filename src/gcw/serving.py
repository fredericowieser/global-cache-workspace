"""Local, serial HTTP adapter for native GCW greedy generation."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import torch

from gcw.checkpoint import load_checkpoint
from gcw.config import CacheProcessorConfig, ModelConfig, to_dict
from gcw.models import build_model


class RequestError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _object(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"unsupported {name} fields: {sorted(unknown)}")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _absolute_directory(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"{name} must be an absolute directory path")
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_directory_sha256(path: Path) -> str:
    """Hash a canonical sorted [relative POSIX path, file SHA256] manifest."""
    manifest = [
        [file.relative_to(path).as_posix(), _file_sha256(file)]
        for file in sorted(path.rglob("*"))
        if file.is_file()
    ]
    if not manifest:
        raise ValueError("tokenizer directory is empty")
    return hashlib.sha256(
        json.dumps(manifest, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


class ServiceConfig:
    def __init__(self, payload: Any):
        cfg = _object(
            payload,
            {
                "model",
                "device",
                "checkpoint",
                "tokenizer",
                "host",
                "port",
                "seed",
                "max_body_bytes",
                "max_output_tokens",
            },
            "config",
        )
        self.model = cfg.get("model")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty served alias")
        self.host = cfg.get("host", "127.0.0.1")
        if self.host != "127.0.0.1":
            raise ValueError(
                "only the IPv4 loopback host 127.0.0.1 is supported"
            )
        self.port = _integer(cfg.get("port", 8000), "port", 0, 65535)
        self.seed = _integer(cfg.get("seed", 0), "seed", 0, 2**32 - 1)
        self.max_body_bytes = _integer(
            cfg.get("max_body_bytes", 65536), "max_body_bytes", 1, 1048576
        )
        self.max_output_tokens = _integer(
            cfg.get("max_output_tokens", 64), "max_output_tokens", 1, 4096
        )
        self.device = cfg.get("device", "cpu")
        if not isinstance(self.device, str) or (
            self.device != "cpu"
            and self.device != "cuda"
            and not (
                self.device.startswith("cuda:") and self.device[5:].isdigit()
            )
        ):
            raise ValueError("device must be cpu, cuda, or cuda:N")
        self.require_allocation()
        self.checkpoint = _object(
            cfg.get("checkpoint"),
            {"kind", "path", "family", "variant", "sequence_len"},
            "checkpoint",
        )
        kind = self.checkpoint.get("kind")
        if kind == "random":
            _object(
                self.checkpoint,
                {"kind", "family", "variant", "sequence_len"},
                "random checkpoint",
            )
            _integer(
                self.checkpoint.get("sequence_len", 256),
                "sequence_len",
                16,
                65536,
            )
            if "tokenizer" in cfg:
                raise ValueError(
                    "random fixtures always use diagnostic UTF-8 bytes"
                )
            self.tokenizer = None
        elif kind == "gcw":
            _object(self.checkpoint, {"kind", "path"}, "gcw checkpoint")
            self.tokenizer = _object(
                cfg.get("tokenizer"),
                {"path", "revision", "sha256"},
                "tokenizer",
            )
            if set(self.tokenizer) != {"path", "revision", "sha256"}:
                raise ValueError(
                    "tokenizer requires path, revision, and sha256"
                )
            if (
                not isinstance(self.tokenizer["revision"], str)
                or not self.tokenizer["revision"].strip()
            ):
                raise ValueError(
                    "tokenizer.revision must be a nonempty provenance label"
                )
            digest = self.tokenizer["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError("tokenizer.sha256 must be a lowercase SHA256")
        else:
            raise ValueError("checkpoint.kind must be random or gcw")

    def require_allocation(self) -> None:
        if self.device != "cpu" and not (
            os.environ.get("SLURM_JOB_ID", "").strip()
            and os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            not in ("", "-1")
        ):
            raise ValueError("CUDA requires an explicit Slurm GPU allocation")

    @classmethod
    def from_file(cls, path: str | Path) -> ServiceConfig:
        path = Path(path)
        if not path.is_absolute():
            raise ValueError("--config must be an absolute JSON path")
        return cls(
            json.loads(
                path.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
        )


class ChatTokenizer:
    """An explicitly diagnostic byte fixture or a verified local HF tokenizer."""

    def __init__(self, cfg: ServiceConfig, vocab_size: int):
        self.enc = None
        self.eos_token_id = None
        if cfg.tokenizer is None:
            if vocab_size != 256:
                raise ValueError("byte fixture requires a 256-token vocabulary")
            self.metadata = {
                "kind": "diagnostic_utf8_bytes_v1",
                "chat_capability": False,
            }
            return
        spec = cfg.tokenizer
        path = _absolute_directory(spec["path"], "tokenizer.path")
        digest = tokenizer_directory_sha256(path)
        if digest != spec["sha256"]:
            raise ValueError("tokenizer directory SHA256 mismatch")
        from transformers import AutoTokenizer

        self.enc = AutoTokenizer.from_pretrained(
            str(path),
            revision=spec["revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
        if not self.enc.chat_template:
            raise ValueError("local tokenizer must provide a chat template")
        ids = list(self.enc.get_vocab().values())
        if not ids or min(ids) < 0 or max(ids) >= vocab_size:
            raise ValueError(
                "tokenizer token IDs do not fit checkpoint vocabulary"
            )
        self.eos_token_id = self.enc.eos_token_id
        if tokenizer_directory_sha256(path) != digest:
            raise ValueError("tokenizer files changed during loading")
        self.metadata = {
            "kind": "local_hf",
            "path": str(path),
            "revision": spec["revision"],
            "sha256": digest,
            "chat_capability": "checkpoint_dependent_unverified",
        }

    def encode(self, messages: list[dict[str, str]]) -> list[int]:
        if self.enc is None:
            text = "".join(
                f"{msg['role']}: {msg['content']}\n" for msg in messages
            )
            return list((text + "assistant:").encode("utf-8"))
        encoded = self.enc.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(encoded, dict) or hasattr(encoded, "keys"):
            encoded = encoded["input_ids"]
        return encoded

    def decode(self, ids: list[int]) -> str:
        if self.enc is None:
            return bytes(ids).decode("utf-8", errors="replace")
        return self.enc.decode(ids, skip_special_tokens=True)


class NativeService:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.lock = threading.Lock()
        config.require_allocation()
        self.device = torch.device(
            "cuda:0" if config.device == "cuda" else config.device
        )
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available in this allocation")
            torch.cuda.set_device(self.device)
        checkpoint = config.checkpoint
        provenance: dict[str, Any] = {"kind": checkpoint["kind"]}
        if checkpoint["kind"] == "random":
            family = checkpoint.get("family", "smollm")
            variant = checkpoint.get("variant", "tf")
            kwargs: dict[str, Any] = {}
            if family in {"qwen3_5", "qwen3_5_moe"}:
                kwargs["layer_types"] = ["linear_attention", "full_attention"]
            if family == "qwen3_5_moe":
                kwargs.update(num_experts=4, num_experts_per_tok=2)
            model_config = ModelConfig(
                family=family,
                variant=variant,
                sequence_len=checkpoint.get("sequence_len", 256),
                vocab_size=256,
                hidden_size=32,
                intermediate_size=64,
                num_layers=2,
                num_heads=4,
                num_kv_heads=2,
                head_dim=8,
                cache_processor=(
                    None
                    if variant == "tf"
                    else CacheProcessorConfig(
                        hidden_size=32,
                        intermediate_size=64,
                        num_heads=4,
                        num_kv_heads=2,
                        process_every=8,
                        topk=4,
                        topk_heads=2,
                        encoder_seq_len=8,
                    )
                ),
                **kwargs,
            )
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.seed)
                self.model = build_model(model_config)
                self.model.init_weights()
            self.model.to(device=self.device, dtype=torch.float32)
            provenance.update(diagnostic_only=True, pretrained_weights=False)
        else:
            path = _absolute_directory(
                checkpoint.get("path"), "checkpoint.path"
            )
            provenance.update(
                path=str(path),
                files={
                    name: _file_sha256(path / name)
                    for name in (
                        "config.json",
                        "model.safetensors",
                        "train_state.json",
                    )
                },
            )
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.seed)
                self.model, model_config, state = load_checkpoint(
                    path, device="cpu", dtype="float32"
                )
            self.model.to(device=self.device, dtype=torch.float32)
            backbone = state.get("backbone_checkpoint")
            if isinstance(backbone, str) and backbone:
                backbone_path = Path(backbone).resolve()
                weights = backbone_path / "model.safetensors"
                provenance["recorded_backbone_checkpoint"] = {
                    "path": str(backbone_path),
                    "model_sha256": (
                        _file_sha256(weights) if weights.is_file() else None
                    ),
                    "reference_may_be_unused": True,
                }
        self.model.eval()
        self.tokenizer = ChatTokenizer(config, model_config.vocab_size)
        self.model_config = model_config
        self.metadata = {
            "backend": "gcw_native",
            "model": config.model,
            "device": config.device,
            "resolved_device": str(self.device),
            "dtype": "float32",
            "seed": config.seed,
            "checkpoint": provenance,
            "tokenizer": self.tokenizer.metadata,
            "model_config": to_dict(model_config),
            "context": {
                "max_prompt_tokens": model_config.sequence_len,
                "max_total_tokens": model_config.sequence_len,
            },
            "cache": {
                "scope": "fresh_per_request",
                "cross_request_reuse": False,
                "logical_prompt_limit": model_config.sequence_len,
                "physical_cache_tokens": None,
                "physical_cache_bytes": None,
            },
            "capabilities": {
                "greedy": True,
                "streaming": False,
                "tools": False,
                "sampling": False,
                "network_downloads": False,
                "max_output_tokens": config.max_output_tokens,
            },
        }
        if model_config.variant == "longshot":
            processor = model_config.cache_processor
            self.metadata["cache"]["selected_history"] = {
                "encoder_method": processor.encoder_method,
                "encoder_seq_len": processor.encoder_seq_len,
                "policy": (
                    "native_linear_padding_or_truncation"
                    if processor.encoder_method == "linear"
                    else "native_query_attention"
                ),
                "distinct_from_prompt_overflow": True,
            }

    def complete(self, payload: Any) -> dict[str, Any]:
        try:
            request = _object(
                payload,
                {
                    "model",
                    "messages",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "stream",
                    "n",
                },
                "request",
            )
            if not isinstance(request.get("model"), str):
                raise ValueError("model must be a served alias string")
            if request["model"] != self.config.model:
                raise RequestError("unknown model alias", 404)
            _integer(request.get("n", 1), "n", 1, 1)
            for field, expected in (("temperature", 0), ("top_p", 1)):
                value = request.get(field, expected)
                if type(value) not in (int, float) or value != expected:
                    raise ValueError(f"only {field}={expected} is supported")
            if request.get("stream", False) is not False:
                raise ValueError("streaming is not supported")
            max_tokens = _integer(
                request.get(
                    "max_tokens", min(32, self.config.max_output_tokens)
                ),
                "max_tokens",
                1,
                self.config.max_output_tokens,
            )
            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError("messages must be a nonempty list")
            for message in messages:
                _object(message, {"role", "content"}, "message")
                if message.get("role") not in ("system", "user", "assistant"):
                    raise ValueError(
                        "only system, user, and assistant roles are supported"
                    )
                if not isinstance(message.get("content"), str):
                    raise ValueError("message content must be text")
            with self.lock, torch.inference_mode():
                try:
                    ids = self.tokenizer.encode(messages)
                except Exception as exc:
                    raise ValueError(
                        f"chat template rejected messages: {exc}"
                    ) from exc
                if (
                    not isinstance(ids, list)
                    or not ids
                    or any(
                        type(token) is not int
                        or not 0 <= token < self.model_config.vocab_size
                        for token in ids
                    )
                ):
                    raise ValueError(
                        "tokenizer must produce nonempty valid token IDs"
                    )
                if len(ids) + max_tokens > self.model_config.sequence_len:
                    raise ValueError(
                        "prompt plus max_tokens exceeds the context limit; truncation is disabled"
                    )
                tensor = torch.tensor(
                    [ids], dtype=torch.long, device=self.device
                )
                devices = [] if self.device.type == "cpu" else [self.device]
                with torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(self.config.seed)
                    if devices:
                        with torch.cuda.device(tensor.device):
                            torch.cuda.manual_seed(self.config.seed)
                    output = self.model.generate(
                        tensor,
                        max_tokens,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated = output[0, len(ids) :].tolist()
                text = self.tokenizer.decode(generated)
            finished = bool(
                generated and generated[-1] == self.tokenizer.eos_token_id
            )
            return {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.config.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop" if finished else "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(generated),
                    "total_tokens": len(ids) + len(generated),
                },
                "gcw": self.metadata,
            }
        except RequestError:
            raise
        except (ValueError, TypeError) as exc:
            raise RequestError(str(exc)) from exc


class ServiceHTTPServer(HTTPServer):
    allow_reuse_address = True

    def __init__(self, service: NativeService):
        self.service = service
        super().__init__(
            (service.config.host, service.config.port), ServiceHandler
        )


class ServiceHandler(BaseHTTPRequestHandler):
    server: ServiceHTTPServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def send_error(
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        self._json(
            code,
            {
                "error": {
                    "message": message or "HTTP error",
                    "type": (
                        "server_error"
                        if code >= 500
                        else "invalid_request_error"
                    ),
                    "param": None,
                    "code": str(code),
                }
            },
        )

    def do_GET(self) -> None:
        service = self.server.service
        if not ipaddress.ip_address(self.client_address[0]).is_loopback:
            self.send_error(403, "loopback clients only")
        elif self.path == "/health":
            self._json(200, {"status": "ok", "gcw": service.metadata})
        elif self.path == "/v1/models":
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": service.config.model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "gcw",
                            "gcw": service.metadata,
                        }
                    ],
                },
            )
        else:
            self.send_error(404, "unknown route")

    def do_POST(self) -> None:
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                raise RequestError("loopback clients only", 403)
            if self.path != "/v1/chat/completions":
                raise RequestError("unknown route", 404)
            lengths = self.headers.get_all("Content-Length", [])
            if self.headers.get("Transfer-Encoding"):
                raise RequestError("Transfer-Encoding is not supported")
            if (
                len(lengths) != 1
                or not lengths[0].isascii()
                or not lengths[0].isdigit()
            ):
                raise RequestError("one valid Content-Length is required", 411)
            length = int(lengths[0])
            if length > self.server.service.config.max_body_bytes:
                raise RequestError("request body exceeds max_body_bytes", 413)
            if self.headers.get_content_type() != "application/json":
                raise RequestError("Content-Type must be application/json", 415)
            body = self.rfile.read(length)
            if len(body) != length:
                raise RequestError("incomplete request body")
            payload = json.loads(
                body.decode("utf-8"),
                object_pairs_hook=_unique_object,
                parse_constant=_reject_constant,
            )
            self._json(200, self.server.service.complete(payload))
        except RequestError as exc:
            self.send_error(exc.status, str(exc))
        except (UnicodeDecodeError, ValueError, RecursionError):
            self.send_error(400, "invalid JSON request")
        except TimeoutError:
            self.send_error(408, "request timed out")
        except Exception:
            self.send_error(500, "native generation failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="absolute backend JSON path"
    )
    args = parser.parse_args()
    service = NativeService(ServiceConfig.from_file(args.config))
    with ServiceHTTPServer(service) as server:
        print(
            json.dumps(
                {
                    "status": "ready",
                    "host": server.server_address[0],
                    "port": server.server_address[1],
                    "gcw": service.metadata,
                }
            ),
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
