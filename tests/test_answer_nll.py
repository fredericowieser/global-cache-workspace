from dataclasses import replace
import json
import subprocess
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from gcw.checkpoint import save_checkpoint
from gcw.config import (
    CacheProcessorConfig,
    DataConfig,
    DiffusionProcessorConfig,
    LoRAConfig,
    ModelConfig,
    to_dict,
)
from gcw.data.sft import sft_batch
from gcw.data.tokenizers import IntTokenizer
from gcw.data.training import SFTCursor
from gcw.evaluation.answer_nll import (
    AnswerNLLData,
    AnswerNLLDataOptions,
    answer_nll,
    evaluate_answer_nll,
    load_answer_nll_rows,
    paired_answer_nll,
    tokenizer_identity_sha256,
    _validate_loaded_config,
)
from gcw.models import build_model
from gcw.provenance import (
    canonical_json_sha256,
    file_sha256,
    tensor_state_sha256,
)
from gcw.training.checkpoint import ResumeState, save_training_checkpoint
from gcw.training.checkpoint.types import TrainingCheckpointError


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dataset(path, rows=None):
    rows = (
        rows
        if rows is not None
        else [
            {"prompt": "p", "answer": "a"},
            {"prompt": "q", "answer": "longer"},
        ]
    )
    path.write_text("\n" + "\n".join(json.dumps(row) for row in rows) + "\n\n")
    return AnswerNLLData(path, file_sha256(path), len(rows))


def native_checkpoint(
    path,
    *,
    processor=False,
    lora=False,
    processor_only=False,
    dtype=torch.float32,
    diffusion=None,
    sequence_len=32,
):
    config = ModelConfig(
        sequence_len=sequence_len,
        vocab_size=256,
        hidden_size=16,
        intermediate_size=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        variant="bgt" if processor else "tf",
        diffusion_processor=diffusion,
        cache_processor=(
            CacheProcessorConfig(
                process_every=2,
                topk=2,
                out_proj_init="zero" if diffusion is not None else "normal",
                hidden_size=16,
                intermediate_size=32,
                num_layers=1,
                num_heads=2,
                num_kv_heads=1,
            )
            if processor
            else None
        ),
        lora=LoRAConfig(rank=2) if lora else None,
    )
    torch.manual_seed(17)
    model = build_model(config).to(dtype=dtype)
    model.init_weights()
    # Perform a real native SFT update before saving (including the adapters).
    x, y, mask = sft_batch(
        [{"prompt": "p", "answer": "a"}],
        IntTokenizer(),
        32,
        return_attention_mask=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model(x, y, attention_mask=mask).backward()
    optimizer.step()
    metadata = {"dtype": str(dtype).removeprefix("torch.")}
    if processor_only:
        backbone = path.with_name(path.name + "-backbone")
        save_checkpoint(backbone, model.backbone, model.backbone.config)
        metadata["backbone_checkpoint"] = str(backbone)
    resume = ResumeState(1, 1.0, SFTCursor(), 1, True, metadata)
    fingerprint = {
        "model": to_dict(config),
        "mode": "sft",
        "data": to_dict(DataConfig(kind="sft", tokenizer="int")),
        "batch": {"sequence_len": sequence_len},
        "precision": "fp32",
    }
    save_training_checkpoint(
        path,
        model,
        config,
        optimizer,
        resume,
        fingerprint,
        cache_processor_only=processor_only,
    )
    model.eval()
    return model


def run(path, data, out, **kwargs):
    return evaluate_answer_nll(
        checkpoint=path,
        expected_manifest_sha256=file_sha256(path / "manifest.json"),
        data=data,
        out=out,
        **kwargs,
    )


def test_solver_override_records_actual_calls_and_allows_paired_depths(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    native_checkpoint(
        checkpoint,
        processor=True,
        diffusion=DiffusionProcessorConfig(
            training_method="scheduled",
            schedule_horizon=50,
            training_solver_steps=2,
            noise_scale=0.1,
        ),
    )
    data = dataset(tmp_path / "data.jsonl")
    receipts = []
    for depth in (1, 4):
        out = tmp_path / f"k{depth}.json"
        receipt = run(checkpoint, data, out, solver_steps=depth)
        assert receipt["scheduled_field_evaluations"] > 0
        assert receipt["scheduled_field_evaluations"] == sum(
            row["scheduled_field_evaluations"] for row in receipt["rows"]
        )
        for row in receipt["rows"]:
            assert row["scheduled_field_evaluations"] == depth * (
                (row["length"] - 1) // 2
            )
        receipts.append(
            load_answer_nll_rows(out, expected_sha256=file_sha256(out))
        )
    paired_answer_nll(*receipts, expected_count=data.count)


def test_stochastic_nll_and_checkpoint_output_are_rejected(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    native_checkpoint(
        checkpoint,
        processor=True,
        diffusion=DiffusionProcessorConfig(
            inference_mode="stochastic",
        ),
    )
    data = dataset(tmp_path / "data.jsonl")
    with pytest.raises(ValueError, match="outside the checkpoint"):
        run(checkpoint, data, checkpoint / "evaluation.json")
    with pytest.raises(ValueError, match="deterministic diffusion"):
        run(checkpoint, data, tmp_path / "evaluation.json")
    assert not (checkpoint / "evaluation.json").exists()
    assert not (tmp_path / "evaluation.json").exists()


@pytest.mark.parametrize("processor", [False, True])
def test_native_projection_is_bounded_before_logits_allocation(
    tmp_path, monkeypatch, processor
):
    from gcw.models.backbones.transformer.tf import TF

    checkpoint = tmp_path / "checkpoint"
    native_checkpoint(checkpoint, processor=processor, sequence_len=128)
    data = dataset(
        tmp_path / "data.jsonl", [{"prompt": "p", "answer": "answer" * 20}]
    )
    import gcw.evaluation.answer_nll as evaluation

    original = TF.logits_from_hidden
    capture = evaluation.capture_file
    captured_weights = []
    projected = []

    def capture_weights(*args, **kwargs):
        blob = capture(*args, **kwargs)
        if kwargs.get("role") == "model.safetensors":
            captured_weights.append(weakref.ref(blob))
        return blob

    def bounded(self, hidden):
        assert captured_weights and all(
            ref() is None for ref in captured_weights
        )
        positions = hidden.numel() // hidden.shape[-1]
        projected.append(positions)
        assert positions <= 32
        return original(self, hidden)

    monkeypatch.setattr(evaluation, "capture_file", capture_weights)
    monkeypatch.setattr(TF, "logits_from_hidden", bounded)
    receipt = run(checkpoint, data, tmp_path / "score.json")
    assert len(projected) > 1
    assert receipt["reference_loss_execution"]["max_projected_positions"] == 32
    assert receipt["result"]["targets"] > 32


def test_already_shifted_causal_answer_masks():
    x, y, mask = sft_batch(
        [{"prompt": "ab", "answer": "cd"}],
        IntTokenizer(),
        32,
        return_attention_mask=True,
    )
    assert x.tolist() == [[97, 98, 99, 100, 0]]
    assert y.tolist() == [[-100, 99, 100, 0, -100]]
    assert mask.all()
    logits = torch.zeros(1, 5, 256)
    for pos, label in ((1, 99), (2, 100), (3, 0)):
        logits[0, pos, label] = 5
    score = answer_nll(logits, y)
    assert score.targets == 3
    assert score.nll_sum == pytest.approx(
        float(F.cross_entropy(logits[0, 1:4], y[0, 1:4], reduction="sum"))
    )
    assert score.nll_sum < float(
        F.cross_entropy(logits[0, :3], y[0, 1:4], reduction="sum")
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16]
)
def test_chunked_full_vocab_fp32_noncontiguous(dtype, monkeypatch):
    logits = torch.randn(
        2, 71, 103, generator=torch.Generator().manual_seed(1), dtype=dtype
    ).transpose(1, 2)
    targets = torch.randint(
        71, (2, 103), generator=torch.Generator().manual_seed(2)
    )
    targets[:, ::7] = -100
    expected = F.cross_entropy(
        logits[targets != -100].float(),
        targets[targets != -100],
        reduction="sum",
    )
    original = F.cross_entropy
    calls = []

    def checked(values, labels, **kwargs):
        assert values.dtype == torch.float32
        assert 0 < len(values) <= 32 and values.shape[-1] == 71
        calls.append(len(values))
        return original(values, labels, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", checked)
    score = answer_nll(logits, targets)
    assert score.nll_sum == pytest.approx(float(expected), abs=1e-4)
    assert sum(calls) == score.targets == int((targets != -100).sum())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_and_zero_targets(bad):
    logits = torch.zeros(1, 2, 3)
    targets = torch.tensor([[-100, 1]])
    logits[0, 0] = bad  # masked logits have no statistical role
    assert answer_nll(logits, targets).targets == 1
    logits[0, 1, 0] = bad
    with pytest.raises(ValueError, match="nonfinite"):
        answer_nll(logits, targets)
    with pytest.raises(ValueError, match="zero valid"):
        answer_nll(logits, torch.full_like(targets, -100))
    with pytest.raises(ValueError, match="outside vocabulary"):
        answer_nll(torch.zeros_like(logits), torch.tensor([[-100, -1]]))


@pytest.mark.parametrize(
    "processor,lora",
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_real_native_load_score_token_weighted_receipt(
    tmp_path, processor, lora
):
    path = tmp_path / "checkpoint"
    model = native_checkpoint(path, processor=processor, lora=lora)
    data = dataset(tmp_path / "data.jsonl")
    out = tmp_path / "score.json"
    expected_hash = tensor_state_sha256(model.state_dict())
    receipt = run(
        path,
        data,
        out,
        expected_loaded_state_sha256=expected_hash,
        expected_step=1,
    )
    assert receipt["complete"] and receipt["n_examples"] == 2
    assert receipt["loaded_state_sha256"] == expected_hash
    assert receipt["rows_sha256"] == canonical_json_sha256(receipt["rows"])
    scores = []
    for row in data.read():
        x, y, mask = sft_batch(
            [row], IntTokenizer(), 32, return_attention_mask=True
        )
        score = answer_nll(model(x, attention_mask=mask), y)
        scores.append(score)
    total = sum(score.nll_sum for score in scores)
    count = sum(score.targets for score in scores)
    assert receipt["result"]["nll"] == pytest.approx(total / count)
    assert receipt["result"]["targets"] == count == 9
    assert receipt["result"]["nll"] != pytest.approx(
        sum(score.nll for score in scores) / 2, abs=1e-5
    )
    assert json.loads(out.read_text()) == receipt
    before = out.read_bytes()
    with pytest.raises(FileExistsError):
        run(path, data, out)
    assert out.read_bytes() == before


def test_processor_only_uses_native_backbone_restore_and_exact_state_anchor(
    tmp_path,
):
    path = tmp_path / "processor"
    model = native_checkpoint(path, processor=True, processor_only=True)
    data = dataset(tmp_path / "data.jsonl")
    with pytest.raises(ValueError, match="requires expected_loaded_state"):
        run(path, data, tmp_path / "missing.json")
    with pytest.raises(ValueError, match="loaded model state SHA256"):
        run(
            path,
            data,
            tmp_path / "wrong.json",
            expected_loaded_state_sha256="a" * 64,
        )
    receipt = run(
        path,
        data,
        tmp_path / "valid.json",
        expected_loaded_state_sha256=tensor_state_sha256(model.state_dict()),
    )
    assert receipt["processor_only_payload"]
    backbone = path.with_name(path.name + "-backbone") / "model.safetensors"
    state = load_file(backbone)
    state[next(iter(state))].add_(1)
    save_file(state, backbone)
    with pytest.raises(ValueError, match="loaded model state SHA256"):
        run(
            path,
            data,
            tmp_path / "tampered.json",
            expected_loaded_state_sha256=receipt["loaded_state_sha256"],
        )


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "train_state.json",
        "model.safetensors",
        "optimizer.pt",
        "rng.pt",
    ],
)
def test_manifest_checks_every_payload(tmp_path, name):
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    member = path / name
    member.write_bytes(member.read_bytes() + b"tampered")
    with pytest.raises(TrainingCheckpointError, match="hash mismatch"):
        run(path, data, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


@pytest.mark.parametrize(
    "change", ["incomplete", "missing", "step", "fingerprint", "partial"]
)
def test_manifest_semantic_guards(tmp_path, change):
    path = tmp_path / "checkpoint"
    native_checkpoint(path, processor=True, lora=True)
    data = dataset(tmp_path / "data.jsonl")
    manifest = json.loads((path / "manifest.json").read_text())
    if change == "incomplete":
        manifest["complete"] = False
    elif change == "missing":
        del manifest["files"]["config.json"]
    elif change == "step":
        manifest["step"] += 1
    elif change == "fingerprint":
        manifest["training_fingerprint"]["model"]["sequence_len"] += 1
    else:
        state = load_file(path / "model.safetensors")
        del state[next(name for name in state if "lora_b" in name)]
        save_file(state, path / "model.safetensors")
        manifest["files"]["model.safetensors"] = file_sha256(
            path / "model.safetensors"
        )
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, KeyError, TrainingCheckpointError)):
        run(path, data, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_external_manifest_and_tokenizer_anchors(tmp_path):
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    with pytest.raises(ValueError, match="manifest SHA256"):
        evaluate_answer_nll(
            checkpoint=path,
            expected_manifest_sha256="a" * 64,
            data=data,
            out=tmp_path / "out.json",
        )
    with pytest.raises(ValueError, match="tokenizer SHA256"):
        run(
            path,
            data,
            tmp_path / "out.json",
            expected_tokenizer_sha256="a" * 64,
        )
    run(
        path,
        data,
        tmp_path / "out.json",
        expected_tokenizer_sha256=tokenizer_identity_sha256(IntTokenizer()),
    )


def test_data_whole_hash_count_duplicate_overlap_and_zero_target(tmp_path):
    data = dataset(tmp_path / "data.jsonl")
    with pytest.raises(ValueError, match="SHA256"):
        replace(data, sha256="a" * 64).read()
    with pytest.raises(ValueError, match="whole-file count"):
        replace(data, count=1).read()
    duplicate = dataset(
        tmp_path / "duplicate.jsonl",
        [{"prompt": " A\nB ", "answer": "a"}, {"prompt": "A B", "answer": "b"}],
    )
    with pytest.raises(ValueError, match="normalized-prompt duplicates"):
        duplicate.read()
    excluded = dataset(
        tmp_path / "excluded.jsonl", [{"prompt": " p ", "answer": "other"}]
    )
    with pytest.raises(ValueError, match="overlap"):
        evaluate_answer_nll(
            checkpoint=tmp_path / "nonexistent",
            expected_manifest_sha256="a" * 64,
            data=data,
            out=tmp_path / "out.json",
            disjoint_from=[excluded],
        )
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    zero = dataset(
        tmp_path / "zero.jsonl", [{"prompt": "p" * 32, "answer": "a"}]
    )
    with pytest.raises(ValueError, match="zero valid"):
        run(path, zero, tmp_path / "out.json")
    empty = dataset(tmp_path / "empty.jsonl", [{"prompt": "p", "answer": ""}])
    assert run(path, empty, tmp_path / "empty.json")["result"]["targets"] == 1
    cap = dataset(tmp_path / "cap.jsonl", [{"prompt": "p" * 31, "answer": "a"}])
    receipt = run(path, cap, tmp_path / "cap.json")
    assert receipt["cap_rows"] == receipt["result"]["targets"] == 1


@pytest.mark.parametrize("target", ["data", "checkpoint", "model", "tokenizer"])
def test_final_immutability_checks_real_forward(tmp_path, monkeypatch, target):
    path = tmp_path / "checkpoint"
    model = native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    original = type(model).forward
    changed = False
    tokenizer = IntTokenizer()
    if target == "tokenizer":
        monkeypatch.setattr(
            "gcw.evaluation.answer_nll.get_tokenizer", lambda name: tokenizer
        )

    def forward(self, *args, **kwargs):
        nonlocal changed
        result = original(self, *args, **kwargs)
        if not changed:
            changed = True
            if target == "data":
                data.path.write_bytes(data.path.read_bytes() + b"\n")
            elif target == "checkpoint":
                member = path / "train_state.json"
                member.write_bytes(member.read_bytes() + b"\n")
            elif target == "model":
                next(self.parameters()).add_(0.1)
            else:
                tokenizer.eot_id = 1
        return result

    monkeypatch.setattr(type(model), "forward", forward)
    with pytest.raises(
        (ValueError, TrainingCheckpointError), match="changed|hash mismatch"
    ):
        run(path, data, tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_paired_identity_guards_and_weighting(tmp_path):
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    run(path, data, tmp_path / "score.json")
    rows = list(
        load_answer_nll_rows(
            tmp_path / "score.json",
            expected_sha256=file_sha256(tmp_path / "score.json"),
        )
    )
    candidate = [replace(row, nll_sum=row.nll_sum + 1) for row in rows]
    paired = paired_answer_nll(rows, candidate, expected_count=data.count)
    with pytest.raises(ValueError, match="missing"):
        paired_answer_nll(rows[:1], candidate[:1], expected_count=data.count)
    assert paired.delta_nll == pytest.approx(2 / 9)
    bad_candidates = [
        candidate[:-1],
        candidate[::-1],
        [candidate[0], candidate[0]],
    ]
    for field in ("row_index", "targets", "length"):
        bad_candidates.append(
            [
                replace(
                    candidate[0], **{field: getattr(candidate[0], field) + 1}
                ),
                candidate[1],
            ]
        )
    for field in (
        "row_sha256",
        "input_sha256",
        "targets_sha256",
        "attention_mask_sha256",
        "tokenizer_sha256",
        "options_sha256",
        "data_sha256",
    ):
        bad_candidates.append(
            [replace(candidate[0], **{field: "a" * 64}), candidate[1]]
        )
    for bad in bad_candidates:
        with pytest.raises(ValueError, match="missing|reordered|mismatched"):
            paired_answer_nll(rows, bad, expected_count=data.count)
    duplicates = [rows[0], replace(rows[0], row_index=1)]
    with pytest.raises(ValueError, match="duplicate"):
        paired_answer_nll(duplicates, duplicates, expected_count=data.count)
    with pytest.raises(ValueError, match="nonfinite"):
        paired_answer_nll(
            rows,
            [replace(candidate[0], nll_sum=float("nan")), candidate[1]],
            expected_count=data.count,
        )


def test_explicit_config_migration_only():
    saved = {
        "legacy_bottleneck_generate": False,
        "processor_generation_mode": "windowed",
        "other": 4,
    }
    loaded = {**saved, "legacy_bottleneck_generate": None}
    assert _validate_loaded_config(saved, loaded)
    assert saved["legacy_bottleneck_generate"] is False
    with pytest.raises(ValueError):
        _validate_loaded_config(saved, {**loaded, "other": 5})
    with pytest.raises(ValueError):
        _validate_loaded_config(
            {**saved, "legacy_bottleneck_generate": True}, loaded
        )
    with pytest.raises(ValueError, match="answer_only"):
        AnswerNLLDataOptions(32, "int", loss_scope="full_sequence")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_dtype_auto_and_override(tmp_path, dtype):
    path = tmp_path / "checkpoint"
    native_checkpoint(path, dtype=dtype)
    data = dataset(tmp_path / "data.jsonl")
    receipt = run(path, data, tmp_path / "auto.json")
    assert receipt["loaded_dtypes"] == [str(dtype)]
    override = run(path, data, tmp_path / "override.json", dtype="float32")
    assert override["loaded_dtypes"] == ["torch.float32"]


def test_native_chat_template_options_and_tokenizer_pipeline(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    from gcw.data.tokenizers import get_tokenizer

    backend = Tokenizer(
        WordLevel(
            {
                "<eos>": 0,
                "<unk>": 1,
                "user": 2,
                "p": 3,
                "q": 4,
                "a": 5,
                "longer": 6,
                "suffix": 7,
                "assistant": 8,
            },
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    hf = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>"
    )
    hf.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    hf.save_pretrained(tmp_path / "tokenizer")
    options = AnswerNLLDataOptions(
        16,
        "hf:" + str(tmp_path / "tokenizer"),
        chat_template=True,
        enable_thinking=True,
        prompt_suffix="suffix",
    )
    path = tmp_path / "checkpoint"
    model = native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    receipt = run(path, data, tmp_path / "chat.json", options=options)
    tokenizer = get_tokenizer(options.tokenizer)
    x, y, mask = sft_batch(
        [data.read()[0]],
        tokenizer,
        16,
        chat_template=True,
        enable_thinking=True,
        prompt_suffix="suffix",
        return_attention_mask=True,
    )
    assert receipt["rows"][0]["input_sha256"] == tensor_state_sha256(
        {"input": x}
    )
    assert receipt["rows"][0]["targets_sha256"] == tensor_state_sha256(
        {"targets": y}
    )
    assert receipt["rows"][0]["nll_sum"] == pytest.approx(
        answer_nll(model(x, attention_mask=mask), y).nll_sum
    )
    before = tokenizer_identity_sha256(tokenizer)
    tokenizer.enc.backend_tokenizer.pre_tokenizer = None
    assert tokenizer_identity_sha256(tokenizer) != before


@pytest.mark.parametrize(
    "change", ["bytes", "incomplete", "rows", "total", "tokenizer"]
)
def test_paired_receipt_authentication(tmp_path, change):
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    out = tmp_path / "score.json"
    receipt = run(path, data, out)
    original_hash = file_sha256(out)
    if change == "bytes":
        out.write_bytes(out.read_bytes() + b"\n")
        expected = original_hash
    else:
        if change == "incomplete":
            receipt["complete"] = False
        elif change == "rows":
            receipt["rows"] = receipt["rows"][:-1]
        elif change == "total":
            receipt["result"]["nll"] += 1
        else:
            receipt["tokenizer_sha256"] = "a" * 64
        out.write_text(json.dumps(receipt))
        expected = file_sha256(out)
    with pytest.raises(ValueError):
        load_answer_nll_rows(out, expected_sha256=expected)


def test_thin_cli_real_cpu_checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    native_checkpoint(path)
    data = dataset(tmp_path / "data.jsonl")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_answer_nll.py",
            "--checkpoint",
            str(path),
            "--expected-manifest-sha256",
            file_sha256(path / "manifest.json"),
            "--data",
            str(data.path),
            "--expected-data-sha256",
            data.sha256,
            "--expected-count",
            str(data.count),
            "--out",
            str(tmp_path / "cli.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["targets"] == 9
