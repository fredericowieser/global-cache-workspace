import json

from gcw.config import load_train_config
from gcw.data.sft import prepare_openmath
from scripts import publish_bottleneck_llm_artifacts as publish


def test_publication_spec_has_expected_artifact_counts():
    legacy = publish.legacy_artifacts()
    sft = publish.sft_artifacts()

    assert len(legacy) == 8
    assert len(sft) == 4
    assert all("/mnt/jfs-01/shared/gcw/checkpoints/bottleneck_llm/" in str(item[3]) for item in legacy)
    assert all("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/" in str(item[2]) for item in sft)


def test_prepare_openmath_normalizes_jsonl(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"problem": "2+2?", "solution": "4"}),
                json.dumps({"problem": "3+3?", "generated_solution": "6", "expected_answer": "six"}),
                json.dumps({"messages": [{"content": "Q"}, {"content": "A"}]}),
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "train.jsonl"

    assert prepare_openmath(source, out) == 3

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"prompt": "2+2?", "answer": "4"},
        {"prompt": "3+3?", "answer": "6"},
        {"prompt": "Q", "answer": "A"},
    ]


def test_bottleneck_llm_training_configs_load_with_expected_modes():
    frozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_frozen.yaml")
    unfrozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_bgt_from_sft_unfrozen.yaml")
    icarus_frozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_frozen.yaml")
    icarus_unfrozen = load_train_config("configs/train/bottleneck_llm/llama32_1b_icarus_from_sft_unfrozen.yaml")
    tf_sft = load_train_config("configs/train/bottleneck_llm/llama32_1b_tf_sft.yaml")

    assert frozen.freeze_backbone
    assert frozen.model.variant == "bgt"
    assert frozen.backbone_checkpoint.startswith("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/")
    assert not unfrozen.freeze_backbone
    assert unfrozen.model.variant == "bgt"
    assert icarus_frozen.freeze_backbone
    assert icarus_frozen.model.variant == "icarus"
    assert icarus_frozen.data.dataset == "/mnt/jfs-01/shared/gcw/data/OpenMathInstruct-2/train.jsonl"
    assert icarus_frozen.backbone_checkpoint.startswith("/mnt/jfs-01/shared/gcw/checkpoints/sft_backbones/")
    assert not icarus_unfrozen.freeze_backbone
    assert icarus_unfrozen.model.variant == "icarus"
    assert tf_sft.model.variant == "tf"
    assert tf_sft.model.cache_processor is None
