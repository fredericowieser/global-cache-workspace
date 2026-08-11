from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from gcw.process_data import AnswerOnlyRecord


ROOT = Path(__file__).parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Tokenizer:
    eot_id = 0

    def __init__(self) -> None:
        self.thinking: list[bool] = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt
        assert messages[0]["role"] == "user"
        self.thinking.append(enable_thinking)
        return [10, 11, 12, 13] if enable_thinking else [20, 21, 22]

    def encode(self, _text):
        return [7]


def test_eval_tokenizer_uses_real_qwen_thinking_switch_and_exact_split() -> None:
    runner = _load_script("run_unified_flow_position.py")
    tokenizer = _Tokenizer()
    record = AnswerOnlyRecord("row", "question", r"\boxed{1}")

    off = runner._tokenize_eval_mode(
        record,
        tokenizer,
        {"use_chat_template": True, "enable_thinking": False, "prompt_suffix": "suffix"},
        max_prompt_tokens=16,
        max_answer_tokens=4,
        device=torch.device("cpu"),
    )
    on = runner._tokenize_eval_mode(
        record,
        tokenizer,
        {"use_chat_template": True, "enable_thinking": True, "prompt_suffix": "suffix"},
        max_prompt_tokens=16,
        max_answer_tokens=4,
        device=torch.device("cpu"),
    )

    assert tokenizer.thinking == [False, True]
    assert off.prompt_ids.tolist() + off.answer_response_prefix_ids.tolist() == [20, 21, 22]
    assert on.prompt_ids.tolist() + on.answer_response_prefix_ids.tolist() == [10, 11, 12, 13]


def test_generated_config_requires_full_gsm8k_and_both_thinking_modes() -> None:
    generator = _load_script("generate_unified_flow_position_config.py")
    config = generator.canonical_config()

    assert config["evaluation"]["expected_rows"] == 1319
    modes = config["evaluation"]["thinking_modes"]
    assert modes["thinking_off"]["enable_thinking"] is False
    assert modes["thinking_on"]["enable_thinking"] is True
    assert config["runtime"] == {"device": "cuda", "dtype": "bfloat16"}
