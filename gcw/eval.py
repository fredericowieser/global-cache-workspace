from __future__ import annotations

from pathlib import Path
import os

import torch

from gcw.checkpoint import load_checkpoint
from gcw.common import base_dir
from gcw.data.shards import data_loader
from gcw.data.tokenizers import COSMO2_VOCAB_SIZE, IntTokenizer, get_token_bytes, get_tokenizer
from gcw.loss_eval import evaluate_nll_bpb

try:
    from lm_eval.api.model import TemplateLM
    HAS_LM_EVAL = True
except ImportError:
    TemplateLM = object
    HAS_LM_EVAL = False

PPT_TASKS = ["lambada_openai", "hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge", "mmlu"]
PPT_FEWSHOT = {"lambada_openai": 0, "hellaswag": 0, "piqa": 0, "winogrande": 5, "arc_easy": 25, "arc_challenge": 25, "mmlu": 5}
ACC_NORM_TASKS = {"hellaswag", "piqa", "arc_easy", "arc_challenge"}
MATH_TASKS = ["gsm8k_openmath", "hendrycks_math", "svamp", "theoremqa", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"]
PAPER7_TASKS = ["gsm8k_openmath", "svamp", "theoremqa", "logiqa", "agieval_gaokao_mathqa", "gsm8k_hard"]
MATH_RUNS = [(MATH_TASKS, 0)]
PAPER7_RUNS = [(PAPER7_TASKS, 0)]
PPT_RUNS = [
    (["lambada_openai", "hellaswag", "piqa"], 0),
    (["winogrande"], 5),
    (["arc_easy", "arc_challenge"], 25),
    (["mmlu"], 5),
]
FORGETTING_TASKS = [
    "arc_easy",
    "arc_challenge",
    "boolq",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "social_iqa",
    "winogrande",
    "mmlu",
    "mmlu_pro",
    "bbh_cot_fewshot",
    "humaneval",
    "mbpp",
    "ifeval",
    "truthfulqa_mc1",
    "truthfulqa_mc2",
]


class GCWEvalLM(TemplateLM):
    def __init__(self, model, tokenizer, batch_size: int = 8):
        if not HAS_LM_EVAL:
            raise ImportError("Install the eval extra: uv sync --extra eval")
        super().__init__()
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = model.config.sequence_len

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eot_id

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens, **kwargs) -> str:
        return self.tokenizer.decode(tokens)

    @torch.no_grad()
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        device = self.model.get_device()
        out = [None] * len(requests)
        order = sorted(range(len(requests)), key=lambda i: len(requests[i][1]) + len(requests[i][2]))
        for start in range(0, len(order), self.batch_size):
            idxs = order[start : start + self.batch_size]
            inputs, continuations = [], []
            for i in idxs:
                _, ctx, cont = requests[i]
                full = (ctx + cont)[-(self.max_length + 1) :]
                inputs.append(full[:-1])
                continuations.append(cont[-self.max_length :])
            max_len = max(len(x) for x in inputs)
            padded = torch.zeros(len(inputs), max_len, dtype=torch.long, device=device)
            for row, inp in enumerate(inputs):
                padded[row, : len(inp)] = torch.tensor(inp, dtype=torch.long, device=device)
            logprobs = torch.log_softmax(self.model(padded), dim=-1)
            for row, i in enumerate(idxs):
                cont = continuations[row]
                cont_len = len(cont)
                inp_len = len(inputs[row])
                logp = logprobs[row, inp_len - cont_len : inp_len]
                target = torch.tensor(cont, dtype=torch.long, device=device)
                vals = logp.gather(-1, target[:, None]).squeeze(-1)
                out[i] = (float(vals.sum().item()), bool((logp.argmax(-1) == target).all().item()))
        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        out = []
        for req in requests:
            text = req.args[0]
            out.append(sum(x[0] for x in self._loglikelihood_tokens([(None, [self.eot_token_id], self.tok_encode(text))])))
        return out

    @torch.no_grad()
    def generate_until(self, requests, disable_tqdm: bool = False):
        outs = []
        device = self.model.get_device()
        for req in requests:
            prompt, kwargs = req.args
            max_new = int(kwargs.get("max_gen_toks", 256))
            until = kwargs.get("until") or []
            ids = torch.tensor(self.tok_encode(prompt)[-self.max_length :], dtype=torch.long, device=device).unsqueeze(0)
            gen = self.model.generate(ids, max_new_tokens=max_new, eos_token_id=self.eot_token_id)[0].tolist()
            text = self.tok_decode(gen[len(ids[0]) :])
            for stop in until:
                if stop in text:
                    text = text.split(stop)[0]
            outs.append(text)
        return outs


def run_core(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, [*PPT_RUNS, *MATH_RUNS], limit=limit, batch_size=batch_size)


def run_ppt(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, PPT_RUNS, limit=limit, batch_size=batch_size)


def run_math(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, MATH_RUNS, limit=limit, batch_size=batch_size)


def run_paper7(checkpoint: str | Path, limit: int | None = None, batch_size: int = 8) -> dict:
    return run_task_groups(checkpoint, PAPER7_RUNS, limit=limit, batch_size=batch_size)


def run_fineweb(
    checkpoint: str | Path,
    data_dir: str | Path | None = None,
    eval_tokens: int = 10 * 524288,
    batch_size: int = 16,
) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg, state = load_checkpoint(checkpoint, device=device)
    if cfg.family != "smollm" or cfg.vocab_size != COSMO2_VOCAB_SIZE:
        raise ValueError("FineWeb NLL/BPB eval currently expects a cosmo2 SmolLM checkpoint")
    steps = max(1, eval_tokens // (batch_size * cfg.sequence_len))
    batches = data_loader(batch_size, cfg.sequence_len, split="val", device=device, data_dir=data_dir, infinite=True)
    nll, bpb = evaluate_nll_bpb(model, batches, steps, get_token_bytes(device))
    return {
        "model_step": state.get("step"),
        "val_nll": nll,
        "val_bpb": bpb,
        "eval_tokens": steps * batch_size * cfg.sequence_len,
        "batch_size": batch_size,
        "sequence_len": cfg.sequence_len,
    }


def run_task_groups(
    checkpoint: str | Path,
    runs: list[tuple[list[str], int]],
    limit: int | None = None,
    batch_size: int = 8,
) -> dict:
    if not HAS_LM_EVAL:
        raise ImportError("Install the eval extra: uv sync --extra eval")
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass
    os.environ.setdefault("HF_DATASETS_CACHE", str(base_dir() / "hf_datasets_eval_v3"))
    import lm_eval
    from lm_eval.tasks import TaskManager
    model, cfg, state = load_checkpoint(checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    lm = GCWEvalLM(model, _tokenizer_for(cfg, state), batch_size=batch_size)
    task_manager = TaskManager(include_path=str(Path("configs/tasks").resolve()), verbosity="ERROR")
    merged: dict = {"results": {}, "groups": {}, "n-samples": {}, "n-shot": {}, "versions": {}, "higher_is_better": {}}
    for tasks, fewshot in runs:
        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=fewshot,
            limit=limit,
            bootstrap_iters=0,
            log_samples=False,
            verbosity="ERROR",
            task_manager=task_manager,
            confirm_run_unsafe_code=True,
        )
        for key in merged:
            if key in out and isinstance(out[key], dict):
                merged[key].update(out[key])
    return merged


def _tokenizer_for(config, state):
    if state.get("source") and config.family in {"llama", "qwen3"}:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Converted HF checkpoints require transformers for eval tokenization.") from exc
        tok = AutoTokenizer.from_pretrained(state["source"], trust_remote_code=False)
        return _HFTokenizerAdapter(tok)
    if config.family == "smollm" and config.vocab_size == COSMO2_VOCAB_SIZE:
        return get_tokenizer("cosmo2")
    return IntTokenizer(vocab_size=config.vocab_size, eot_id=0)


class _HFTokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eot_id = tokenizer.eos_token_id or tokenizer.pad_token_id or 0
        self.chat_template = getattr(tokenizer, "chat_template", None)
        self.default_chat_template = getattr(tokenizer, "default_chat_template", None)

    def encode(self, text: str, add_special_tokens: bool | None = None) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=bool(add_special_tokens))

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)
