from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .constants import ORDER_CANONICAL, ORDER_SWAPPED, PRIORITY_A_OVER_B, PRIORITY_B_OVER_A
from .prompts import (
    build_context_forward_prompt,
    build_judge_user_prompt,
    build_preflight_prompt,
    build_single_criterion_prompt,
    chat_format,
    ordered_candidates,
    paraphrase_count,
    priority_criteria,
)
from .schema import CPOItem
from .utils import read_jsonl, write_jsonl


@dataclass
class ModelConfig:
    tag: str
    name: str
    trust_remote_code: bool = True
    load_in_4bit: bool = True
    # float16 for sm86 / sm89; bfloat16 supported by Blackwell (sm100) too
    # but float16 is fine and avoids any driver edge cases.
    torch_dtype: str = "float16"
    # RTX 5060 Ti has 16 GiB VRAM; 15 GiB budget leaves ~1 GiB for activations
    # and OS overhead.  Using 14 GiB was conservative; 15 GiB is better for
    # 7-8B models at 4-bit which need ~5-6 GiB model weight + KV cache.
    gpu_max_memory: str = "15GiB"
    cpu_max_memory: str = "48GiB"
    max_length: int = 1536
    batch_size: int = 4
    cache_dir: Optional[str] = None


# ---------------------------------------------------------------------------
# A/B token discovery and next-token indexing
# ---------------------------------------------------------------------------

def _discover_ab_ids(tokenizer) -> Tuple[List[int], List[int]]:
    """Return disjoint token ID sets that decode to bare labels A and B.

    Several prompt-tokenizer combinations encode label letters differently:
    plain ``A``, leading-space `` A``, and newline-prefixed ``\nA`` can be
    distinct single tokens. We keep every single-token form that decodes to the
    bare label after stripping, then aggregate scores with logsumexp.
    """

    def probe(label: str) -> List[int]:
        ids: List[int] = []
        for surface in [label, " " + label, "\n" + label]:
            enc = tokenizer.encode(surface, add_special_tokens=False)
            if len(enc) == 1:
                decoded = tokenizer.decode([enc[0]]).strip()
                if decoded == label:
                    ids.append(int(enc[0]))
        return sorted(set(ids))

    a_ids = probe("A")
    b_ids = probe("B")
    overlap = set(a_ids) & set(b_ids)
    if overlap:
        raise RuntimeError(
            f"A/B token ID overlap for model tokenizer: "
            f"A={a_ids}, B={b_ids}, overlap={overlap}. "
            f"This model cannot be used with next-token A/B scoring."
        )
    if not a_ids:
        raise RuntimeError(
            "Could not find any single-token encoding for 'A'. "
            "Check tokenizer chat_template or try a different model."
        )
    if not b_ids:
        raise RuntimeError(
            "Could not find any single-token encoding for 'B'. "
            "Check tokenizer chat_template or try a different model."
        )
    return a_ids, b_ids


def _last_real_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final non-padding token index for each sequence.

    The previous implementation used ``attention_mask.sum(dim=1) - 1``. That is
    correct only under right padding. Some decoder-only tokenizers default to
    left padding during batched inference, where the final real token is at the
    right edge of the sequence. This helper works for right padding, left
    padding, and any mixed mask where at least one token is real.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be rank-2, got shape={tuple(attention_mask.shape)}")
    if attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must have non-zero sequence length")
    if (attention_mask.sum(dim=1) == 0).any():
        raise ValueError("attention_mask contains an all-padding sequence")

    # Flip each row and find the first real token from the right. If seq_len is
    # T and the rightmost 1 is r positions from the right, the original index is
    # T - 1 - r. This avoids any assumption about padding side.
    reversed_first_real = attention_mask.flip(dims=[1]).argmax(dim=1)
    return attention_mask.shape[1] - 1 - reversed_first_real


# ---------------------------------------------------------------------------
# ABJudge
# ---------------------------------------------------------------------------

class ABJudge:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self.device = None
        self.a_ids: List[int] = []
        self.b_ids: List[int] = []
        self.a_tensor: Optional[torch.Tensor] = None
        self.b_tensor: Optional[torch.Tensor] = None

    def load(self) -> None:
        dtype = torch.float16 if self.cfg.torch_dtype == "float16" else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.name,
            trust_remote_code=self.cfg.trust_remote_code,
            cache_dir=self.cfg.cache_dir,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Keep batched prompt tensors predictable. The scoring helper below is
        # padding-side robust, but right padding makes stored diagnostics easier
        # to inspect and avoids tokenizer defaults changing silently.
        self.tokenizer.padding_side = "right"

        max_memory = {0: self.cfg.gpu_max_memory, "cpu": self.cfg.cpu_max_memory}
        common = dict(
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=self.cfg.trust_remote_code,
            low_cpu_mem_usage=True,
            cache_dir=self.cfg.cache_dir,
        )
        if self.cfg.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name, quantization_config=bnb_config, **common
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.name, torch_dtype=dtype, **common
            )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.a_ids, self.b_ids = _discover_ab_ids(self.tokenizer)
        self.a_tensor = torch.tensor(self.a_ids, device=self.device, dtype=torch.long)
        self.b_tensor = torch.tensor(self.b_ids, device=self.device, dtype=torch.long)

    def scorer_info(self) -> Dict[str, Any]:
        assert self.tokenizer is not None
        return {
            "model_tag": self.cfg.tag,
            "model_name": self.cfg.name,
            "interface": "chat_template__next_token__logsumexp",
            "padding_side": getattr(self.tokenizer, "padding_side", None),
            "last_token_indexing": "rightmost_nonpad_attention_mask_robust",
            "A_TOKEN_IDS": self.a_ids,
            "B_TOKEN_IDS": self.b_ids,
            "A_DECODED": [self.tokenizer.decode([i]) for i in self.a_ids],
            "B_DECODED": [self.tokenizer.decode([i]) for i in self.b_ids],
        }

    def unload(self) -> None:
        for attr in ["model", "tokenizer"]:
            try:
                delattr(self, attr)
                setattr(self, attr, None)
            except Exception:
                pass
        self.a_tensor = None
        self.b_tensor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def score_batch(
        self,
        user_texts: Sequence[str],
        system_texts: Sequence[str],
    ) -> List[Dict[str, Any]]:
        assert self.model is not None and self.tokenizer is not None
        assert self.device is not None
        assert self.a_tensor is not None and self.b_tensor is not None

        prompts = [chat_format(self.tokenizer, u, s) for u, s in zip(user_texts, system_texts)]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            out = self.model(**inputs, use_cache=False)

        logits = out.logits.float()  # (B, T, V)
        last_idx = _last_real_token_indices(inputs["attention_mask"])
        batch_idx = torch.arange(logits.shape[0], device=logits.device)
        last_logits = logits[batch_idx, last_idx, :]  # (B, V)

        score_a = last_logits[:, self.a_tensor].logsumexp(dim=-1).detach().cpu().numpy()
        score_b = last_logits[:, self.b_tensor].logsumexp(dim=-1).detach().cpu().numpy()

        results: List[Dict[str, Any]] = []
        for a, b in zip(score_a, score_b):
            a = float(a)
            b = float(b)
            pred = "A" if a >= b else "B"
            mx = max(a, b)
            ea, eb = math.exp(a - mx), math.exp(b - mx)
            denom = ea + eb
            results.append(
                {
                    "pred_label": pred,
                    "score_A": a,
                    "score_B": b,
                    "prob_A": ea / denom,
                    "prob_B": eb / denom,
                    "margin_abs": abs(a - b),
                    "margin_signed_A_minus_B": a - b,
                }
            )

        del out, logits, last_logits, inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def make_preflight_items(n: int) -> List[Dict[str, Any]]:
    """Deterministic preflight items that cover both position slots."""
    base = [
        ("Which answer is factually correct?", "Paris is the capital of France.", "Berlin is the capital of France.", "A"),
        ("Which answer is factually correct?", "The Sun is a planet.", "The Sun is a star.", "B"),
        ("Which answer solves 2 + 2 correctly?", "The answer is 4.", "The answer is 5.", "A"),
        ("Which answer solves 10 - 3 correctly?", "The answer is 9.", "The answer is 7.", "B"),
        ("Which answer follows valid JSON with key answer?", '{"answer": 42}', "The answer is 42.", "A"),
        ("Which answer is a single word only?", "The answer is yes.", "Yes", "B"),
        ("Which answer is more complete?", "Apples, bananas, oranges, and grapes.", "Apples and bananas.", "A"),
        ("Which answer is more concise while still answering?", "The color is blue with many unnecessary details added here.", "Blue.", "B"),
    ]
    rows: List[Dict[str, Any]] = []
    for i, (q, a, b, correct) in enumerate(base):
        rows.append({"example_id": f"{i}_orig", "question": q, "answer_a": a, "answer_b": b, "correct": correct})
        rows.append(
            {
                "example_id": f"{i}_swap",
                "question": q,
                "answer_a": b,
                "answer_b": a,
                "correct": "B" if correct == "A" else "A",
            }
        )
    extended = rows[:]
    while len(extended) < n:
        extended.extend(rows)
    return extended[:n]


def run_preflight(judge: ABJudge, n: int, batch_size: int) -> pd.DataFrame:
    items = make_preflight_items(n)
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        user_texts = [build_preflight_prompt(x["question"], x["answer_a"], x["answer_b"]) for x in batch]
        system_texts = ["You are an impartial evaluator."] * len(batch)
        scores = judge.score_batch(user_texts, system_texts)
        for item, score in zip(batch, scores):
            row = {**item, **score}
            row["is_correct"] = int(row["pred_label"] == row["correct"])
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Item loading
# ---------------------------------------------------------------------------

def load_items(paths: Sequence[Path]) -> List[CPOItem]:
    items: List[CPOItem] = []
    for path in paths:
        for d in read_jsonl(path):
            items.append(CPOItem.from_dict(d))
    return items


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_request_list(
    judge: ABJudge,
    requests: List[Dict[str, Any]],
    batch_size: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        scores = judge.score_batch(
            [x["user_text"] for x in batch],
            [x["system_text"] for x in batch],
        )
        for item, score in zip(batch, scores):
            row = dict(item["meta"])
            row.update(score)
            if "label_to_candidate_A" in row:
                row["pred_candidate"] = row["label_to_candidate_A"] if row["pred_label"] == "A" else row["label_to_candidate_B"]
                row["is_priority_obedient"] = int(row["pred_candidate"] == row["expected_candidate"])
            rows.append(row)
    return pd.DataFrame(rows)


def build_direct_requests(
    items: List[CPOItem],
    method: str,
    n_paraphrases: int,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    n_para = paraphrase_count(method, n_paraphrases)
    for item in items:
        for order in [ORDER_CANONICAL, ORDER_SWAPPED]:
            mp = ordered_candidates(item, order)
            for priority in [PRIORITY_A_OVER_B, PRIORITY_B_OVER_A]:
                primary, secondary = priority_criteria(item, priority)
                expected_candidate = item.expected_candidate(priority)
                for paraphrase_id in range(n_para):
                    text = build_judge_user_prompt(
                        item.task,
                        str(mp["A"]),
                        str(mp["B"]),
                        primary,
                        secondary,
                        method,
                        paraphrase_id,
                    )
                    requests.append(
                        {
                            "user_text": text,
                            "system_text": "You are an impartial LLM-as-a-judge evaluator.",
                            "meta": {
                                "item_id": item.item_id,
                                "tier": item.tier,
                                "source": item.source,
                                "family": item.family,
                                "pair_type": item.pair_type,
                                "method": method,
                                "order": order,
                                "priority": priority,
                                "paraphrase_id": paraphrase_id,
                                "expected_candidate": expected_candidate,
                                "label_to_candidate_A": mp["label_to_candidate"]["A"],
                                "label_to_candidate_B": mp["label_to_candidate"]["B"],
                            },
                        }
                    )
    return requests


def run_direct_like(
    judge: ABJudge,
    items: List[CPOItem],
    method: str,
    n_paraphrases: int,
    batch_size: int,
) -> pd.DataFrame:
    reqs = build_direct_requests(items, method, n_paraphrases)
    return _score_request_list(judge, reqs, batch_size)


# ---------------------------------------------------------------------------
# Decomposed criterion scoring
# ---------------------------------------------------------------------------

def run_decomposed(
    judge: ABJudge,
    items: List[CPOItem],
    batch_size: int,
) -> pd.DataFrame:
    """Score criteria independently, then aggregate by explicit priority rule."""
    criterion_requests: List[Dict[str, Any]] = []
    for item in items:
        for order in [ORDER_CANONICAL, ORDER_SWAPPED]:
            mp = ordered_candidates(item, order)
            for criterion_name, criterion_text, expected_candidate in [
                ("criterion_a", item.criterion_a, item.winner_criterion_a),
                ("criterion_b", item.criterion_b, item.winner_criterion_b),
            ]:
                criterion_requests.append(
                    {
                        "user_text": build_single_criterion_prompt(
                            item.task, str(mp["A"]), str(mp["B"]), criterion_text
                        ),
                        "system_text": "You are an impartial evaluator.",
                        "meta": {
                            "item_id": item.item_id,
                            "tier": item.tier,
                            "source": item.source,
                            "family": item.family,
                            "pair_type": item.pair_type,
                            "method": "criterion_only",
                            "order": order,
                            "criterion_name": criterion_name,
                            "expected_candidate": expected_candidate,
                            "label_to_candidate_A": mp["label_to_candidate"]["A"],
                            "label_to_candidate_B": mp["label_to_candidate"]["B"],
                        },
                    }
                )

    crit_df = _score_request_list(judge, criterion_requests, batch_size)
    lookup: Dict[tuple, dict] = {}
    for _, r in crit_df.iterrows():
        lookup[(r["item_id"], r["order"], r["criterion_name"])] = r.to_dict()

    rows: List[Dict[str, Any]] = []
    for item in items:
        for order in [ORDER_CANONICAL, ORDER_SWAPPED]:
            a_key = (item.item_id, order, "criterion_a")
            b_key = (item.item_id, order, "criterion_b")
            if a_key not in lookup or b_key not in lookup:
                continue
            a_row = lookup[a_key]
            b_row = lookup[b_key]
            for priority in [PRIORITY_A_OVER_B, PRIORITY_B_OVER_A]:
                chosen = a_row["pred_candidate"] if priority == PRIORITY_A_OVER_B else b_row["pred_candidate"]
                expected = item.expected_candidate(priority)
                rows.append(
                    {
                        "item_id": item.item_id,
                        "tier": item.tier,
                        "source": item.source,
                        "family": item.family,
                        "pair_type": item.pair_type,
                        "method": "decomposed",
                        "order": order,
                        "priority": priority,
                        "paraphrase_id": 0,
                        "expected_candidate": expected,
                        "pred_candidate": chosen,
                        "is_priority_obedient": int(chosen == expected),
                        "criterion_a_pred_candidate": a_row["pred_candidate"],
                        "criterion_b_pred_candidate": b_row["pred_candidate"],
                        "margin_abs": float(max(a_row.get("margin_abs", 0.0) or 0.0, b_row.get("margin_abs", 0.0) or 0.0)),
                        "score_A": np.nan,
                        "score_B": np.nan,
                        "prob_A": np.nan,
                        "prob_B": np.nan,
                        "pred_label": "NA",
                    }
                )

    out = pd.DataFrame(rows)
    out.attrs["criterion_rows"] = crit_df
    return out


# ---------------------------------------------------------------------------
# Context-forward judging
# ---------------------------------------------------------------------------

def run_context_forward(
    judge: ABJudge,
    items: List[CPOItem],
    batch_size: int,
) -> pd.DataFrame:
    """Run model-mediated final judging after explicit criterion winners."""
    decomp = run_decomposed(judge, items, batch_size)
    decomp_lookup: Dict[tuple, dict] = {}
    for _, r in decomp.iterrows():
        decomp_lookup[(r["item_id"], r["order"], r["priority"])] = r.to_dict()

    requests: List[Dict[str, Any]] = []
    for item in items:
        for order in [ORDER_CANONICAL, ORDER_SWAPPED]:
            mp = ordered_candidates(item, order)
            candidate_to_label = mp["candidate_to_label"]
            base_key = (item.item_id, order, PRIORITY_A_OVER_B)
            if base_key not in decomp_lookup:
                continue
            base = decomp_lookup[base_key]
            a_pred_candidate = base.get("criterion_a_pred_candidate")
            b_pred_candidate = base.get("criterion_b_pred_candidate")
            if a_pred_candidate is None or b_pred_candidate is None:
                continue

            for priority in [PRIORITY_A_OVER_B, PRIORITY_B_OVER_A]:
                primary, secondary = priority_criteria(item, priority)
                expected = item.expected_candidate(priority)
                if priority == PRIORITY_A_OVER_B:
                    primary_winner_cand = a_pred_candidate
                    secondary_winner_cand = b_pred_candidate
                else:
                    primary_winner_cand = b_pred_candidate
                    secondary_winner_cand = a_pred_candidate

                if primary_winner_cand not in candidate_to_label or secondary_winner_cand not in candidate_to_label:
                    continue

                requests.append(
                    {
                        "user_text": build_context_forward_prompt(
                            item.task,
                            str(mp["A"]),
                            str(mp["B"]),
                            primary,
                            secondary,
                            candidate_to_label[primary_winner_cand],
                            candidate_to_label[secondary_winner_cand],
                        ),
                        "system_text": (
                            "You are an impartial evaluator. "
                            "Follow the structured intermediate findings exactly."
                        ),
                        "meta": {
                            "item_id": item.item_id,
                            "tier": item.tier,
                            "source": item.source,
                            "family": item.family,
                            "pair_type": item.pair_type,
                            "method": "context_forward",
                            "order": order,
                            "priority": priority,
                            "paraphrase_id": 0,
                            "expected_candidate": expected,
                            "label_to_candidate_A": mp["label_to_candidate"]["A"],
                            "label_to_candidate_B": mp["label_to_candidate"]["B"],
                        },
                    }
                )

    return _score_request_list(judge, requests, batch_size)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataframe_jsonl(df: pd.DataFrame, path: Path) -> None:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    write_jsonl(records, path)
