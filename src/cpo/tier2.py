from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .schema import CPOItem
from .utils import atomic_write_text, read_jsonl, truncate_text, write_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_first(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in [None, ""]:
            return d[k]
    return None


def _label_to_candidate(label: Any) -> Optional[str]:
    if label is None:
        return None
    s = str(label).strip().lower()
    if s in {"1", "a", "output_1", "candidate_1", "response_1", "model_a", "left"}:
        return "candidate_1"
    if s in {"2", "b", "output_2", "candidate_2", "response_2", "model_b", "right"}:
        return "candidate_2"
    return None


# ---------------------------------------------------------------------------
# LLMBar normalizer
# ---------------------------------------------------------------------------

def normalize_llmbar(
    raw_path: Path,
    out_path: Path,
    target_n: int,
    seed: int,
    require_adversarial: bool = True,
) -> int:
    """Build Tier 2 CPO items from LLMBar.

    Criterion assignment:
      criterion_a = "instruction following and task correctness"
      criterion_b = "surface fluency, engagement, or apparent helpfulness"

    The LLMBar gold label (output_1 vs output_2) is used as the winner of
    criterion_a (instruction following).  The adversarial nature of LLMBar
    means the non-gold output was *crafted* to look superficially attractive,
    making it a naturalistic instance of a surface-quality winner.  This
    assignment is documented in the paper's dataset section and in item metadata.

    IMPORTANT: The criterion assignment is a CPO-specific annotation layered
    on top of LLMBar — not a claim about LLMBar's internal labels.  The paper
    must clearly state this.

    Parameters
    ----------
    require_adversarial:
        When True (default), keep only rows whose source metadata contains
        known adversarial split markers.  Set to False to include all rows
        (useful when the HuggingFace download flattens splits).
    """
    rows = read_jsonl(raw_path)
    rng = random.Random(seed)
    candidates: List[CPOItem] = []
    diagnostics: Dict[str, int] = {
        "raw_rows": len(rows),
        "skipped_not_adversarial": 0,
        "skipped_missing_fields": 0,
        "normalized": 0,
    }

    ADVERSARIAL_MARKERS = [
        "adversarial", "neighbor", "gptinst", "gptout", "manual",
        "llmbar",  # HF config name marker
    ]

    for r in rows:
        # Build a source fingerprint from all metadata fields.
        source_path = str(r.get("_source_path", r.get("_hf_split", r.get("_hf_config", ""))))
        source_fingerprint = (source_path + " " + json.dumps(r, ensure_ascii=False)[:800]).lower()

        is_adversarial = any(m in source_fingerprint for m in ADVERSARIAL_MARKERS)

        if require_adversarial and not is_adversarial:
            diagnostics["skipped_not_adversarial"] += 1
            continue

        task = _get_first(r, ["input", "instruction", "prompt", "question", "task"])
        out1 = _get_first(r, ["output_1", "response_1", "answer_1", "candidate_1", "output1"])
        out2 = _get_first(r, ["output_2", "response_2", "answer_2", "candidate_2", "output2"])
        label = _get_first(r, ["label", "gold", "winner", "preference", "preferred"])
        gold = _label_to_candidate(label)

        if task is None or out1 is None or out2 is None or gold is None:
            diagnostics["skipped_missing_fields"] += 1
            continue

        other = "candidate_2" if gold == "candidate_1" else "candidate_1"

        item = CPOItem(
            item_id=f"tier2_llmbar_{len(candidates):05d}",
            tier="tier2_llmbar",
            source="llmbar_adversarial",
            family="instruction_following_vs_surface_quality",
            pair_type="conflict",
            task=truncate_text(task),
            criterion_a="instruction following and task correctness",
            criterion_b="surface fluency, engagement, or apparent helpfulness",
            candidate_1=truncate_text(out1),
            candidate_2=truncate_text(out2),
            winner_criterion_a=gold,
            winner_criterion_b=other,
            metadata={
                "original_label": label,
                "source_path": source_path,
                "is_adversarial": is_adversarial,
                # Critical documentation: this is how the CPO annotation maps
                # onto LLMBar's original labels.  Must be reproducible by any
                # reader of the paper's Appendix.
                "criterion_assignment_note": (
                    "LLMBar gold label used as winner of criterion_a "
                    "(instruction following/correctness). The non-gold "
                    "adversarial output is assigned as winner of criterion_b "
                    "(surface quality), consistent with LLMBar's design that "
                    "adversarial outputs are superficially attractive but "
                    "instruction-non-compliant."
                ),
            },
        )
        candidates.append(item)
        diagnostics["normalized"] += 1

    rng.shuffle(candidates)
    if target_n > 0:
        candidates = candidates[:target_n]
    n = write_jsonl([x.to_dict() for x in candidates], out_path)
    diagnostics["written"] = n
    atomic_write_text(
        out_path.with_suffix(".diagnostics.json"),
        json.dumps(diagnostics, indent=2),
    )
    return n


# ---------------------------------------------------------------------------
# WildBench response extractor
# ---------------------------------------------------------------------------

def _find_responses_in_wildbench_row(
    r: Dict[str, Any],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Extract candidate response (name, text, metadata) pairs from a WildBench row."""
    responses: List[Tuple[str, str, Dict[str, Any]]] = []

    # Explicit list fields
    for key in ["responses", "model_outputs", "outputs", "answers", "candidates"]:
        val = r.get(key)
        if isinstance(val, list):
            for j, item in enumerate(val):
                if isinstance(item, dict):
                    text = _get_first(
                        item, ["response", "output", "answer", "text", "content"]
                    )
                    name = str(
                        _get_first(item, ["model", "model_name", "name", "id"])
                        or f"candidate_{j}"
                    )
                    if text:
                        responses.append((name, str(text), item))
                elif isinstance(item, str) and item.strip():
                    responses.append((f"candidate_{j}", item, {}))

    # Pairwise schemas
    for a_key, b_key in [
        ("response_a", "response_b"),
        ("answer_a", "answer_b"),
        ("output_a", "output_b"),
        ("model_a_output", "model_b_output"),
        ("candidate_a", "candidate_b"),
    ]:
        if a_key in r and b_key in r:
            responses.append(("A", str(r[a_key]), {"field": a_key}))
            responses.append(("B", str(r[b_key]), {"field": b_key}))

    # Wide column schemas (model output columns by name pattern)
    known_fields = {x[2].get("field") for x in responses if isinstance(x[2], dict)}
    for k, v in r.items():
        kl = k.lower()
        if (
            isinstance(v, str)
            and len(v) > 80  # minimum meaningful response length
            and any(s in kl for s in ["response", "output", "answer"])
            and k not in known_fields
        ):
            responses.append((k, v, {"field": k}))

    # Deduplicate by normalized text
    seen: set = set()
    deduped: List[Tuple[str, str, Dict[str, Any]]] = []
    for name, text, meta in responses:
        norm = re.sub(r"\s+", " ", text).strip()[:600]
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append((name, text, meta))
    return deduped


def _extract_task(r: Dict[str, Any]) -> Optional[str]:
    return _get_first(
        r,
        ["instruction", "prompt", "query", "question", "input", "task",
         "conversation", "user_prompt"],
    )


def _extract_checklists(r: Dict[str, Any]) -> List[str]:
    val = _get_first(
        r,
        ["checklist", "checklists", "criteria", "rubric",
         "score_rubric", "evaluation_criteria"],
    )
    if val is None:
        return []
    if isinstance(val, str):
        return [x.strip() for x in re.split(r"\n+|;", val) if x.strip()]
    if isinstance(val, list):
        out = []
        for x in val:
            if isinstance(x, dict):
                q = _get_first(x, ["question", "criterion", "description", "text"])
                if q:
                    out.append(str(q))
            elif isinstance(x, str):
                out.append(x)
        return out
    if isinstance(val, dict):
        return [str(v) for v in val.values() if isinstance(v, str)]
    return []


# ---------------------------------------------------------------------------
# WildBench heuristic dimension scorer
# ---------------------------------------------------------------------------

def _score_dimensions(text: str) -> Dict[str, float]:
    """Simple heuristic dimension scores used ONLY for candidate-pair mining.

    These scores are never presented to the judge model and are not reported
    as evaluation results.  They are used only to identify candidate pairs
    where two responses differ on opposing dimensions, providing the raw
    material for a CPO-style tradeoff pair.

    Threshold note: the mining step requires a difference of ≥0.15 on each
    dimension (raised from 0.05 in the original draft).  This reduces false
    positives — pairs where the "tradeoff" is noise in the heuristic rather
    than a genuine quality difference.  The tradeoff is yield: fewer pairs
    are mined but each is more reliable.  The diagnostics file reports the
    yield at this threshold so it can be tuned if needed.
    """
    words = re.findall(r"\w+", text)
    n_words = max(len(words), 1)
    has_json = 1.0 if re.search(r"\{\s*\"?\w+\"?\s*:", text) else 0.0
    has_citation = (
        1.0
        if re.search(r"\[[A-Za-z0-9_\-]+\]|\(.*?\d{4}.*?\)|https?://", text)
        else 0.0
    )
    has_uncertainty = (
        1.0
        if re.search(
            r"\b(maybe|might|could|uncertain|not sure|likely)\b", text.lower()
        )
        else 0.0
    )
    has_confidence = (
        1.0
        if re.search(
            r"\b(definitely|certainly|clearly|without doubt|must)\b", text.lower()
        )
        else 0.0
    )
    return {
        "length": float(n_words),
        "concision": 1.0 / n_words,
        "informativeness": min(n_words / 120.0, 1.0),
        "format_compliance": has_json,
        "citation_compliance": has_citation,
        # Net confidence: high if confident language present, penalised by hedging
        "confidence": has_confidence - 0.2 * has_uncertainty,
    }


# Minimum absolute margin on each dimension required to call a pair a
# genuine tradeoff (raised from 0.05 to reduce noise).
_MINING_MARGIN_THRESHOLD = 0.15

# Candidate CPO families mined from WildBench heuristics.
# Each entry is (family_name, dim_that_c1_wins, dim_that_c2_wins).
_WILDBENCH_CANDIDATE_FAMILIES = [
    ("informativeness_vs_concision", "informativeness", "concision"),
    ("citation_vs_informativeness", "citation_compliance", "informativeness"),
    ("format_vs_informativeness", "format_compliance", "informativeness"),
    ("confidence_vs_caution", "confidence", "informativeness"),
]


def normalize_wildbench(
    raw_path: Path,
    out_path: Path,
    target_n: int,
    seed: int,
) -> int:
    """Mine naturalistic CPO-style pairs from WildBench.

    This miner is conservative by design:
    - A pair is emitted only when two responses have opposing heuristic scores
      above _MINING_MARGIN_THRESHOLD on their respective dimensions.
    - At most one pair is emitted per source row (break-on-first).
    - The metadata mining_note flag must be present on every output item.

    REVIEWER WARNING: Items produced here require manual verification before
    being used as ground truth.  The proposed workflow is:
      1. Run this miner to generate candidates.
      2. Annotate a random sample (~50 items) to estimate precision.
      3. Report annotation precision in the paper (Section 4 / Appendix).

    If precision is below ~0.80, discard WildBench Tier 2 or move to appendix.
    """
    rows = read_jsonl(raw_path)
    rng = random.Random(seed)
    items: List[CPOItem] = []
    diagnostics: Dict[str, int] = {
        "raw_rows": len(rows),
        "skipped_no_task": 0,
        "skipped_fewer_than_2_responses": 0,
        "pairs_emitted": 0,
        "rows_with_pair": 0,
    }

    for r in rows:
        task = _extract_task(r)
        if not task:
            diagnostics["skipped_no_task"] += 1
            continue

        responses = _find_responses_in_wildbench_row(r)
        if len(responses) < 2:
            diagnostics["skipped_fewer_than_2_responses"] += 1
            continue

        scored = [
            (name, text, meta, _score_dimensions(text))
            for name, text, meta in responses
        ]
        pair_made = False
        for a_idx in range(len(scored)):
            for b_idx in range(a_idx + 1, len(scored)):
                name1, text1, meta1, s1 = scored[a_idx]
                name2, text2, meta2, s2 = scored[b_idx]
                for fam, da, db in _WILDBENCH_CANDIDATE_FAMILIES:
                    diff_a = s1[da] - s2[da]   # positive: response1 wins dim A
                    diff_b = s1[db] - s2[db]   # negative: response2 wins dim B

                    if diff_a > _MINING_MARGIN_THRESHOLD and diff_b < -_MINING_MARGIN_THRESHOLD:
                        wa, wb = "candidate_1", "candidate_2"
                    elif diff_a < -_MINING_MARGIN_THRESHOLD and diff_b > _MINING_MARGIN_THRESHOLD:
                        wa, wb = "candidate_2", "candidate_1"
                    else:
                        continue

                    item = CPOItem(
                        item_id=f"tier2_wildbench_{len(items):05d}",
                        tier="tier2_wildbench",
                        source="wildbench_mined",
                        family=fam,
                        pair_type="conflict",
                        task=truncate_text(task),
                        criterion_a=da.replace("_", " "),
                        criterion_b=db.replace("_", " "),
                        candidate_1=truncate_text(text1),
                        candidate_2=truncate_text(text2),
                        winner_criterion_a=wa,
                        winner_criterion_b=wb,
                        metadata={
                            "response_1_name": name1,
                            "response_2_name": name2,
                            "source_path": r.get("_source_path", r.get("_hf_split", "")),
                            "checklists": _extract_checklists(r)[:10],
                            "scores_1": s1,
                            "scores_2": s2,
                            "diff_a": diff_a,
                            "diff_b": diff_b,
                            "margin_threshold": _MINING_MARGIN_THRESHOLD,
                            # This flag must be disclosed in the paper.
                            # The proposed methodology is: sample 50 items,
                            # annotate manually, report precision in the Appendix.
                            "mining_note": (
                                "HEURISTIC MINED: criterion winners assigned by "
                                f"simple text dimension scores (threshold={_MINING_MARGIN_THRESHOLD}). "
                                "Manual annotation required before treating as ground truth. "
                                "See proposal Section 14 / QC rules."
                            ),
                        },
                    )
                    items.append(item)
                    diagnostics["pairs_emitted"] += 1
                    pair_made = True
                    break  # one pair per row
                if pair_made:
                    break
            if pair_made:
                break

        if pair_made:
            diagnostics["rows_with_pair"] += 1

    rng.shuffle(items)
    if target_n > 0:
        items = items[:target_n]
    n = write_jsonl([x.to_dict() for x in items], out_path)
    diagnostics["written"] = n
    atomic_write_text(
        out_path.with_suffix(".diagnostics.json"),
        json.dumps(diagnostics, indent=2),
    )
    return n


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_benchmarks(paths: List[Path], out_path: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return write_jsonl(rows, out_path)
