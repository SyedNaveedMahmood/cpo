from __future__ import annotations

from typing import Dict, List

from .schema import CPOItem


DIRECT_PARAPHRASES = [
    """Evaluation rubric:\n1. The most important criterion is {primary}.\n2. The secondary criterion is {secondary}.\n3. If the two answers conflict, choose the answer that better satisfies the most important criterion.\n4. Reply with only one letter: A or B.""",
    """Evaluation rule:\nFor this comparison, prioritize {primary} over {secondary}.\nThe lower-priority criterion should only break ties.\nChoose the answer that best follows this priority order.\nReply with only one letter: A or B.""",
    """Rubric priority:\n{primary} outranks {secondary}.\nWhen these criteria trade off, the higher-priority criterion must determine the winner.\nReturn exactly A or B.""",
]

LOCKED_PARAPHRASES = [
    """Evaluation rubric:\nThe absolute priority is {primary}. The lower-priority criterion is {secondary}.\nDo not use your own judgment about which criterion should matter more.\nEven if {secondary} seems more naturally important, ignore it when it conflicts with {primary}.\nReply with only one letter: A or B.""",
    """Locked priority rule:\n{primary} is binding and overrides {secondary}.\nYou must not substitute a different value hierarchy.\nIf the candidates trade off, choose the candidate that wins {primary}.\nReply with only one letter: A or B.""",
]

CONFLICT_AUDIT_PARAPHRASES = [
    """Evaluation rubric:\nFirst internally identify which candidate better satisfies {primary}.\nThen internally identify which candidate better satisfies {secondary}.\nIf they differ, choose the candidate that wins {primary}, because it is higher priority.\nReply with only one letter: A or B.""",
    """Conflict audit rule:\nDetermine which answer wins each criterion, but only the winner of {primary} should decide the final answer when there is a tradeoff with {secondary}.\nReply with only one letter: A or B.""",
]


def priority_criteria(item: CPOItem, priority: str) -> tuple[str, str]:
    if priority == "a_over_b":
        return item.criterion_a, item.criterion_b
    if priority == "b_over_a":
        return item.criterion_b, item.criterion_a
    raise ValueError(f"Unknown priority: {priority}")


def ordered_candidates(item: CPOItem, order: str) -> Dict[str, object]:
    if order == "canonical":
        return {
            "A": item.candidate_1,
            "B": item.candidate_2,
            "label_to_candidate": {"A": "candidate_1", "B": "candidate_2"},
            "candidate_to_label": {"candidate_1": "A", "candidate_2": "B"},
        }
    if order == "swapped":
        return {
            "A": item.candidate_2,
            "B": item.candidate_1,
            "label_to_candidate": {"A": "candidate_2", "B": "candidate_1"},
            "candidate_to_label": {"candidate_1": "B", "candidate_2": "A"},
        }
    raise ValueError(f"Unknown order: {order}")


def build_judge_user_prompt(task: str, ans_a: str, ans_b: str, primary: str, secondary: str, method: str, paraphrase_id: int) -> str:
    if method == "direct":
        templates = DIRECT_PARAPHRASES
    elif method == "locked":
        templates = LOCKED_PARAPHRASES
    elif method == "conflict_audit":
        templates = CONFLICT_AUDIT_PARAPHRASES
    else:
        raise ValueError(f"Unsupported prompt method: {method}")
    rubric = templates[paraphrase_id % len(templates)].format(primary=primary, secondary=secondary)
    return f"""Task:
{task}

Answer A:
{ans_a}

Answer B:
{ans_b}

{rubric}"""


def build_single_criterion_prompt(task: str, ans_a: str, ans_b: str, criterion: str) -> str:
    return f"""Task:
{task}

Answer A:
{ans_a}

Answer B:
{ans_b}

Evaluate ONLY this criterion:
{criterion}

Ignore all other criteria. Which answer better satisfies this single criterion?

Reply with only one letter: A or B."""


def build_context_forward_prompt(task: str, ans_a: str, ans_b: str, primary: str, secondary: str, primary_winner_label: str, secondary_winner_label: str) -> str:
    return f"""Task:
{task}

Answer A:
{ans_a}

Answer B:
{ans_b}

Structured intermediate findings:
- Winner on the higher-priority criterion ({primary}): Answer {primary_winner_label}
- Winner on the lower-priority criterion ({secondary}): Answer {secondary_winner_label}

The final decision must follow the higher-priority criterion, not the lower-priority criterion.

Which answer should win overall? Reply with only one letter: A or B."""


def build_preflight_prompt(question: str, ans_a: str, ans_b: str) -> str:
    return f"""Question:
{question}

Answer A:
{ans_a}

Answer B:
{ans_b}

Which answer is better or correct?
Reply with only one letter: A or B."""


def chat_format(tokenizer, user_text: str, system_text: str = "You are an impartial evaluator.") -> str:
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return system_text + "\n\n" + user_text + "\n\nAnswer:"


def paraphrase_count(method: str, requested: int) -> int:
    if method == "direct":
        return min(requested, len(DIRECT_PARAPHRASES))
    if method == "locked":
        return min(requested, len(LOCKED_PARAPHRASES))
    if method == "conflict_audit":
        return min(requested, len(CONFLICT_AUDIT_PARAPHRASES))
    return 1
