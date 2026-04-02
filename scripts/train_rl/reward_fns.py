"""Reward functions for GRPO medical training.

Three reward functions scored independently, combined via reward_weights:
  [0.15]  format_reward       — structured output compliance
  [0.70]  answer_reward       — exact / letter / substring / token-F1
  [0.15]  tool_quality_reward — tool usage quality
"""

import re
from collections import Counter

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Patterns for extracting a letter answer from free-form text
_LETTER_RE = re.compile(r"^\s*([A-Ea-e])[.\):\s]")  # "A." / "A)" / "A:" / "A "
_FINAL_LETTER_RE = re.compile(
    r"(?:answer\s+is|therefore|correct\s+(?:answer|option)[:\s]*)\s*"
    r"(?:\*\*)?([A-Ea-e])\b",
    re.IGNORECASE,
)


def _get_assistant_content(completion: list[dict]) -> str:
    """Concatenate all assistant message content from a multi-turn completion."""
    return " ".join(
        turn.get("content", "")
        for turn in completion
        if turn.get("role") == "assistant" and turn.get("content")
    )


def format_reward(completions, **kwargs) -> list[float]:
    """Reward for structured output: tool usage + <think>/<answer> tags.

    Scoring:
        +0.25  tool call issued AND tool response received
        +0.25  <think>...</think> present
        +0.50  <answer>...</answer> present (bonus +0.25 if answer is concise)
    """
    rewards = []
    for completion in completions:
        score = 0.0
        all_content = _get_assistant_content(completion)

        has_tool_call = any(t.get("tool_calls") for t in completion)
        has_tool_response = any(t.get("role") == "tool" for t in completion)
        if has_tool_call and has_tool_response:
            score += 0.25

        if _THINK_RE.search(all_content):
            score += 0.25

        answer_match = _ANSWER_RE.search(all_content)
        if answer_match:
            score += 0.50
            # Bonus for concise MCQ-style answers (<=10 words)
            answer_text = answer_match.group(1).strip()
            if len(answer_text.split()) <= 10:
                score += 0.25

        rewards.append(score)
    return rewards


def _token_f1(pred: str, gt: str) -> float:
    pred_tokens = pred.lower().split()
    gt_tokens = gt.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_reward(completions, answer, answer_idx=None, **kwargs) -> list[float]:
    """MCQ-aware reward: exact > letter > in-text letter > substring > token F1.

    Args:
        completions: list of completions (each is list[dict] of conversation turns).
        answer:      list[str] ground-truth full answer text.
        answer_idx:  list[str] correct option letter (e.g. "A","B","C","D").
                     Passed automatically when 'answer_idx' column is in the dataset.
    """
    gt_indices = answer_idx if answer_idx is not None else [None] * len(answer)
    rewards = []

    for completion, gt, gt_idx in zip(completions, answer, gt_indices):
        all_content = _get_assistant_content(completion)

        # Extract prediction from <answer> tag; fallback to last assistant turn
        match = _ANSWER_RE.search(all_content)
        if match:
            pred = match.group(1).strip()
        else:
            pred = ""
            for turn in reversed(completion):
                if turn.get("role") == "assistant" and turn.get("content"):
                    pred = turn["content"].strip()
                    break

        if not pred:
            rewards.append(0.0)
            continue

        # 1. Exact text match (case-insensitive)
        if pred.lower() == gt.lower():
            rewards.append(1.0)
            continue

        # 2. Letter match — model outputs "A" / "A." / "A)" and it's the right letter
        if gt_idx:
            gt_letter = gt_idx.strip().upper()

            letter_match = _LETTER_RE.match(pred)
            if letter_match and letter_match.group(1).upper() == gt_letter:
                rewards.append(1.0)
                continue
            # Bare single-letter answer: "A"
            if pred.strip().upper() == gt_letter:
                rewards.append(1.0)
                continue

            # 3. In-text letter — "the answer is A", "therefore A", "correct option: A"
            final_letter = _FINAL_LETTER_RE.search(pred)
            if final_letter and final_letter.group(1).upper() == gt_letter:
                rewards.append(0.8)
                continue

        # 4. Substring match — GT text appears within the prediction
        if gt.lower() in pred.lower():
            rewards.append(0.5)
            continue

        # 5. Token F1 fallback (partial credit)
        rewards.append(_token_f1(pred, gt))

    return rewards


def tool_quality_reward(completions, **kwargs) -> list[float]:
    """Reward for appropriate tool usage frequency.

    Scoring:
        0 calls  -> -0.1  (mild penalty, not catastrophic)
        1-2 calls -> +0.3 (good usage)
        3+ calls  ->  0.0 (neutral — don't punish exploration)
    """
    rewards = []
    for completion in completions:
        n_calls = sum(1 for turn in completion if turn.get("tool_calls"))
        if n_calls == 0:
            rewards.append(-0.1)
        elif n_calls <= 2:
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards
