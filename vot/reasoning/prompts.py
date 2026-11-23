"""Prompt templates of the five VoT steps (paper Sec. 4 and Appendix A.1).

The wording of every template follows the paper. Two kinds of additions are
kept separate so they can be switched off (``PromptTemplates.format_hints``):

* *format hints*: one sentence asking for the output shape the paper shows in
  its I/O examples (e.g. "The involved targets are [..], [..]"), which makes
  outputs of general-purpose API models parseable;
* the STSG grammar (:data:`vot.stsg.STSG_FORMAT_SPEC`), given only to models
  that were not trained on STSG expressions (i.e. not MotionEpic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class PromptTemplates:
    # ---- task definition (system prompt), Appendix A.1 Step-1
    task_mc: str = (
        "Now you are an expert in analyzing video data, and you should answer a question based on the given video.\n"
        "For the question, several candidate answers are provided, where you need to choose {choice_scope}."
    )
    choice_scope_single: str = "the most suitable option"
    choice_scope_multi: str = "all possible correct option(s)"
    task_oe: str = (
        "Now you are an expert in analyzing video data, and you should answer a question based on the given video.\n"
        "For the question, you should answer in an open-ended format."
    )

    # ---- Step-1: target identification
    step1: str = "Given the question: [{question}], what are the possible targets of the [Video] mainly mentioned or involved?"
    step1_hint: str = "Answer in the form: The involved targets are [target 1], [target 2], ..."

    # ---- Step-2: object tracking
    step2: str = (
        "Provide the tracklet of involved {targets} by outputting the corresponding partial expression in the [STSG]."
    )
    step2_hint: str = (
        "Answer in the form: The partial STSG in tracking {targets} is: <the partial STSG expression>."
    )
    step2_with_reference: str = (
        "The full [STSG] of the video is given above as reference; select the nodes, triplets and temporal "
        "coreference edges relevant to the involved targets and their neighbor scenes, correcting obvious errors."
    )

    # ---- Step-3: action analyzing
    step3: str = (
        "Combining all possible related commonsense, analyze the motion behavior based on the {targets} and the "
        "neighbor scenes within [STSG]. Describing the action observations and implications."
    )
    step3_hint: str = (
        "First describe the observations grounded in the STSG (who does what, when, and how the targets move and "
        "interact), then their implications."
    )

    # ---- Step-4-Pre: open-ended -> multi-choice
    step4_pre: str = (
        "For the question [{question}], please based on the action's [{observation}] combined with commonsense, "
        "output {k} distinct optional answers with the rationality score of this answer with a 1-10 scale."
    )
    step4_pre_hint: str = (
        "Use exactly one line per answer in the form: Answer A: <answer> | Score: <1-10> | Rationale: <short rationale>"
    )

    # ---- Step-4-A: per-candidate scoring
    step4_score: str = (
        "For the question [{question}], given a candidate answer [{candidate}], please based on the action's "
        "[{observation}] combined with commonsense, score the rationality of this answer with a 1-10 scale, and "
        "also output the rationale."
    )
    step4_score_hint: str = (
        "Begin your response with: Based on the observations and commonsense, the rationality of the answer "
        "[{candidate}] is <score>."
    )
    step4_feedback: str = (
        "Note that the previous answer [{answer}] was rejected by verification for the following reason: {reason}"
    )

    # ---- Step-4-B: ranking
    step4_rank: str = (
        "Now, we know {score_list}. Please rank the candidate the answer based on the rationale score of each "
        "candidate's answer."
    )
    step4_rank_item: str = "the rationale score of the answer [{candidate}] is {score}"
    step4_rank_hint: str = (
        "Answer in the form: Based on the rationality, the ranking results in descending order is [..] [..] ... "
        "So, the final answer is [..]."
    )

    # ---- Step-5: verification
    step5_head: str = (
        "Given the [Video], and the raw question [{question}], now you need to verify the previous answer by"
    )
    step5_perception: str = (
        "checking the pixel grounding information if the answer [{answer}] aligns with the facts presented in the "
        "video from a perception standpoint;"
    )
    step5_cognition: str = (
        "determining from a cognition perspective if the commonsense implications inherent in the answer "
        "[{answer}] contradict any of the main [{observation}] inferred in the 3-rd reasoning step."
    )
    step5_tail: str = "Output the verification result with rationale."
    step5_hint: str = "End your response with exactly one line: Verdict: CORRECT or Verdict: INCORRECT"

    # ---- baselines
    direct: str = "Question: {question}\nAnswer with the option's letter from the given choices directly."
    direct_oe: str = "Question: {question}\nAnswer the question using a short phrase."
    cot: str = "Question: {question}\nLet's think step by step."
    cot_hint_mc: str = "Finish with one line in the form: Final answer: <letter>."
    cot_hint_oe: str = "Finish with one line in the form: Final answer: <short answer>."

    format_hints: bool = True

    # ------------------------------------------------------------ helpers
    def task_definition(self, multi_choice: bool, multi_answer: bool = False) -> str:
        if not multi_choice:
            return self.task_oe
        scope = self.choice_scope_multi if multi_answer else self.choice_scope_single
        return self.task_mc.format(choice_scope=scope)

    def hint(self, text: str, **kw: str) -> str:
        return ("\n" + text.format(**kw)) if self.format_hints else ""


def format_options(candidates: Sequence[str], sep: str = "  ") -> str:
    return sep.join(f"{OPTION_LETTERS[i]}. {c}" for i, c in enumerate(candidates))


def format_question(question: str, candidates: Optional[Sequence[str]]) -> str:
    """``"What ...? A. x  B. y"`` as in the paper's examples."""
    q = question.strip()
    return f"{q} {format_options(candidates)}" if candidates else q


def format_candidate(candidates: Sequence[str], i: int) -> str:
    return f"{OPTION_LETTERS[i]}. {candidates[i]}"


def format_targets(targets: Sequence[str]) -> str:
    """``[a] and [b]`` / ``[a], [b] and [c]``."""
    items = [f"[{t}]" for t in targets]
    if not items:
        return "[the main objects]"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def truncate(text: str, max_chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= max_chars else text[: max_chars - 3].rstrip() + "..."


def wrap(tag: str, text: str) -> str:
    """``<Tag>\\ntext\\n</Tag>`` blocks used in the Appendix I/O format."""
    return f"<{tag}>\n{text}\n</{tag}>"


def letters(n: int) -> List[str]:
    return list(OPTION_LETTERS[:n])
