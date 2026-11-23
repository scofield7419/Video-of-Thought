"""Causal-VidQA (4 options). https://github.com/bcmi/Causal-VidQA

Assumed raw layout (official release)::

    QA/<video_id>/text.json     {"descriptive":    {"question": q, "answer": [a0..a3]},
                                 "explanatory":    {"question": q, "answer": [a0..a3]},
                                 "predictive":     {"question": q, "answer": [a0..a3], "reason": [r0..r3]},
                                 "counterfactual": {"question": q, "answer": [a0..a3], "reason": [r0..r3]}}
    QA/<video_id>/answer.json   {"descriptive": {"answer": i}, "explanatory": {"answer": i},
                                 "predictive": {"answer": i, "reason": j}, "counterfactual": {"answer": i, "reason": j}}
    split/{train,val,test}.pkl  list of video ids (``.txt``/``.json`` lists also accepted)

Predictive and counterfactual questions have an *answer* and a *reason*
sub-question. Both become separate samples sharing ``meta["group"]`` so that
:mod:`vot.eval.metrics` can compute A, R and AR (both correct) accuracies.
By default the reason question does not reveal the gold answer
(``reason_with_answer=False``); set it to ``True`` for the oracle-answer setting.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..videoqa import VideoQASample
from ._common import RawPath, as_list, maybe_unified, read_id_list, read_json

NAME = "causalvidqa"
QTYPES = ("descriptive", "explanatory", "predictive", "counterfactual")
QTYPE_CODE: Dict[str, str] = {"descriptive": "D", "explanatory": "E", "predictive": "P", "counterfactual": "C"}
REASON_QUESTION = "{question} What is the most plausible reason?"
REASON_QUESTION_WITH_ANSWER = "{question} The answer is: {answer}. What is the reason?"


def _qa_root(raw: RawPath) -> str:
    for p in as_list(raw):
        if os.path.isdir(os.path.join(p, "QA")):
            return os.path.join(p, "QA")
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Causal-VidQA QA directory not found in {raw}")


def _split_ids(raw: RawPath, split: str, split_file: Optional[str]) -> Optional[List[str]]:
    if split_file:
        return read_id_list(split_file)
    for p in as_list(raw):
        for ext in (".pkl", ".txt", ".json"):
            cand = os.path.join(p, "split", split + ext)
            if os.path.isfile(cand):
                return read_id_list(cand)
    return None


def convert(
    raw: RawPath,
    split: str = "test",
    split_file: Optional[str] = None,
    reason_with_answer: bool = False,
) -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    root = _qa_root(raw)
    ids = _split_ids(raw, split, split_file)
    if ids is None:
        ids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))

    samples: List[VideoQASample] = []
    for vid in ids:
        text_path = os.path.join(root, vid, "text.json")
        if not os.path.isfile(text_path):
            continue
        text = read_json(text_path)
        ans_path = os.path.join(root, vid, "answer.json")
        gold = read_json(ans_path) if os.path.isfile(ans_path) else {}
        for qtype in QTYPES:
            if qtype not in text:
                continue
            item = text[qtype]
            question = str(item["question"]).strip()
            cands = [str(a).strip() for a in item["answer"]]
            g = gold.get(qtype, {})
            a_idx = int(g["answer"]) if "answer" in g else None
            has_reason = "reason" in item
            group = f"{vid}_{qtype}"
            samples.append(
                VideoQASample(
                    qid=f"{group}_answer" if has_reason else group,
                    video=vid,
                    question=question,
                    candidates=cands,
                    answer=a_idx,
                    category=QTYPE_CODE[qtype],
                    benchmark=NAME,
                    meta={"qtype": qtype, "part": "answer", "group": group, "split": split},
                )
            )
            if has_reason:
                if reason_with_answer and a_idx is not None:
                    rq = REASON_QUESTION_WITH_ANSWER.format(question=question, answer=cands[a_idx])
                else:
                    rq = REASON_QUESTION.format(question=question)
                samples.append(
                    VideoQASample(
                        qid=f"{group}_reason",
                        video=vid,
                        question=rq,
                        candidates=[str(r).strip() for r in item["reason"]],
                        answer=int(g["reason"]) if "reason" in g else None,
                        category=QTYPE_CODE[qtype],
                        benchmark=NAME,
                        meta={"qtype": qtype, "part": "reason", "group": group, "split": split},
                    )
                )
    return samples
