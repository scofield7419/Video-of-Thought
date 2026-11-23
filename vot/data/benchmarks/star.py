"""STAR (situated reasoning, 4 options). https://github.com/csbobby/STAR_Benchmark

Raw files: ``STAR_{train,val,test}.json`` -- a list of records with
``question_id`` (e.g. ``Interaction_T1_13``), ``video_id`` (Charades id),
``question``, ``answer`` (text of the correct choice; absent on test),
``choices`` (``[{"choice_id": 0, "choice": "..."}, ...]``), ``start`` and
``end`` (seconds). The category is the ``question_id`` prefix: Interaction,
Sequence, Prediction or Feasibility.
"""

from __future__ import annotations

from typing import List

from ..videoqa import VideoQASample
from ._common import RawPath, find_split_file, maybe_unified, read_json, to_float

NAME = "star"
CATEGORIES = ("Interaction", "Sequence", "Prediction", "Feasibility")


def _norm(s: str) -> str:
    return " ".join(str(s).lower().strip().rstrip(".").split())


def convert(raw: RawPath, split: str = "val") -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    path = find_split_file(raw, split, ["STAR_{split}.json", "{split}.json"])
    samples: List[VideoQASample] = []
    for r in read_json(path):
        choices = sorted(r.get("choices", []), key=lambda c: int(c.get("choice_id", 0)))
        cands = [str(c["choice"]).strip() for c in choices]
        answer = None
        if r.get("answer") not in (None, ""):
            gold = _norm(r["answer"])
            matches = [i for i, c in enumerate(cands) if _norm(c) == gold]
            answer = matches[0] if matches else None
        qid = str(r["question_id"])
        cat = qid.split("_")[0]
        samples.append(
            VideoQASample(
                qid=qid,
                video=str(r["video_id"]),
                question=str(r["question"]).strip(),
                candidates=cands,
                answer=answer,
                category=cat if cat in CATEGORIES else None,
                start=to_float(r.get("start")),
                end=to_float(r.get("end")),
                benchmark=NAME,
                meta={"split": split},
            )
        )
    return samples
