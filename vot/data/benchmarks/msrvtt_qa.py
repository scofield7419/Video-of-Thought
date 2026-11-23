"""MSRVTT-QA (open-ended, zero-shot in the paper).
https://github.com/xudejing/video-question-answering

Raw files: ``{train,val,test}_qa.json`` -- a list of
``{"id", "video_id", "question", "answer", "category_id"}``; videos are
``video<video_id>.mp4``. The question type (what/who/how/when/where/...) is
taken from the first word of the question.
"""

from __future__ import annotations

from typing import List

from ..videoqa import VideoQASample
from ._common import RawPath, find_split_file, maybe_unified, read_json

NAME = "msrvtt_qa"
QUESTION_TYPES = ("what", "who", "how", "when", "where", "why", "which")


def question_type(question: str) -> str:
    first = question.strip().lower().split(" ", 1)[0] if question.strip() else ""
    return first if first in QUESTION_TYPES else "other"


def convert(raw: RawPath, split: str = "test") -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    path = find_split_file(raw, split, ["{split}_qa.json", "{split}.json"])
    samples: List[VideoQASample] = []
    for i, r in enumerate(read_json(path)):
        vid = str(r["video_id"])
        video = vid if vid.startswith("video") else f"video{vid}"
        q = str(r["question"]).strip()
        samples.append(
            VideoQASample(
                qid=str(r.get("id", i)),
                video=video,
                question=q,
                answer=str(r["answer"]).strip() if r.get("answer") is not None else None,
                category=question_type(q),
                benchmark=NAME,
                meta={"split": split, "category_id": r.get("category_id")},
            )
        )
    return samples
