"""VLEP (video-and-language future event prediction, 2 options).
https://github.com/jayleicn/VideoLanguageFuturePred

Raw files: ``vlep_{train,dev,test}_release.jsonl`` with records
``{"example_id", "vid_name", "ts": [st, ed], "events": [e0, e1], "answer": 0|1}``.
``ts`` delimits the premise segment; the model must predict which event is
more likely to happen next. VLEP has no question text, so a fixed question
(``QUESTION``) is used.
"""

from __future__ import annotations

from typing import List

from ..videoqa import VideoQASample
from ._common import RawPath, find_split_file, maybe_unified, read_jsonl, to_float

NAME = "vlep"
QUESTION = "Which event is more likely to happen next in the video?"


def convert(raw: RawPath, split: str = "dev") -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    path = find_split_file(raw, split, ["vlep_{split}_release.jsonl", "{split}.jsonl"])
    samples: List[VideoQASample] = []
    for r in read_jsonl(path):
        ts = r.get("ts") or [None, None]
        ans = r.get("answer")
        samples.append(
            VideoQASample(
                qid=str(r["example_id"]),
                video=str(r["vid_name"]),
                question=QUESTION,
                candidates=[str(e).strip() for e in r["events"]],
                answer=int(ans) if ans is not None else None,
                start=to_float(ts[0]),
                end=to_float(ts[1]),
                benchmark=NAME,
                meta={"split": split},
            )
        )
    return samples
