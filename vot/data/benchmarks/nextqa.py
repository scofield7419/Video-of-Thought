"""NExT-QA (multiple choice, 5 options). https://github.com/doc-doc/NExT-QA

Raw files (official release): ``{train,val,test}.csv`` with columns
``video, frame_count, width, height, question, answer, qid, type, a0..a4`` and
``map_vid_vidorID.json`` mapping a video id to ``<folder>/<vidor id>`` inside
``NExTVideo/``. Question types: CW/CH (causal), TN/TC/TP (temporal),
DL/DC/DO (descriptive).

``raw`` may be the csv itself or the directory holding it; if
``map_vid_vidorID.json`` is found next to the csv (or passed as ``video_map``)
the ``video`` field becomes the relative path ``<folder>/<vidor id>``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from ..videoqa import VideoQASample
from ._common import RawPath, find_split_file, maybe_unified, read_csv, read_json

NAME = "nextqa"
TYPE_GROUPS: Dict[str, str] = {
    "CW": "Causal", "CH": "Causal",
    "TN": "Temporal", "TC": "Temporal", "TP": "Temporal",
    "DL": "Descriptive", "DC": "Descriptive", "DO": "Descriptive",
}


def convert_nextqa_style(
    raw: RawPath,
    split: str,
    benchmark: str,
    video_map: Optional[str] = None,
    num_options: int = 5,
) -> List[VideoQASample]:
    unified = maybe_unified(raw, benchmark)
    if unified is not None:
        return unified
    path = find_split_file(raw, split, ["{split}.csv", "nextqa/{split}.csv", "{split}_qa.csv"])
    if video_map is None:
        cand = os.path.join(os.path.dirname(path), "map_vid_vidorID.json")
        video_map = cand if os.path.isfile(cand) else None
    vmap: Dict[str, str] = {str(k): str(v) for k, v in read_json(video_map).items()} if video_map else {}

    samples: List[VideoQASample] = []
    for r in read_csv(path):
        vid = str(r.get("video") or r.get("video_id") or r.get("vid")).strip()
        cands = [str(r[f"a{i}"]).strip() for i in range(num_options) if f"a{i}" in r and r[f"a{i}"] is not None]
        ans = r.get("answer")
        answer = int(ans) if ans not in (None, "") and str(ans).strip().lstrip("-").isdigit() else None
        qtype = (r.get("type") or "").strip() or None
        samples.append(
            VideoQASample(
                qid=f"{vid}_{r.get('qid', len(samples))}",
                video=vmap.get(vid, vid),
                question=str(r["question"]).strip(),
                candidates=cands,
                answer=answer,
                category=qtype,
                benchmark=benchmark,
                meta={"video_id": vid, "split": split},
            )
        )
    return samples


def convert(raw: RawPath, split: str = "val", video_map: Optional[str] = None) -> List[VideoQASample]:
    return convert_nextqa_style(raw, split, NAME, video_map=video_map)
