"""ActivityNet-QA (open-ended, zero-shot in the paper). https://github.com/MILVLG/activitynet-qa

Raw files: ``dataset/{split}_q.json`` (``[{"video_name", "question_id", "question"}]``)
and ``dataset/{split}_a.json`` (``[{"question_id", "answer", "type"}]``).
``raw`` may be the directory containing them, the ``dataset/`` folder, or the
two file paths ``[q_json, a_json]``. ActivityNet videos are usually stored as
``v_<youtube id>.mp4``; both spellings are recorded in ``meta["video_alts"]``
for :func:`vot.data.videoqa.resolve_video`.

Answer types (ActivityNet-QA paper): 0 motion, 1 spatial relation, 2 temporal
relation, 3 yes/no, 4 color, 5 object, 6 location, 7 number, 8 other.
"""

from __future__ import annotations

import os
from typing import Dict, List

from ..videoqa import VideoQASample
from ._common import RawPath, as_list, maybe_unified, read_json

NAME = "activitynet_qa"
ANSWER_TYPES: Dict[int, str] = {
    0: "motion", 1: "spatial", 2: "temporal", 3: "yes_no", 4: "color",
    5: "object", 6: "location", 7: "number", 8: "other",
}


def _find_pair(raw: RawPath, split: str):
    paths = as_list(raw)
    if len(paths) == 2 and all(os.path.isfile(p) for p in paths):
        q = next((p for p in paths if "_q" in os.path.basename(p)), paths[0])
        return q, paths[1] if q == paths[0] else paths[0]
    for p in paths:
        for base in (p, os.path.join(p, "dataset")):
            q = os.path.join(base, f"{split}_q.json")
            if os.path.isfile(q):
                a = os.path.join(base, f"{split}_a.json")
                return q, (a if os.path.isfile(a) else None)
    raise FileNotFoundError(f"ActivityNet-QA {split}_q.json not found in {paths}")


def convert(raw: RawPath, split: str = "test") -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    q_path, a_path = _find_pair(raw, split)
    answers = {str(a["question_id"]): a for a in read_json(a_path)} if a_path else {}
    samples: List[VideoQASample] = []
    for q in read_json(q_path):
        qid = str(q["question_id"])
        name = str(q["video_name"])
        alts = [name[2:]] if name.startswith("v_") else ["v_" + name]
        a = answers.get(qid, {})
        atype = a.get("type")
        samples.append(
            VideoQASample(
                qid=qid,
                video=name,
                question=str(q["question"]).strip(),
                answer=str(a["answer"]).strip() if "answer" in a else None,
                category=ANSWER_TYPES.get(int(atype), str(atype)) if atype is not None else None,
                benchmark=NAME,
                meta={"split": split, "video_alts": alts},
            )
        )
    return samples
