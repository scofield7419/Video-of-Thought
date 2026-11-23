"""Social-IQ (2-way and 4-way multiple choice).

Social-IQ 1.0 (used in the paper) is distributed through the CMU Multimodal
SDK, whose raw QA files are not a simple table. Convert them first into the
intermediate JSONL below (one line per question)::

    {"qid": "xyz_q3", "video": "xyz", "question": "...",
     "correct": ["...", ...], "incorrect": ["...", ...], "start": null, "end": null}

From each question this converter enumerates the answer sets used by the
Social-IQ protocol:

* 2-way: every (correct, incorrect) pair;
* 4-way: every correct answer combined with 3 incorrect answers (all
  combinations of the incorrect set).

``max_per_question`` caps the number of items per question and way (a
deterministic subset is kept). Option order is shuffled deterministically
from the item id, so results do not depend on ``PYTHONHASHSEED``.

The Social-IQ 2.0 JSONL format (``qid, q, a0..a3, answer_idx, vid_name, ts``)
is also accepted and yields 4-way items directly.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional, Sequence

from ..videoqa import VideoQASample
from ._common import RawPath, find_split_file, maybe_unified, read_records, stable_shuffle, to_float

NAME = "socialiq"


def _subset(items: List[Any], k: Optional[int], *seed: Any) -> List[Any]:
    if k is None or len(items) <= k:
        return items
    keep = sorted(stable_shuffle(range(len(items)), *seed)[:k])
    return [items[i] for i in keep]


def _make_items(rec: Dict[str, Any], ways: Sequence[int], max_per_question: Optional[int], split: str) -> List[VideoQASample]:
    qid = str(rec["qid"])
    correct = [str(a).strip() for a in rec.get("correct", [])]
    incorrect = [str(a).strip() for a in rec.get("incorrect", [])]
    out: List[VideoQASample] = []
    for way in ways:
        n_wrong = way - 1
        if len(incorrect) < n_wrong or not correct:
            continue
        combos = [(c, list(w)) for c in correct for w in itertools.combinations(incorrect, n_wrong)]
        combos = _subset(combos, max_per_question, qid, way)
        for k, (c, wrong) in enumerate(combos):
            item_id = f"{qid}_{way}way_{k}"
            options = stable_shuffle([c] + wrong, item_id)
            out.append(
                VideoQASample(
                    qid=item_id,
                    video=str(rec["video"]),
                    question=str(rec["question"]).strip(),
                    candidates=options,
                    answer=options.index(c),
                    category=f"{way}-way",
                    start=to_float(rec.get("start")),
                    end=to_float(rec.get("end")),
                    benchmark=NAME,
                    meta={"question_id": qid, "split": split},
                )
            )
    return out


def _from_siq2(rec: Dict[str, Any], split: str) -> VideoQASample:
    cands = [str(rec[f"a{i}"]).strip() for i in range(4) if f"a{i}" in rec]
    ts = rec.get("ts")
    start = end = None
    if isinstance(ts, (list, tuple)) and len(ts) == 2:
        start, end = to_float(ts[0]), to_float(ts[1])
    ans = rec.get("answer_idx")
    return VideoQASample(
        qid=str(rec["qid"]),
        video=str(rec.get("vid_name") or rec.get("video")),
        question=str(rec["q"]).strip(),
        candidates=cands,
        answer=int(ans) if ans is not None else None,
        category="4-way",
        start=start,
        end=end,
        benchmark=NAME,
        meta={"split": split, "format": "siq2"},
    )


def convert(
    raw: RawPath,
    split: str = "val",
    ways: Sequence[int] = (2, 4),
    max_per_question: Optional[int] = None,
) -> List[VideoQASample]:
    unified = maybe_unified(raw, NAME)
    if unified is not None:
        return unified
    path = find_split_file(raw, split, ["socialiq_{split}.jsonl", "{split}.jsonl", "qa_{split}.json", "{split}.json"])
    ways = tuple(int(w) for w in ways)
    samples: List[VideoQASample] = []
    for rec in read_records(path):
        if "correct" in rec:
            samples.extend(_make_items(rec, ways, max_per_question, split))
        elif "a0" in rec and "q" in rec:
            samples.append(_from_siq2(rec, split))
        else:
            raise ValueError(f"Unrecognized Social-IQ record keys: {sorted(rec)}")
    return samples
