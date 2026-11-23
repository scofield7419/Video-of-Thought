"""Unified VideoQA sample format shared by all benchmarks.

Every benchmark converter (:mod:`vot.data.benchmarks`) produces
:class:`VideoQASample` objects that are stored as one JSON object per line::

    {"qid": "4010069381_6", "video": "0004/4010069381", "question": "why did the boy ...",
     "candidates": ["...", "...", "...", "...", "..."], "answer": 2, "category": "CW",
     "start": null, "end": null, "benchmark": "nextqa", "meta": {}}

``answer`` is an option index for multiple-choice QA, a list of indices for
multi-answer QA and a string for open-ended QA (``None`` for unlabeled test sets).
"""

from __future__ import annotations

import functools
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v")
OPTION_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

Answer = Union[int, str, List[int], None]


@dataclass
class VideoQASample:
    qid: str
    video: str
    question: str
    candidates: Optional[List[str]] = None
    answer: Answer = None
    category: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    benchmark: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multi_choice(self) -> bool:
        return bool(self.candidates)

    @property
    def is_multi_answer(self) -> bool:
        return isinstance(self.answer, list)

    @property
    def has_answer(self) -> bool:
        return self.answer is not None

    @property
    def answer_text(self) -> Union[str, List[str], None]:
        """Gold answer as text (option text for MC, list of texts for multi-answer)."""
        if self.answer is None:
            return None
        if not self.is_multi_choice:
            return str(self.answer)
        assert self.candidates is not None
        if isinstance(self.answer, list):
            return [self.candidates[i] for i in self.answer]
        return self.candidates[int(self.answer)]

    @property
    def answer_letter(self) -> Union[str, List[str], None]:
        if self.answer is None or not self.is_multi_choice:
            return None
        if isinstance(self.answer, list):
            return [OPTION_LETTERS[i] for i in self.answer]
        return OPTION_LETTERS[int(self.answer)]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoQASample":
        ans = d.get("answer")
        cands = d.get("candidates")
        if cands is not None:
            cands = [str(c) for c in cands]
            if isinstance(ans, str) and ans.strip().lstrip("-").isdigit():
                ans = int(ans)
        return cls(
            qid=str(d["qid"]),
            video=str(d["video"]),
            question=str(d["question"]),
            candidates=cands,
            answer=ans,
            category=d.get("category"),
            start=_opt_float(d.get("start")),
            end=_opt_float(d.get("end")),
            benchmark=d.get("benchmark"),
            meta=dict(d.get("meta") or {}),
        )


def _opt_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    return float(v)


def load_samples(path: str) -> List[VideoQASample]:
    """Read the unified format (``.jsonl``, or a ``.json`` list)."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            data = json.load(f)
            return [VideoQASample.from_dict(d) for d in data]
        return [VideoQASample.from_dict(json.loads(line)) for line in f if line.strip()]


def save_samples(samples: Iterable[VideoQASample], path: str) -> int:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n


def is_unified_file(path: Union[str, Sequence[str]]) -> bool:
    """True for a unified-format file: a ``.jsonl`` whose first record has
    ``qid``/``video``/``question`` and no keys outside :class:`VideoQASample`."""
    if not isinstance(path, str) or not path.endswith(".jsonl") or not os.path.isfile(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    return False
                return isinstance(rec, dict) and {"qid", "video", "question"} <= set(rec) <= _FIELDS
    return False


_FIELDS = {"qid", "video", "question", "candidates", "answer", "category", "start", "end", "benchmark", "meta"}


@functools.lru_cache(maxsize=16)
def _index_videos(root: str, exts: tuple) -> Dict[str, str]:
    """Map ``stem`` and ``relative/path/stem`` to file paths below ``root`` (first hit wins)."""
    index: Dict[str, str] = {}
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            stem, ext = os.path.splitext(fn)
            if ext.lower() not in exts:
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.splitext(os.path.relpath(full, root))[0]
            index.setdefault(stem, full)
            index.setdefault(rel.replace(os.sep, "/"), full)
    return index


def resolve_video(
    sample: Union[VideoQASample, str],
    video_root: Optional[str] = None,
    exts: Sequence[str] = VIDEO_EXTS,
    search: bool = True,
) -> Optional[str]:
    """Locate the video file (or frame directory) of a sample.

    Tries, for the sample's ``video`` and any ``meta["video_alts"]`` names:
    the path itself, ``video_root/<name>``, both with each extension appended,
    and finally (``search=True``) a recursive lookup of the file stem below
    ``video_root``. Returns ``None`` if nothing is found.
    """
    if isinstance(sample, VideoQASample):
        names = [sample.video] + [str(a) for a in sample.meta.get("video_alts", [])]
    else:
        names = [str(sample)]
    exts = tuple(e.lower() for e in exts)
    for name in names:
        bases = [name] if (os.path.isabs(name) or video_root is None) else [os.path.join(video_root, name), name]
        for base in bases:
            if os.path.isfile(base):
                return base
            for ext in exts:
                if os.path.isfile(base + ext):
                    return base + ext
            if os.path.isdir(base):  # directory of extracted frames
                return base
    if search and video_root and os.path.isdir(video_root):
        index = _index_videos(os.path.abspath(video_root), exts)
        for name in names:
            key = os.path.splitext(name)[0] if os.path.splitext(name)[1].lower() in exts else name
            for k in (key.replace(os.sep, "/"), os.path.basename(key)):
                if k in index:
                    return index[k]
    return None
