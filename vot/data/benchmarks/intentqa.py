"""IntentQA (multiple choice, 5 options). https://github.com/JoannaLiXY/IntentQA

Assumed raw format: NExT-QA-style ``{train,val,test}.csv`` with columns
``video_id`` (or ``video``), ``question``, ``answer``, ``qid``, ``type``,
``a0..a4``. Question types ``CW`` (why), ``CH`` (how) and ``TP``/``TN``/
``TP&TN`` (before/after) are kept as ``category``; reports group them into
Why / How / Before&After.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..videoqa import VideoQASample
from ._common import RawPath
from .nextqa import convert_nextqa_style

NAME = "intentqa"
TYPE_GROUPS: Dict[str, str] = {"CW": "Why", "CH": "How", "TP": "Before&After", "TN": "Before&After", "TP&TN": "Before&After"}


def convert(raw: RawPath, split: str = "test", video_map: Optional[str] = None) -> List[VideoQASample]:
    return convert_nextqa_style(raw, split, NAME, video_map=video_map)
