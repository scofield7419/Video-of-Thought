"""STSG providers: where the scene graph of a video comes from.

MotionEpic parses (partial) STSGs itself, so VoT with MotionEpic runs
STSG-free. General-purpose API models are weak at precise pixel grounding,
therefore the API version can optionally be given an explicit STSG that Step-2
selects from (``step2_mode="select"``) or matches against (``"match"``):

* :class:`StaticSTSGProvider` - pre-computed / annotated graphs (e.g. Action Genome);
* :class:`vot.grounding.vlm_parser.VLMSTSGParser` - the VLM parses the full STSG
  first (the analogue of MotionEpic's L2 objective);
* :class:`vot.grounding.detector_parser.DetectorSTSGParser` - open-vocabulary
  detection + IoU tracking + geometric predicates.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Union

from ..stsg.schema import STSG
from ..video.io import VideoClip


class STSGProvider(ABC):
    name: str = "provider"

    @abstractmethod
    def __call__(
        self,
        clip: VideoClip,
        *,
        targets: Optional[Sequence[str]] = None,
        question: Optional[str] = None,
        video_id: Optional[str] = None,
    ) -> STSG:
        """Return the STSG of ``clip``; frame ``k`` must correspond to ``clip.frame(k)``."""


class StaticSTSGProvider(STSGProvider):
    """Serve pre-computed STSGs from a dict or a directory of ``<video_id>.json``.

    The stored graph must use the same frame numbering as the sampled clip; if it
    was built at another rate, set ``resample=True`` to keep only the frames whose
    ``timestamp`` is closest to the clip's timestamps (frames are then renumbered).
    """

    name = "static"

    def __init__(self, source: Union[str, Dict[str, STSG]], resample: bool = False) -> None:
        self.source = source
        self.resample = resample

    def _load(self, video_id: str) -> Optional[STSG]:
        if isinstance(self.source, dict):
            return self.source.get(video_id)
        path = os.path.join(self.source, f"{video_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return STSG.from_dict(json.load(f))

    def __call__(self, clip, *, targets=None, question=None, video_id=None) -> STSG:
        if video_id is None:
            raise ValueError("StaticSTSGProvider needs a video_id")
        g = self._load(video_id)
        if g is None:
            return STSG(meta={"video_id": video_id, "missing": True})
        if self.resample and all(f.timestamp is not None for f in g.frames) and g.frames:
            return resample_to_clip(g, clip)
        return g


def resample_to_clip(g: STSG, clip: VideoClip) -> STSG:
    """Pick, for every clip frame, the STSG frame nearest in time and renumber 1..N."""
    from ..stsg.schema import CorefLink

    frames = sorted(g.frames, key=lambda f: f.timestamp)  # type: ignore[arg-type,return-value]
    picked = []
    for t in clip.timestamps:
        picked.append(min(frames, key=lambda f: abs(f.timestamp - t)).index)  # type: ignore[operator]
    sub = g.select_frames(set(picked))
    remap = {}
    out = STSG(meta=dict(g.meta))
    for k, (t, idx) in enumerate(zip(clip.timestamps, picked), start=1):
        f = sub.frame(idx)
        if f is None or idx in remap:
            continue
        remap[idx] = k
        f2 = f.__class__(index=k, objects=list(f.objects), triplets=list(f.triplets), timestamp=t)
        out.frames.append(f2)
    out.coref = [CorefLink(remap[c.src_frame], c.src, remap[c.dst_frame], c.dst) for c in sub.coref if c.src_frame in remap and c.dst_frame in remap]
    return out
