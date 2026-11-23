"""Full-STSG parsing with a (video) LLM, the prompting analogue of MotionEpic's
L2 grounding objective ("given a video, generate the whole STSG")."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ..backends.base import GenerationConfig, Message, VideoLLM, VideoPart
from ..stsg.expression import STSG_FORMAT_SPEC, format_frame, parse_stsg
from ..stsg.schema import STSG, CorefLink
from ..video.io import VideoClip
from .base import STSGProvider

PARSE_PROMPT = (
    "Parse the spatial-temporal scene graph (STSG) of the given video frames.\n{spec}\n"
    "{focus}List at most {max_objects} salient objects per frame, and only predicates that are visible "
    "(spatial relations such as 'on', 'next to', 'in front of', and actions such as 'holding', 'riding', 'pouring')."
    "{previous}\nOutput only the STSG expression."
)


class VLMSTSGParser(STSGProvider):
    """Ask a VLM for the STSG of a clip, in chunks of ``chunk_size`` frames.

    Chunks are parsed sequentially; the last frame graph of the previous chunk is
    given as context so object ids stay consistent, and coreference edges across
    chunk borders are inferred from identical ids.
    """

    name = "vlm"

    def __init__(
        self,
        model: VideoLLM,
        chunk_size: int = 8,
        max_objects: int = 10,
        focus_on_targets: bool = True,
        max_new_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.chunk_size = chunk_size
        self.max_objects = max_objects
        self.focus_on_targets = focus_on_targets
        self.gen = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)

    def __call__(self, clip: VideoClip, *, targets: Optional[Sequence[str]] = None, question=None, video_id=None) -> STSG:
        n = len(clip)
        out = STSG(meta={"source": f"vlm:{self.model.describe().get('model', self.model.name)}", "video_id": video_id})
        prev_frame = None
        for s in range(0, n, self.chunk_size):
            idx = list(range(s, min(n, s + self.chunk_size)))
            sub = VideoClip([clip.frames[i] for i in idx], [clip.timestamps[i] for i in idx], clip.source, clip.native_fps, clip.duration)
            focus = ""
            if self.focus_on_targets and targets:
                focus = "Make sure to include " + ", ".join(f"[{t}]" for t in targets) + " and the objects interacting with them. "
            previous = ""
            if prev_frame is not None:
                pf = prev_frame.__class__(index=0, objects=prev_frame.objects, triplets=prev_frame.triplets)
                previous = (
                    "\nThe frame right before these frames was parsed as follows; reuse its object ids for the same objects:\n"
                    + format_frame(pf, compact=True).replace("Frame 0", "Previous frame")
                )
            spec = f"The {len(sub)} frames are numbered 1 to {len(sub)}.\n" + STSG_FORMAT_SPEC
            prompt = PARSE_PROMPT.format(spec=spec, focus=focus, max_objects=self.max_objects, previous=previous)
            r = self.model.generate([Message.user(VideoPart(sub), "\n" + prompt)], self.gen)
            g = parse_stsg(r.text, image_size=sub.size)
            for f in g.frames:
                if 1 <= f.index <= len(idx):
                    f.timestamp = sub.timestamps[f.index - 1]
                    f.index = idx[f.index - 1] + 1
                    out.frames.append(f)
            for c in g.coref:
                if 1 <= c.src_frame <= len(idx) and 1 <= c.dst_frame <= len(idx):
                    out.coref.append(CorefLink(idx[c.src_frame - 1] + 1, c.src, idx[c.dst_frame - 1] + 1, c.dst))
            chunk_frames: List = [f for f in out.frames if f.index - 1 in idx]
            prev_frame = chunk_frames[-1] if chunk_frames else prev_frame
        out.repair().infer_coref()
        return out.sort()
