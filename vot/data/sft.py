"""Supervised fine-tuning data for MotionEpic (all training stages).

One record per sample (JSON list or JSONL)::

    {
      "id": "001YG-L3",
      "video": "videos/001YG.mp4",              # optional: video file
      "frames": ["001YG.mp4/000089.png", ...],  # optional: exact frames to show
      "frame_times": [3.0, 3.5, ...],           # optional: exact timestamps to sample
      "start": 2.0, "end": 9.5,                 # optional: segment for uniform sampling
      "stsg": {...} | "stsgs/001YG.json",       # optional: STSG for <stsg> placeholders
      "stsg_video"/"stsg_frames"/"stsg_frame_times": ...,  # optional: frames the STSG boxes refer to
      "system": "...",                          # optional system prompt
      "conversations": [{"from": "human", "value": "<video>\\n..."}, {"from": "gpt", "value": "..."}]
    }

Frame selection priority: ``frames`` > ``frame_times`` > uniform sampling at
``sample_fps`` (paper: 8 fps) capped by ``max_frames``. Grounding records use
explicit frames so that "Frame k" of the STSG targets is the k-th frame shown.

Stages: (1) WebVid caption pretraining (projector only), (2) grounding-aware
tuning on L1-L5 (projector + STSG encoder + LoRA), (3) video instruction tuning
(VideoChat / Video-ChatGPT; same trainables), (4) optional downstream finetuning.
All use this format; only the record files and trainable flags differ.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..models.motionepic.configuration import IGNORE_INDEX, STSG_TOKEN, STSG_TOKEN_INDEX, VIDEO_TOKEN, VIDEO_TOKEN_INDEX
from ..models.motionepic.conversation import vicuna_v1
from ..models.motionepic.featurizer import TIMESTAMPS_RELIABLE
from ..stsg.expression import parse_stsg
from ..stsg.schema import STSG
from ..video.io import VideoClip, load_video

logger = logging.getLogger(__name__)


def load_records(paths: Union[str, Sequence[str]]) -> List[Dict[str, Any]]:
    if isinstance(paths, str):
        paths = [p for p in paths.split(",") if p]
    out: List[Dict[str, Any]] = []
    for p in paths:
        with open(p) as f:
            if p.endswith(".jsonl"):
                out.extend(json.loads(line) for line in f if line.strip())
            else:
                data = json.load(f)
                out.extend(data if isinstance(data, list) else [data])
    return out


def load_video_at_times(path: str, times: Sequence[float]) -> VideoClip:
    """Decode the frames closest to ``times`` (seconds)."""
    from ..video import io as vio

    times = list(times)
    errors = []
    for fn in (getattr(vio, "_decord", None), getattr(vio, "_pyav", None), getattr(vio, "_opencv", None)):
        if fn is None:
            continue
        try:
            return fn(path, lambda duration: [min(max(0.0, t), max(0.0, duration - 1e-3)) for t in times])
        except ImportError as e:
            errors.append(str(e))
    raise RuntimeError(f"No video decoder available for {path}: {errors}")


def _join(root: Optional[str], p: str) -> str:
    return p if (root is None or os.path.isabs(p)) else os.path.join(root, p)


class SFTDataset(Dataset):
    def __init__(
        self,
        records: Union[str, Sequence[str], List[Dict[str, Any]]],
        processor,
        video_root: Optional[str] = None,
        frames_root: Optional[str] = None,
        stsg_root: Optional[str] = None,
        sample_fps: float = 8.0,
        max_frames: int = 32,
        system: Optional[str] = None,
        max_retries: int = 10,
    ) -> None:
        self.records = records if isinstance(records, list) and (not records or isinstance(records[0], dict)) else load_records(records)  # type: ignore[arg-type]
        self.processor = processor
        self.video_root = video_root
        self.frames_root = frames_root
        self.stsg_root = stsg_root
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self.system = system
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.records)

    @property
    def lengths(self) -> List[int]:
        """Rough token lengths (for length-grouped sampling)."""
        return [sum(len(c["value"].split()) for c in r["conversations"]) + (32 if "video" in r or "frames" in r else 0) for r in self.records]

    # ----------------------------------------------------------- loading
    def _clip(self, rec: Dict[str, Any], prefix: str = "") -> Optional[VideoClip]:
        frames = rec.get(prefix + "frames")
        times = rec.get(prefix + "frame_times")
        video = rec.get(prefix + "video")
        if frames:
            imgs = [Image.open(_join(self.frames_root, p)).convert("RGB") for p in frames]
            if times:
                return VideoClip(imgs, list(times), source=frames[0])
            # nominal times only: the featurizer must align STSG frames by index, not by time
            ts = [i / self.sample_fps for i in range(len(imgs))]
            return VideoClip(imgs, ts, source=frames[0], meta={TIMESTAMPS_RELIABLE: False})
        if video and times:
            return load_video_at_times(_join(self.video_root, video), times)
        if video:
            return load_video(
                _join(self.video_root, video),
                fps=self.sample_fps,
                max_frames=self.max_frames,
                start=rec.get("start"),
                end=rec.get("end"),
            )
        return None

    def _stsg(self, rec: Dict[str, Any]) -> STSG:
        s = rec["stsg"]
        if isinstance(s, dict):
            return STSG.from_dict(s)
        if isinstance(s, str) and s.strip().startswith("Frame"):
            return parse_stsg(s)
        with open(_join(self.stsg_root, s)) as f:
            return STSG.from_dict(json.load(f))

    def _conversation(self, rec: Dict[str, Any], has_video: bool):
        conv = vicuna_v1(rec.get("system", self.system))
        turns = rec["conversations"]
        if has_video and not any(VIDEO_TOKEN in t["value"] for t in turns):
            turns = [dict(turns[0], value=VIDEO_TOKEN + "\n" + turns[0]["value"])] + list(turns[1:])
        for t in turns:
            role = conv.roles[0] if t["from"] in ("human", "user") else conv.roles[1]
            conv.append(role, t["value"])
        return conv

    def build(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        clip = self._clip(rec)
        conv = self._conversation(rec, clip is not None)
        input_ids, labels = self.processor.encode_conversation(conv)
        n_vid = int((input_ids == VIDEO_TOKEN_INDEX).sum())
        n_sg = int((input_ids == STSG_TOKEN_INDEX).sum())
        if n_vid and clip is None:
            raise ValueError(f"{rec.get('id')}: {VIDEO_TOKEN} without video")
        videos = [self.processor.process_video(clip)] * n_vid
        stsgs = []
        if n_sg:
            if "stsg" not in rec:
                raise ValueError(f"{rec.get('id')}: {STSG_TOKEN} without stsg")
            sclip = self._clip(rec, "stsg_") if any(k in rec for k in ("stsg_frames", "stsg_video", "stsg_frame_times")) else clip
            stsgs = [self.processor.featurizer.encode(self._stsg(rec), sclip)] * n_sg
        return {"input_ids": input_ids, "labels": labels, "videos": videos, "stsgs": stsgs, "id": rec.get("id")}

    def __getitem__(self, i: int) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                return self.build(self.records[i])
            except Exception as e:  # corrupted videos should not stop a long run
                if attempt + 1 == self.max_retries:
                    raise
                logger.warning("Skipping record %s: %s", self.records[i].get("id", i), e)
                i = random.randrange(len(self.records))
        raise RuntimeError("unreachable")


@dataclass
class SFTCollator:
    """Pads a batch and enforces ``max_length`` on the *spliced* sequence.

    Each ``<video>`` / ``<stsg>`` placeholder expands to the projector's / STSG
    encoder's token count inside the model, so the text budget is reduced
    accordingly. A reply cut by truncation is masked (the model must never learn
    to stop mid-expression without ``</s>``); samples left without any
    supervised token are dropped from the batch.
    """

    processor: Any
    max_length: Optional[int] = None

    def _expansion(self, ids: torch.Tensor) -> int:
        cfg = self.processor.cfg
        return int((ids == VIDEO_TOKEN_INDEX).sum()) * (cfg.num_query_tokens - 1) + int(
            (ids == STSG_TOKEN_INDEX).sum()
        ) * (cfg.stsg_num_tokens - 1)

    def truncate(self, ids: torch.Tensor, labels: torch.Tensor):
        if not self.max_length or len(ids) + self._expansion(ids) <= self.max_length:
            return ids, labels
        cut = max(0, self.max_length - self._expansion(ids))
        cut = max(0, self.max_length - self._expansion(ids[:cut]))
        ids, kept = ids[:cut], labels[:cut].clone()
        if cut < len(labels) and labels[cut] != IGNORE_INDEX:
            i = cut - 1
            while i >= 0 and kept[i] != IGNORE_INDEX:
                kept[i] = IGNORE_INDEX
                i -= 1
        return ids, kept

    def __call__(self, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        cut = [self.truncate(x["input_ids"], x["labels"]) for x in items]
        keep = [i for i, (_, lab) in enumerate(cut) if bool((lab != IGNORE_INDEX).any())]
        if len(keep) < len(items):
            logger.warning("Dropping %d sample(s) without supervision after truncation", len(items) - len(keep))
        if not keep:  # the model returns a zero loss for an all-masked batch
            keep = [0]
        items = [items[i] for i in keep]
        ids = [cut[i][0] for i in keep]
        labels = [cut[i][1] for i in keep]
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = self.processor.pad(ids, pad_id)
        attention_mask = self.processor.pad([torch.ones_like(s) for s in ids], 0)
        batch: Dict[str, Any] = {
            "input_ids": input_ids,
            "labels": self.processor.pad(labels, IGNORE_INDEX),
            "attention_mask": attention_mask,
        }
        videos, stsgs = [], []
        for x, s in zip(items, ids):  # keep only placeholders that survived truncation
            videos.extend(x["videos"][: int((s == VIDEO_TOKEN_INDEX).sum())])
            stsgs.extend(x["stsgs"][: int((s == STSG_TOKEN_INDEX).sum())])
        if videos:
            batch["videos"], batch["video_mask"] = self.processor.collate_videos(videos)
        if stsgs:
            batch["stsg"] = self.processor.featurizer.collate(stsgs)
        return batch
