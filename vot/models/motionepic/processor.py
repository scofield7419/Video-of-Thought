"""Text/video/STSG preprocessing for MotionEpic.

``<video>`` and ``<stsg>`` placeholders become negative sentinel ids
(:data:`VIDEO_TOKEN_INDEX`, :data:`STSG_TOKEN_INDEX`) instead of new vocabulary
entries; the model replaces each sentinel by the projected video tokens or
STSG tokens. This keeps the LLM vocabulary (and LoRA-only training) untouched.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...video.io import VideoClip
from .configuration import IGNORE_INDEX, STSG_TOKEN, STSG_TOKEN_INDEX, VIDEO_TOKEN, VIDEO_TOKEN_INDEX, MotionEpicConfig
from .conversation import Conversation
from .featurizer import STSGFeaturizer, square_pad

_PLACEHOLDER_RE = re.compile("(" + re.escape(VIDEO_TOKEN) + "|" + re.escape(STSG_TOKEN) + ")")


def default_image_processor(image_size: int = 224):
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor(
        size={"shortest_edge": image_size},
        crop_size={"height": image_size, "width": image_size},
    )


class MotionEpicProcessor:
    def __init__(self, tokenizer, image_processor, cfg: MotionEpicConfig, model_max_length: int = 4096) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cfg = cfg
        self.model_max_length = model_max_length
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
        self.featurizer = STSGFeaturizer(
            image_processor=image_processor,
            max_frames=cfg.max_stsg_frames,
            max_nodes=cfg.max_stsg_nodes,
            roi_min_size=cfg.roi_min_size,
            default_fps=cfg.sample_fps,
        )

    @classmethod
    def from_config(cls, cfg: MotionEpicConfig, tokenizer=None, image_processor=None, model_max_length: int = 4096) -> "MotionEpicProcessor":
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name_or_path, use_fast=False)
        if image_processor is None:
            image_processor = default_image_processor(cfg.image_size)
        return cls(tokenizer, image_processor, cfg, model_max_length)

    # ------------------------------------------------------------------ text
    def tokenize(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in _PLACEHOLDER_RE.split(text):
            if piece == VIDEO_TOKEN:
                ids.append(VIDEO_TOKEN_INDEX)
            elif piece == STSG_TOKEN:
                ids.append(STSG_TOKEN_INDEX)
            elif piece.strip():
                ids.extend(self.tokenizer(piece.strip("\n"), add_special_tokens=False)["input_ids"])
        return ids

    def encode_conversation(self, conv: Conversation) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token ids and SFT labels (instructions masked with ``IGNORE_INDEX``)."""
        ids: List[int] = []
        labels: List[int] = []
        if self.tokenizer.bos_token_id is not None:
            ids.append(self.tokenizer.bos_token_id)
            labels.append(IGNORE_INDEX)
        eos = self.tokenizer.eos_token_id
        for seg in conv.segments():
            toks = self.tokenize(seg.text)
            ids.extend(toks)
            labels.extend(toks if seg.target else [IGNORE_INDEX] * len(toks))
            if seg.eos and eos is not None:
                ids.append(eos)
                labels.append(eos if seg.target else IGNORE_INDEX)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def encode_prompt(self, conv: Conversation) -> torch.Tensor:
        return self.encode_conversation(conv)[0]

    def pad(self, seqs: Sequence[torch.Tensor], value: int, side: str = "right") -> torch.Tensor:
        n = max(len(s) for s in seqs)
        out = torch.full((len(seqs), n), value, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            if side == "right":
                out[i, : len(s)] = s
            else:
                out[i, n - len(s) :] = s
        return out

    # ----------------------------------------------------------------- video
    def process_video(self, clip: VideoClip) -> torch.Tensor:
        """``VideoClip -> [T, 3, H, W]`` (subsampled to ``cfg.max_frames``).

        Frames are padded to a square first: CLIP's resize + centre crop would
        otherwise cut the frame borders, while STSG boxes cover the full frame.
        """
        clip = clip.subsample(self.cfg.max_frames)
        frames = [square_pad(f.convert("RGB")) for f in clip.frames]
        return self.image_processor(images=frames, return_tensors="pt")["pixel_values"]

    @staticmethod
    def collate_videos(videos: Sequence[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not videos:
            return None, None
        t = max(v.shape[0] for v in videos)
        out = torch.zeros(len(videos), t, *videos[0].shape[1:], dtype=videos[0].dtype)
        mask = torch.zeros(len(videos), t, dtype=torch.bool)
        for i, v in enumerate(videos):
            out[i, : v.shape[0]] = v
            mask[i, : v.shape[0]] = True
        return out, mask

    # ------------------------------------------------------------- inference
    def build_inputs(
        self,
        conv: Conversation,
        clips: Sequence[VideoClip] = (),
        stsgs: Sequence[Tuple[object, Optional[VideoClip]]] = (),
    ) -> Dict[str, object]:
        """Single-example model inputs for ``MotionEpicForCausalLM.generate``.

        ``clips`` and ``stsgs`` are consumed in the order their placeholders
        appear; each STSG is paired with the clip its boxes refer to.
        """
        ids = self.encode_prompt(conv)
        n_vid = int((ids == VIDEO_TOKEN_INDEX).sum())
        n_sg = int((ids == STSG_TOKEN_INDEX).sum())
        if n_vid != len(clips) or n_sg != len(stsgs):
            raise ValueError(f"prompt has {n_vid} <video>/{n_sg} <stsg> placeholders but got {len(clips)} clips/{len(stsgs)} STSGs")
        videos, video_mask = self.collate_videos([self.process_video(c) for c in clips])
        stsg = self.featurizer.collate([self.featurizer.encode(g, c) for g, c in stsgs]) if stsgs else None
        return {
            "input_ids": ids[None],
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "videos": videos,
            "video_mask": video_mask,
            "stsg": stsg,
        }
