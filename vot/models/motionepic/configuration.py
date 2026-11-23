"""MotionEpic configuration.

MotionEpic = Vicuna-7B (v1.5, frozen + LoRA) + CLIP ViT-L/14 video encoder (frozen)
+ video Q-Former projector (trainable) + recurrent graph-transformer STSG encoder
(trainable, 6 layers, 768-d). Defaults reproduce the paper (Sec. 3.1, Sec. 5).

The multimodal modules are configured separately from the LLM config: the LLM is
loaded from its own checkpoint and the multimodal modules are attached after
loading (see :mod:`vot.models.motionepic.builder`). A trained MotionEpic
checkpoint therefore only stores this config, the trainable multimodal weights
and the LoRA adapter.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

IGNORE_INDEX = -100
VIDEO_TOKEN_INDEX = -300
STSG_TOKEN_INDEX = -200
VIDEO_TOKEN = "<video>"
STSG_TOKEN = "<stsg>"

CONFIG_NAME = "motionepic_config.json"
TRAINABLES_NAME = "mm_trainables.safetensors"
LORA_DIRNAME = "lora"


@dataclass
class MotionEpicConfig:
    # ---- backbones
    llm_name_or_path: str = "lmsys/vicuna-7b-v1.5"
    vision_name_or_path: str = "openai/clip-vit-large-patch14"
    #: hidden layer whose patch tokens feed the Q-Former (LLaVA convention: penultimate)
    vision_select_layer: int = -2
    vision_select_feature: str = "patch"  # "patch" | "cls_patch"
    #: build the vision tower / text encoder from these configs instead of
    #: downloading (used by tests and for custom towers)
    vision_config: Optional[Dict[str, Any]] = None
    text_config: Optional[Dict[str, Any]] = None
    image_size: int = 224

    # ---- video sampling (paper: 8 fps)
    sample_fps: float = 8.0
    max_frames: int = 32

    # ---- video Q-Former projector
    qformer_hidden_size: int = 768
    qformer_num_heads: int = 12
    qformer_intermediate_size: int = 3072
    frame_qformer_layers: int = 12
    video_qformer_layers: int = 2
    num_frame_query_tokens: int = 32
    num_query_tokens: int = 32
    #: optional BLIP-2 checkpoint to initialise the frame-level Q-Former
    blip2_qformer_init: Optional[str] = None

    # ---- STSG encoder (recurrent graph transformer)
    #: dim of CLIP joint-space embeddings used for RoI features f_i, labels c_i and predicates
    stsg_feature_dim: int = 768
    stsg_hidden_size: int = 768
    stsg_num_layers: int = 6
    stsg_num_heads: int = 8
    stsg_ffn_mult: int = 2
    stsg_dropout: float = 0.1
    stsg_num_tokens: int = 32
    stsg_resampler_layers: int = 2
    max_stsg_frames: int = 32
    max_stsg_nodes: int = 24
    roi_min_size: int = 8
    #: "clip" (CLIP text tower) or "hash" (deterministic random vectors; offline tests)
    phrase_encoder: str = "clip"

    # ---- filled at attach time
    llm_hidden_size: Optional[int] = None

    extra: Dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- io
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotionEpicConfig":
        names = {f.name for f in fields(cls)}
        known = {k: v for k, v in d.items() if k in names}
        unknown = {k: v for k, v in d.items() if k not in names}
        cfg = cls(**known)
        cfg.extra.update(unknown)
        return cfg

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, CONFIG_NAME)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "MotionEpicConfig":
        if os.path.isdir(path):
            path = os.path.join(path, CONFIG_NAME)
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: str) -> "MotionEpicConfig":
        import yaml

        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f) or {})
