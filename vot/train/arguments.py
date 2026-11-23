"""Command-line / YAML arguments for MotionEpic training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelArguments:
    llm_name_or_path: str = field(default="lmsys/vicuna-7b-v1.5", metadata={"help": "Backbone LLM (Vicuna-7B v1.5 in the paper)."})
    vision_name_or_path: str = field(default="openai/clip-vit-large-patch14", metadata={"help": "CLIP ViT-L/14."})
    motionepic_config: Optional[str] = field(default=None, metadata={"help": "YAML/JSON with MotionEpicConfig overrides."})
    init_from: Optional[str] = field(default=None, metadata={"help": "Checkpoint of the previous stage (mm weights [+ LoRA])."})
    blip2_qformer_init: Optional[str] = field(default=None, metadata={"help": "BLIP-2 checkpoint for the frame Q-Former (stage 1)."})
    train_projector: bool = True
    train_stsg_encoder: bool = False
    lora_enable: bool = False
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    mm_lr: Optional[float] = field(default=None, metadata={"help": "Learning rate of projector / STSG encoder (default: learning_rate)."})
    attn_implementation: Optional[str] = field(default=None, metadata={"help": "e.g. flash_attention_2 or sdpa."})
    model_max_length: int = 2048


@dataclass
class DataArguments:
    data_path: List[str] = field(default_factory=list, metadata={"help": "SFT JSON/JSONL files (vot.data.sft format)."})
    video_root: Optional[str] = None
    frames_root: Optional[str] = None
    stsg_root: Optional[str] = None
    sample_fps: float = field(default=8.0, metadata={"help": "Uniform sampling rate (paper: 8 fps)."})
    max_frames: int = 32
    system_prompt: Optional[str] = None
    max_samples: Optional[int] = None
