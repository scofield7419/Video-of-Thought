"""MotionEpic: fine-grained spatial-temporal grounded video MLLM (paper Sec. 3)."""

from .configuration import (
    IGNORE_INDEX,
    STSG_TOKEN,
    STSG_TOKEN_INDEX,
    VIDEO_TOKEN,
    VIDEO_TOKEN_INDEX,
    MotionEpicConfig,
)
from .conversation import Conversation, vicuna_v1
from .featurizer import STSGBatch, STSGFeaturizer, STSGInputs

__all__ = [
    "IGNORE_INDEX",
    "STSG_TOKEN",
    "STSG_TOKEN_INDEX",
    "VIDEO_TOKEN",
    "VIDEO_TOKEN_INDEX",
    "Conversation",
    "MotionEpicConfig",
    "STSGBatch",
    "STSGFeaturizer",
    "STSGInputs",
    "vicuna_v1",
    # lazily importable (need transformers): MotionEpicForCausalLM, MotionEpicProcessor,
    # build_motionepic, load_motionepic, save_motionepic, apply_lora
    "MotionEpicForCausalLM",
    "MotionEpicProcessor",
    "build_motionepic",
    "load_motionepic",
    "save_motionepic",
    "apply_lora",
]


def __getattr__(name):
    if name == "MotionEpicForCausalLM":
        from .modeling import MotionEpicForCausalLM

        return MotionEpicForCausalLM
    if name == "MotionEpicProcessor":
        from .processor import MotionEpicProcessor

        return MotionEpicProcessor
    if name in {"build_motionepic", "load_motionepic", "save_motionepic", "apply_lora"}:
        from . import builder

        return getattr(builder, name)
    raise AttributeError(name)
