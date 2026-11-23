"""The Video-of-Thought reasoning framework and baselines."""

from .baselines import DirectAnswer, VanillaCoT
from .config import VoTConfig
from .prompts import PromptTemplates
from .result import ReasoningResult, StepRecord
from .vot import VideoOfThought

METHODS = {"vot": VideoOfThought, "cot": VanillaCoT, "direct": DirectAnswer}

__all__ = [
    "METHODS",
    "DirectAnswer",
    "PromptTemplates",
    "ReasoningResult",
    "StepRecord",
    "VanillaCoT",
    "VideoOfThought",
    "VoTConfig",
]
