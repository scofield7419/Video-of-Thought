"""Video-of-Thought: step-by-step video reasoning from perception to cognition (ICML 2024).

Top-level API::

    from vot import VideoOfThought, VoTConfig, build_backend
"""

from .backends import build_backend
from .reasoning import DirectAnswer, ReasoningResult, VanillaCoT, VideoOfThought, VoTConfig
from .stsg import STSG, FrameGraph, SGObject, Triplet, parse_stsg
from .video import VideoClip, load_video

__version__ = "1.0.0"

__all__ = [
    "STSG",
    "DirectAnswer",
    "FrameGraph",
    "ReasoningResult",
    "SGObject",
    "Triplet",
    "VanillaCoT",
    "VideoClip",
    "VideoOfThought",
    "VoTConfig",
    "build_backend",
    "load_video",
    "parse_stsg",
    "__version__",
]
