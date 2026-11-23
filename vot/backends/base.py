"""Backend-agnostic multimodal chat interface.

The VoT reasoning engine talks to any video LLM through :class:`VideoLLM`.
A message is a list of parts: text, a sampled video (:class:`VideoPart`) or an
STSG (:class:`STSGPart`). Each backend decides how to realise the non-text
parts:

* ``MotionEpicBackend`` splices learned video tokens (ViT-L/14 + Q-Former) and
  STSG tokens (recurrent graph transformer) into the LLM input, exactly as in
  the paper;
* API backends (OpenAI-compatible, Anthropic) send the frames as images with
  "Frame k (t=...s)" captions and the STSG as its textual expression.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from ..stsg.schema import STSG
from ..video.io import VideoClip

Role = Literal["system", "user", "assistant"]


@dataclass
class TextPart:
    text: str


@dataclass
class VideoPart:
    clip: VideoClip


@dataclass
class STSGPart:
    """An STSG given to the model. ``label`` names it in the text rendering."""

    stsg: STSG
    label: str = "STSG"
    max_frames: Optional[int] = None


Part = Union[TextPart, VideoPart, STSGPart]


@dataclass
class Message:
    role: Role
    content: List[Part]

    @classmethod
    def system(cls, text: str) -> "Message":
        return cls("system", [TextPart(text)])

    @classmethod
    def user(cls, *parts: Union[str, Part]) -> "Message":
        return cls("user", [TextPart(p) if isinstance(p, str) else p for p in parts])

    @classmethod
    def assistant(cls, text: str) -> "Message":
        return cls("assistant", [TextPart(text)])

    def text(self) -> str:
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    def has_video(self) -> bool:
        return any(isinstance(p, VideoPart) for p in self.content)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None
    stop: Optional[List[str]] = None


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency: float = 0.0
    model: Optional[str] = None
    cached: bool = False
    raw: Any = field(default=None, repr=False)


@dataclass
class BackendCapabilities:
    #: the backend encodes STSGs natively (graph encoder) instead of as text
    native_stsg: bool = False
    #: the model was trained on the STSG expression grammar (no format spec needed)
    knows_stsg_format: bool = False
    #: maximum images per request (None = unlimited)
    max_images: Optional[int] = None


class UsageMeter:
    """Thread-safe accumulator of API usage across calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_calls = 0

    def add(self, r: LLMResponse) -> None:
        with self._lock:
            self.calls += 1
            self.cached_calls += int(r.cached)
            self.prompt_tokens += r.prompt_tokens
            self.completion_tokens += r.completion_tokens

    def as_dict(self) -> Dict[str, int]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class VideoLLM(ABC):
    """Abstract video LLM used by the VoT engine."""

    name: str = "base"
    capabilities: BackendCapabilities = BackendCapabilities()

    def __init__(self) -> None:
        self.usage = UsageMeter()

    @abstractmethod
    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        """Run one chat completion."""

    def generate(self, messages: Sequence[Message], gen: Optional[GenerationConfig] = None) -> LLMResponse:
        gen = gen or GenerationConfig()
        t0 = time.time()
        r = self._generate(list(messages), gen)
        if not r.latency:
            r.latency = time.time() - t0
        self.usage.add(r)
        return r

    def generate_many(
        self,
        batch: Sequence[Sequence[Message]],
        gen: Optional[GenerationConfig] = None,
        max_workers: int = 1,
    ) -> List[LLMResponse]:
        """Run independent conversations (e.g. per-candidate scoring in Step-4).

        API backends benefit from ``max_workers > 1``; local backends may
        override this with true batched decoding.
        """
        if max_workers <= 1 or len(batch) <= 1:
            return [self.generate(m, gen) for m in batch]
        with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as ex:
            return list(ex.map(lambda m: self.generate(m, gen), batch))

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name}


def render_stsg_text(part: STSGPart, with_spec: bool = False) -> str:
    """Textual rendering of an STSG part: ``<STSG> ... </STSG>``."""
    from ..stsg.expression import STSG_FORMAT_SPEC

    body = part.stsg.to_expression(max_frames=part.max_frames) if not part.stsg.is_empty else "(empty)"
    out = f"<{part.label}>\n{body}\n</{part.label}>"
    if with_spec:
        out = STSG_FORMAT_SPEC + "\n" + out
    return out
