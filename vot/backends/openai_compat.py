"""OpenAI-compatible chat-completions backend.

Covers the OpenAI API (GPT-4o / GPT-4.1 / GPT-5 / o-series) and every server
that speaks the same protocol, e.g. vLLM / SGLang / LMDeploy serving Qwen2.5-VL,
InternVL or LLaVA-Video, DashScope, or Gemini's OpenAI-compatible endpoint
(set ``base_url`` and ``api_key``).

Videos are sent as a sequence of JPEG frames, each preceded by a
"Frame k (t=..s)" caption so that the model can refer to frame numbers in STSG
expressions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from ..video.io import encode_image
from .base import (
    BackendCapabilities,
    GenerationConfig,
    LLMResponse,
    Message,
    STSGPart,
    TextPart,
    VideoLLM,
    VideoPart,
    render_stsg_text,
)

_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAICompatibleBackend(VideoLLM):
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        image_detail: str = "auto",
        jpeg_quality: int = 85,
        max_images: Optional[int] = None,
        send_sampling_params: Optional[bool] = None,
        token_param: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        min_completion_tokens: Optional[int] = None,
        timeout: float = 600.0,
        max_retries: int = 5,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'openai>=1.40' (or `pip install -e .[api]`)") from e
        self.model = model
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.base_url = base_url
        self.image_detail = image_detail
        self.jpeg_quality = jpeg_quality
        is_reasoning = model.split("/")[-1].startswith(_REASONING_PREFIXES)
        # reasoning models reject non-default temperature/top_p and use max_completion_tokens
        self.send_sampling_params = (not is_reasoning) if send_sampling_params is None else send_sampling_params
        self.token_param = token_param or ("max_completion_tokens" if is_reasoning else "max_tokens")
        # reasoning tokens count against the completion budget; a small cap yields empty answers
        self.min_completion_tokens = (16000 if is_reasoning else 0) if min_completion_tokens is None else min_completion_tokens
        self.reasoning_effort = reasoning_effort
        self.extra_body = extra_body or {}
        self.capabilities = BackendCapabilities(native_stsg=False, knows_stsg_format=False, max_images=max_images)

    def _video_content(self, part: VideoPart) -> List[Dict[str, Any]]:
        clip = part.clip
        if self.capabilities.max_images and len(clip) > self.capabilities.max_images:
            clip = clip.subsample(self.capabilities.max_images)
        out: List[Dict[str, Any]] = []
        for k, (img, t) in enumerate(zip(clip.frames, clip.timestamps), start=1):
            out.append({"type": "text", "text": f"Frame {k} (t={t:.2f}s):"})
            out.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encode_image(img, quality=self.jpeg_quality),
                        "detail": self.image_detail,
                    },
                }
            )
        return out

    def to_api_messages(self, messages: Sequence[Message]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m.role != "user" or all(isinstance(p, TextPart) for p in m.content):
                text = "".join(p.text for p in m.content if isinstance(p, TextPart))
                out.append({"role": m.role, "content": text if text.strip() else "(no output)"})
                continue
            content: List[Dict[str, Any]] = []
            for p in m.content:
                if isinstance(p, TextPart):
                    if p.text:
                        content.append({"type": "text", "text": p.text})
                elif isinstance(p, VideoPart):
                    content.extend(self._video_content(p))
                elif isinstance(p, STSGPart):
                    content.append({"type": "text", "text": render_stsg_text(p)})
            out.append({"role": "user", "content": content})
        return out

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": self.to_api_messages(messages)}
        kwargs[self.token_param] = max(gen.max_new_tokens, self.min_completion_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if self.send_sampling_params:
            kwargs["temperature"] = gen.temperature
            kwargs["top_p"] = gen.top_p
        if gen.seed is not None:
            kwargs["seed"] = gen.seed
        if gen.stop:
            kwargs["stop"] = gen.stop
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            model=getattr(resp, "model", self.model),
            raw={"finish_reason": choice.finish_reason, "id": getattr(resp, "id", None)},
        )

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model, "base_url": self.base_url, "detail": self.image_detail}
