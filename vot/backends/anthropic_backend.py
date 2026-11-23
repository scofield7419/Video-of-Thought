"""Anthropic (Claude) backend via the official ``anthropic`` SDK.

Notes on current Claude models (Opus 5.5 / Sonnet 5.5 and newer):

* sampling parameters (``temperature``/``top_p``) are rejected, so
  :class:`GenerationConfig` sampling fields are not sent; outputs are therefore
  not bit-reproducible across runs (use :class:`~vot.backends.cache.CachedLLM`);
* thinking is adaptive and always on; depth is controlled by ``effort``.
  ``max_tokens`` covers thinking + answer, so it is floored at ``min_max_tokens``;
* the video frames are the large, repeated prefix of Steps 1-3 and 5, so a
  prompt-cache breakpoint is placed on the last frame of the first video.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

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

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicBackend(VideoLLM):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        effort: Optional[str] = "medium",
        min_max_tokens: int = 16000,
        fallbacks: Union[str, List[Dict[str, str]], None] = "default",
        cache_video: bool = True,
        jpeg_quality: int = 85,
        max_images: int = 100,
        timeout: float = 600.0,
        max_retries: int = 5,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install anthropic (or `pip install -e .[api]`)") from e
        kw: Dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if api_key:
            kw["api_key"] = api_key
        if base_url:
            kw["base_url"] = base_url
        self.client = anthropic.Anthropic(**kw)
        self.model = model
        self.effort = effort
        self.min_max_tokens = min_max_tokens
        self.fallbacks = fallbacks
        self.cache_video = cache_video
        self.jpeg_quality = jpeg_quality
        self.capabilities = BackendCapabilities(native_stsg=False, knows_stsg_format=False, max_images=max_images)

    def _video_blocks(self, part: VideoPart, cache: bool) -> List[Dict[str, Any]]:
        clip = part.clip
        if self.capabilities.max_images and len(clip) > self.capabilities.max_images:
            clip = clip.subsample(self.capabilities.max_images)
        blocks: List[Dict[str, Any]] = []
        for k, (img, t) in enumerate(zip(clip.frames, clip.timestamps), start=1):
            blocks.append({"type": "text", "text": f"Frame {k} (t={t:.2f}s):"})
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": encode_image(img, quality=self.jpeg_quality)},
                }
            )
        if cache and blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def to_api(self, messages: Sequence[Message]) -> Dict[str, Any]:
        system = "\n".join(m.text() for m in messages if m.role == "system")
        out: List[Dict[str, Any]] = []
        cached = False
        for m in messages:
            if m.role == "system":
                continue
            content: List[Dict[str, Any]] = []
            for p in m.content:
                if isinstance(p, TextPart):
                    if p.text:
                        content.append({"type": "text", "text": p.text})
                elif isinstance(p, VideoPart):
                    content.extend(self._video_blocks(p, self.cache_video and not cached))
                    cached = True
                elif isinstance(p, STSGPart):
                    content.append({"type": "text", "text": render_stsg_text(p)})
            if not content:  # e.g. an empty (refused) assistant turn recycled in chain mode
                content = [{"type": "text", "text": "(no output)"}]
            if out and out[-1]["role"] == m.role:  # the API requires alternating roles
                out[-1]["content"].extend(content)
            else:
                out.append({"role": m.role, "content": content})
        return {"system": system, "messages": out}

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        payload = self.to_api(messages)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(gen.max_new_tokens, self.min_max_tokens),
            "messages": payload["messages"],
        }
        if payload["system"]:
            kwargs["system"] = payload["system"]
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if gen.stop:
            kwargs["stop_sequences"] = gen.stop
        if self.fallbacks:
            resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks=self.fallbacks, **kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        u = resp.usage
        prompt_tokens = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        raw = {
            "stop_reason": resp.stop_reason,
            "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
            "request_id": getattr(resp, "_request_id", None),
        }
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raw["refusal_category"] = getattr(details, "category", None) if details else None
        return LLMResponse(text=text, prompt_tokens=prompt_tokens, completion_tokens=u.output_tokens or 0, model=resp.model, raw=raw)

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.name, "model": self.model, "effort": self.effort}
