"""Model backends. Heavy dependencies are imported lazily by :func:`build_backend`."""

from __future__ import annotations

from typing import Any

from .base import (
    BackendCapabilities,
    GenerationConfig,
    LLMResponse,
    Message,
    STSGPart,
    TextPart,
    VideoLLM,
    VideoPart,
)


def build_backend(spec: str, cache: str | None = None, **kwargs: Any) -> VideoLLM:
    """Create a backend from ``"<provider>:<model>"``.

    * ``openai:gpt-4o`` / ``openai:Qwen/Qwen2.5-VL-72B-Instruct`` (with ``base_url=...``)
    * ``anthropic:claude-opus-5-5``
    * ``motionepic:/path/to/checkpoint`` (local MotionEpic, needs ``.[train]`` extras)
    * ``mock`` (offline rule-based stand-in for tests / dry runs)

    ``cache`` wraps the backend with an SQLite response cache at that path.
    """
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider in {"openai", "oai", "vllm"}:
        from .openai_compat import OpenAICompatibleBackend

        backend: VideoLLM = OpenAICompatibleBackend(model=model or "gpt-4o", **kwargs)
    elif provider in {"anthropic", "claude"}:
        from .anthropic_backend import AnthropicBackend

        backend = AnthropicBackend(model=model or "claude-opus-5-5", **kwargs)
    elif provider == "motionepic":
        from .motionepic import MotionEpicBackend

        backend = MotionEpicBackend.from_pretrained(model, **kwargs)
    elif provider == "mock":
        from .mock import RuleBasedMockLLM

        backend = RuleBasedMockLLM(**kwargs)
    else:
        raise ValueError(f"Unknown backend provider {provider!r} in {spec!r}")
    if cache:
        from .cache import CachedLLM

        backend = CachedLLM(backend, cache)
    return backend


__all__ = [
    "BackendCapabilities",
    "GenerationConfig",
    "LLMResponse",
    "Message",
    "STSGPart",
    "TextPart",
    "VideoLLM",
    "VideoPart",
    "build_backend",
]
