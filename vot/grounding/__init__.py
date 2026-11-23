"""STSG providers for the API version of VoT (MotionEpic parses STSGs itself)."""

from __future__ import annotations

from typing import Any

from .base import StaticSTSGProvider, STSGProvider, resample_to_clip


def build_provider(spec: str, model: Any = None, **kwargs: Any) -> STSGProvider:
    """``"vlm"`` (needs ``model``), ``"detector"``, or ``"static:<dir>"``."""
    kind, _, arg = spec.partition(":")
    if kind == "vlm":
        from .vlm_parser import VLMSTSGParser

        if model is None:
            raise ValueError("the 'vlm' STSG provider needs a backend")
        return VLMSTSGParser(model, **kwargs)
    if kind == "detector":
        from .detector_parser import DetectorSTSGParser

        if arg:
            kwargs.setdefault("model_id", arg)
        return DetectorSTSGParser(**kwargs)
    if kind == "static":
        return StaticSTSGProvider(arg, **kwargs)
    raise ValueError(f"Unknown STSG provider {spec!r}")


__all__ = ["STSGProvider", "StaticSTSGProvider", "build_provider", "resample_to_clip"]
