"""Configuration of the VoT reasoning engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from ..backends.base import GenerationConfig


@dataclass
class VoTConfig:
    # ---- video sampling. MotionEpic uses 8 fps; API models are run with fewer
    # frames for cost (Sec. 5.1 discusses the sampling-rate trade-off).
    fps: float = 1.0
    max_frames: int = 16
    frame_max_side: Optional[int] = 768
    #: frame rate of inputs given as a directory of frame images
    frames_fps: float = 1.0

    # ---- context handling. "chain": Steps 1-3 form one dialogue (video given
    # once, outputs recycled as context); "isolated": every step is a fresh call
    # that only sees the slot-filled prompt and the modalities shown in the
    # paper's step boxes. Steps 4 and 5 are always self-contained.
    context: str = "chain"

    # ---- Step-1
    max_targets: int = 5

    # ---- Step-2. "auto" = "generate" without an STSG, "select" with one.
    #   generate: the model outputs the partial STSG from the video (STSG-free, as MotionEpic)
    #   select:   the model is shown the full STSG and outputs the relevant partial STSG
    #   match:    targets are matched to STSG nodes programmatically (no LLM call)
    step2_mode: str = "auto"
    neighbor_hops: int = 1
    stsg_max_frames: Optional[int] = 16
    snap_boxes_to_reference: bool = True

    # ---- Step-3
    step3_with_video: bool = False

    # ---- Step-4
    num_open_candidates: int = 4
    rescore_open_candidates: bool = True
    #   model: always run the Step-4-B ranking prompt (paper); unique score maxima win.
    #   auto:  run Step-4-B only when scores tie or are missing.
    #   score: never run Step-4-B; ties are broken by option order.
    ranking: str = "model"
    multi_answer: bool = False
    multi_answer_threshold: float = 7.0
    observation_max_chars: Optional[int] = None

    # ---- Step-5. Disabling one perspective reproduces the "w/o Verify-G/C"
    # ablation of Table 4.
    verify: bool = True
    verify_perception: bool = True
    verify_cognition: bool = True
    max_verify_rounds: int = 2
    rescore_on_reject: bool = True
    feedback_max_chars: int = 600

    # ---- generation
    gen: GenerationConfig = field(default_factory=GenerationConfig)
    step2_max_new_tokens: int = 3072
    parallel_calls: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VoTConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown VoTConfig keys: {sorted(unknown)}")
        d = dict(d)
        if isinstance(d.get("gen"), dict):
            d["gen"] = GenerationConfig(**d["gen"])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str) -> "VoTConfig":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get("vot", data))

    def validate(self) -> None:
        if self.context not in {"chain", "isolated"}:
            raise ValueError(f"context must be 'chain' or 'isolated', got {self.context!r}")
        if self.step2_mode not in {"auto", "generate", "select", "match"}:
            raise ValueError(f"invalid step2_mode {self.step2_mode!r}")
        if self.ranking not in {"model", "auto", "score"}:
            raise ValueError(f"invalid ranking {self.ranking!r}")
