"""Reasoning traces and results (JSON-serialisable for logging / evaluation)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from ..backends.base import LLMResponse, Message, STSGPart, TextPart, VideoPart
from ..stsg.schema import STSG


def render_messages(messages: Sequence[Message]) -> str:
    """Human-readable rendering of a prompt (videos/STSGs shown as placeholders)."""
    lines = []
    for m in messages:
        buf = []
        for p in m.content:
            if isinstance(p, TextPart):
                buf.append(p.text)
            elif isinstance(p, VideoPart):
                ts = p.clip.timestamps
                span = f"{ts[0]:.1f}-{ts[-1]:.1f}s" if ts else ""
                buf.append(f"<video: {len(p.clip)} frames {span}>")
            elif isinstance(p, STSGPart):
                g = p.stsg
                buf.append(f"<{p.label}: {len(g)} frames, {g.num_nodes} nodes, {g.num_triplets} triplets>")
        lines.append(f"[{m.role}] " + "".join(buf))
    return "\n".join(lines)


@dataclass
class StepRecord:
    step: str
    round: int
    prompt: str
    response: str
    parsed: Any = None
    latency: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @classmethod
    def from_call(cls, step: str, round_: int, messages: Sequence[Message], r: LLMResponse, parsed: Any = None) -> "StepRecord":
        return cls(step, round_, render_messages(messages), r.text, parsed, r.latency, r.prompt_tokens, r.completion_tokens)


@dataclass
class ReasoningResult:
    """Outcome of one QA instance under VoT (or a baseline)."""

    method: str
    question: str
    candidates: List[str]
    answer_index: Union[int, List[int], None]
    answer_text: str
    multi_choice: bool
    targets: List[str] = field(default_factory=list)
    partial_stsg: Optional[STSG] = None
    observation: str = ""
    scores: List[Optional[float]] = field(default_factory=list)
    rationales: List[str] = field(default_factory=list)
    verified: Optional[bool] = None
    verification: str = ""
    rejected: List[int] = field(default_factory=list)
    rounds: int = 0
    generated_candidates: bool = False
    trace: List[StepRecord] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def prediction(self) -> Union[int, List[int], str, None]:
        """What metrics consume: an option index for multiple-choice, text otherwise."""
        return self.answer_index if self.multi_choice else self.answer_text

    def to_dict(self, with_trace: bool = True) -> Dict[str, Any]:
        d = asdict(self)
        d["partial_stsg"] = self.partial_stsg.to_dict() if self.partial_stsg is not None else None
        d["prediction"] = self.prediction
        if not with_trace:
            d.pop("trace")
        return d

    def to_json(self, **kw: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kw)

    def pretty(self) -> str:
        """Compact multi-line summary for terminals."""
        out = [f"Question : {self.question}"]
        if self.candidates:
            for i, c in enumerate(self.candidates):
                s = self.scores[i] if i < len(self.scores) else None
                mark = "*" if (self.answer_index == i or (isinstance(self.answer_index, list) and i in self.answer_index)) else " "
                rej = " (rejected)" if i in self.rejected else ""
                out.append(f"  {mark} {chr(65 + i)}. {c}" + (f"   score={s:g}" if s is not None else "") + rej)
        if self.targets:
            out.append(f"Targets  : {', '.join(self.targets)}")
        if self.partial_stsg is not None and not self.partial_stsg.is_empty:
            g = self.partial_stsg
            out.append(f"Tracklet : {len(g)} frames, {g.num_nodes} nodes, {g.num_triplets} triplets")
        if self.observation:
            obs = " ".join(self.observation.split())
            out.append(f"Observe  : {obs[:300]}{'...' if len(obs) > 300 else ''}")
        out.append(f"Answer   : {self.answer_text}")
        if self.verified is not None:
            out.append(f"Verified : {self.verified} (rounds={self.rounds})")
        if self.usage:
            out.append(f"Usage    : {self.usage}")
        return "\n".join(out)
