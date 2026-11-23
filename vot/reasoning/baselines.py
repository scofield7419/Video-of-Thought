"""Baselines compared against VoT in the paper: direct answering and vanilla
CoT ("Let's think step by step"), optionally with STSG input (the
"Video-LLaVA+STSG" setting)."""

from __future__ import annotations

from typing import Optional, Sequence

from ..backends.base import Message, STSGPart, TextPart, VideoLLM, VideoPart
from ..grounding.base import STSGProvider
from ..stsg.schema import STSG
from .config import VoTConfig
from .parsing import parse_choice, parse_final_answer_line
from .prompts import PromptTemplates, format_question
from .result import ReasoningResult, StepRecord
from .vot import VideoInput, VideoOfThought


class _SingleCall(VideoOfThought):
    method = "base"

    def __init__(
        self,
        model: VideoLLM,
        config: Optional[VoTConfig] = None,
        stsg_provider: Optional[STSGProvider] = None,
        prompts: Optional[PromptTemplates] = None,
    ) -> None:
        super().__init__(model, config, stsg_provider, prompts)

    def _prompt(self, question: str, mc: bool) -> str:
        raise NotImplementedError

    def run(
        self,
        video: VideoInput,
        question: str,
        candidates: Optional[Sequence[str]] = None,
        *,
        stsg: Optional[STSG] = None,
        video_id: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> ReasoningResult:
        clip = self.load_clip(video, start, end)
        cands = [str(c) for c in candidates] if candidates else []
        mc = bool(cands)
        g = stsg
        if g is None and self.stsg_provider is not None:
            g = self.stsg_provider(clip, question=question, video_id=video_id)
        parts: list = [TextPart("<Video>\n"), VideoPart(clip), TextPart("\n</Video>\n")]
        if g is not None and not g.is_empty:
            parts += [STSGPart(g, label="STSG", max_frames=self.cfg.stsg_max_frames), TextPart("\n")]
        q = format_question(question, cands if mc else None)
        msgs = [Message.system(self.P.task_definition(mc)), Message.user(*parts, self._prompt(q, mc))]
        r = self.model.generate(msgs, self._gen())
        if mc:
            idx = parse_choice(r.text, cands)
            answer = cands[idx] if idx is not None else ""
        else:
            idx = None
            answer = parse_final_answer_line(r.text) or r.text.strip().split("\n")[-1].strip()
        rec = StepRecord.from_call(self.method, 0, msgs, r, idx if mc else answer)
        return ReasoningResult(
            method=self.method,
            question=question,
            candidates=cands,
            answer_index=idx,
            answer_text=answer,
            multi_choice=mc,
            observation=r.text.strip() if self.method == "cot" else "",
            trace=[rec],
            usage={"calls": 1, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens},
            meta={"backend": self.model.describe(), "num_frames": len(clip), "with_stsg": g is not None},
        )


class DirectAnswer(_SingleCall):
    """Answer directly from the video (no reasoning chain)."""

    method = "direct"

    def _prompt(self, question: str, mc: bool) -> str:
        return (self.P.direct if mc else self.P.direct_oe).format(question=question)


class VanillaCoT(_SingleCall):
    """Zero-shot CoT with a single "Let's think step by step" prompt."""

    method = "cot"

    def _prompt(self, question: str, mc: bool) -> str:
        hint = self.P.cot_hint_mc if mc else self.P.cot_hint_oe
        return self.P.cot.format(question=question) + ("\n" + hint if self.P.format_hints else "")
