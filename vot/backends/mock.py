"""Offline backends for tests and dry runs. They are NOT models: the rule-based
mock recognises which VoT step a prompt belongs to and returns well-formed
outputs derived from the given STSG, so the whole pipeline (parsing, ranking,
verification loop, logging) can be exercised without GPUs or API keys."""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Sequence

from ..reasoning.prompts import OPTION_LETTERS
from ..stsg.matching import ground_targets, normalize_phrase
from ..stsg.schema import STSG
from .base import BackendCapabilities, GenerationConfig, LLMResponse, Message, STSGPart, VideoLLM


class ScriptedLLM(VideoLLM):
    """Delegates every call to ``fn(messages) -> str``."""

    name = "scripted"

    def __init__(self, fn: Callable[[Sequence[Message]], str], capabilities: Optional[BackendCapabilities] = None) -> None:
        super().__init__()
        self.fn = fn
        self.capabilities = capabilities or BackendCapabilities()
        self.calls: List[Sequence[Message]] = []

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        self.calls.append(messages)
        text = self.fn(messages)
        return LLMResponse(text=text, prompt_tokens=sum(len(m.text().split()) for m in messages), completion_tokens=len(text.split()))


def _last_user_text(messages: Sequence[Message]) -> str:
    for m in reversed(messages):
        if m.role == "user":
            return m.text()
    return ""


def _stsg_in(messages: Sequence[Message]) -> Optional[STSG]:
    for m in reversed(messages):
        for p in m.content:
            if isinstance(p, STSGPart):
                return p.stsg
    return None


class RuleBasedMockLLM(VideoLLM):
    """Deterministic stand-in for a video LLM.

    Args:
        stsg: oracle STSG of the video, used to answer Step-1/2/3 prompts.
        answer_key: optional substring of the "correct" option; it gets score 9,
            others 3 (otherwise scores come from word overlap with the observation).
        reject_first: make the first verification fail (exercises Step-4 re-execution).
    """

    name = "mock"

    def __init__(self, stsg: Optional[STSG] = None, answer_key: Optional[str] = None, reject_first: bool = False) -> None:
        super().__init__()
        self.stsg = stsg
        self.answer_key = answer_key
        self.reject_first = reject_first
        self._verifications = 0
        self.capabilities = BackendCapabilities(knows_stsg_format=False)

    def _targets(self, question: str) -> List[str]:
        q = set(normalize_phrase(question))
        cats = self.stsg.categories() if self.stsg else []
        hits = [c for c in cats if set(normalize_phrase(c)) & q]
        return hits or cats[:2] or ["the main subject"]

    def _observe(self, g: Optional[STSG]) -> str:
        if g is None or g.is_empty:
            return "No reliable grounding was available; the scene shows the main subject acting in its environment."
        facts: Dict[str, List[int]] = {}
        for f in g.frames:
            for t in f.triplets:
                facts.setdefault(f"{t.subject} {t.predicate} {t.object}", []).append(f.index)
        lines = [f"In frames {min(v)}-{max(v)}, {k}." for k, v in list(facts.items())[:8]]
        tracks = g.tracks()
        moving = []
        for name, members in tracks.items():
            boxes = [o.bbox for _, o in members if o.bbox]
            if len(boxes) >= 2:
                dx = boxes[-1][0] - boxes[0][0]
                if abs(dx) > 2:
                    moving.append(f"{name} moves {'right' if dx > 0 else 'left'}")
        if moving:
            lines.append("Motion: " + "; ".join(moving) + ".")
        lines.append("Implication: the targets interact as described, which suggests the corresponding activity.")
        return " ".join(lines)

    def _score(self, candidate: str, observation: str) -> int:
        if self.answer_key is not None:
            return 9 if self.answer_key.lower() in candidate.lower() else 3
        c, o = set(normalize_phrase(candidate)), set(normalize_phrase(observation))
        return max(1, min(10, 2 + 3 * len(c & o)))

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        u = _last_user_text(messages)
        low = u.lower()
        g = _stsg_in(messages) or self.stsg
        if "possible targets" in low:
            q = re.search(r"Given the question: \[(.*?)\], what", u, re.S)
            text = "The involved targets are " + ", ".join(f"[{t}]" for t in self._targets(q.group(1) if q else u)) + "."
        elif "provide the tracklet" in low:
            tg = re.findall(r"\[([^\]]+)\]", u.split("Provide the tracklet of involved")[-1].split("by outputting")[0])
            sub = ground_targets(tg, g, hops=1) if g is not None else STSG()
            text = f"The partial STSG in tracking {', '.join(f'[{t}]' for t in tg)} is:\n{sub.to_expression()}"
        elif "analyze the motion behavior" in low:
            text = self._observe(g)
        elif "distinct optional answers" in low:
            obs = re.search(r"action's \[(.*?)\] combined", u, re.S)
            facts = re.findall(r"frames \d+-\d+, ([^.]+)\.", obs.group(1) if obs else "")
            opts = (facts + ["something unrelated happens", "nothing happens", "the scene is static"])[:4]
            text = "\n".join(
                f"Answer {OPTION_LETTERS[i]}: {o} | Score: {8 - 2 * i if i < len(facts) else 2} | Rationale: grounded in the observations."
                for i, o in enumerate(opts)
            )
        elif "score the rationality" in low:
            cand = re.search(r"given a candidate answer \[(.*?)\], please", u, re.S)
            obs = re.search(r"action's \[(.*?)\] combined", u, re.S)
            c = cand.group(1) if cand else ""
            s = self._score(c, obs.group(1) if obs else "")
            if "rejected by verification" in low and c and c in u.split("Note that the previous answer [")[-1].split("]")[0]:
                s = 1
            text = f"Based on the observations and commonsense, the rationality of the answer [{c}] is {s}. The rationale is derived from the grounded observations."
        elif "please rank the candidate" in low:
            items = re.findall(r"answer \[([^\]]+)\] is (\d+(?:\.\d+)?|unknown)", u)
            ranked = sorted(items, key=lambda x: -(float(x[1]) if x[1] != "unknown" else 0))
            order = " ".join(f"[{c}]" for c, _ in ranked)
            text = f"Based on the rationality, the ranking results in descending order is {order}. So, the final answer is [{ranked[0][0]}]." if ranked else "No ranking."
        elif "verify the previous answer" in low:
            self._verifications += 1
            ok = not (self.reject_first and self._verifications == 1)
            text = (
                "1. Grounding information: the tracklets support the answer.\n2. Commonsense verification: no contradiction.\nVerdict: CORRECT"
                if ok
                else "1. Grounding information: the answer is not supported by the tracklets.\n2. Commonsense: it contradicts the observations.\nVerdict: INCORRECT"
            )
        elif "let's think step by step" in low:
            text = self._observe(g) + "\nFinal answer: A"
        elif "option's letter" in low:
            text = "A"
        else:
            text = self._observe(g)
        return LLMResponse(text=text, prompt_tokens=len(u.split()), completion_tokens=len(text.split()))
