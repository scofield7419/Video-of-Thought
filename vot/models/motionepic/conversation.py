"""Vicuna v1 conversation template, rendered as tokenization segments.

Prompt string: ``{system} USER: {q1} ASSISTANT: {a1}</s>USER: {q2} ASSISTANT:``.

Training and inference both tokenize segment by segment (instruction segments
masked, assistant replies supervised + EOS), so the token boundaries the model
sees at inference are exactly those it was trained on, for any tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass
class Segment:
    text: str
    target: bool
    eos: bool = False


@dataclass
class Conversation:
    system: str = DEFAULT_SYSTEM
    roles: Tuple[str, str] = ("USER", "ASSISTANT")
    sep: str = " "
    messages: List[Tuple[str, Optional[str]]] = field(default_factory=list)

    def append(self, role: str, text: Optional[str]) -> "Conversation":
        if role not in self.roles:
            raise ValueError(f"role must be one of {self.roles}")
        self.messages.append((role, text))
        return self

    def segments(self) -> List[Segment]:
        """Instruction segments end with ``ASSISTANT:``; a trailing one is the generation prompt."""
        user, asst = self.roles
        segs: List[Segment] = []
        pending = self.system.strip()
        for role, text in self.messages:
            if role == user:
                pending = f"{pending}{self.sep if pending else ''}{user}: {(text or '').strip()}{self.sep}{asst}:"
            else:
                if pending:
                    segs.append(Segment(pending, target=False))
                    pending = ""
                if text is not None:
                    segs.append(Segment(text.strip(), target=True, eos=True))
        if pending:
            segs.append(Segment(pending, target=False))
        return segs

    def get_prompt(self, eos_token: str = "</s>") -> str:
        return "".join(s.text + (eos_token if s.eos else " ") for s in self.segments()).rstrip()


def vicuna_v1(system: Optional[str] = None) -> Conversation:
    return Conversation(system=DEFAULT_SYSTEM if system is None else system)
