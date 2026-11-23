"""Robust parsers for the outputs of each VoT step.

The paper's I/O examples (Appendix A.1) fix the expected surface forms, e.g.
"The involved targets are [the white truck], [the neighborhood]" or "the
rationality of the answer [A. Transportation] is 5". Generations of real models
deviate, so every parser has fallbacks and returns ``None``/empty instead of
raising.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .prompts import OPTION_LETTERS

_BRACKET_RE = re.compile(r"\[([^\[\]]{1,200})\]")
_NUM = r"(\d+(?:\.\d+)?)"


def _clean_item(s: str) -> str:
    s = s.strip().strip("\"'`*").strip()
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".,;:").strip()


# ----------------------------------------------------------------- Step-1
def parse_targets(text: str, max_targets: int = 6) -> List[str]:
    """Targets from "The involved targets are [a], [b]" (with fallbacks)."""
    if not text:
        return []
    m = re.search(r"targets?\s+(?:are|is|include|includes)\s*:?(.*)", text, re.IGNORECASE | re.DOTALL)
    region = m.group(1) if m else text
    items = [_clean_item(x) for x in _BRACKET_RE.findall(region)]
    if not items:
        first = region.strip().split("\n")[0]
        first = re.sub(r"^(the involved targets are|targets?:)", "", first, flags=re.IGNORECASE)
        items = [_clean_item(x) for x in re.split(r",|;|\band\b", first)]
    out: List[str] = []
    seen = set()
    for it in items:
        if not it or len(it.split()) > 12 or it.lower() in {"video", "target", "targets", "target 1", "target 2"}:
            continue
        if it.lower() not in seen:
            seen.add(it.lower())
            out.append(it)
    return out[:max_targets]


# ----------------------------------------------------------------- Step-4
def _blank_brackets(text: str) -> str:
    # candidate texts inside [...] may contain digits ("B. 3 people")
    return _BRACKET_RE.sub("[]", text)


_NOT_SCALE = r"(?!\s*(?:-|–|to)\s*10\b)"
_LETTER_END = r"(?=[\.\):,\]*]|\s*$|\s*\n)"


def parse_score(text: str, lo: float = 1.0, hi: float = 10.0) -> Optional[float]:
    """Rationality score on a 1-10 scale (``None`` when absent)."""
    if not text:
        return None
    t = _blank_brackets(text)
    patterns = [
        r"rationality[^\n\d]{0,80}?\bis\s*[:=]?\s*(?:a\s*|of\s*)?\**\s*" + _NUM + _NOT_SCALE,
        r"\bscore[sd]?\b[^\n\d]{0,30}?" + _NUM + _NOT_SCALE + r"(?:\s*/\s*10|\s*out of\s*10)?",
        _NUM + r"\s*/\s*10\b",
        _NUM + r"\s*out of\s*10\b",
        r"\b(?:rated?|deserves?|assign|give it)\s*(?:it\s*)?(?:as\s*|a\s*(?:score of\s*)?)?" + _NUM + _NOT_SCALE,
    ]
    for p in patterns:
        for m in re.finditer(p, t, re.IGNORECASE):
            v = float(m.group(1))
            if lo <= v <= hi:
                return v
    return None


def _letter_index(s: str, n: int) -> Optional[int]:
    """A bare option letter: "C", "(C)", "C.", "C)", "C:" - but not the article in "A dog runs"."""
    m = re.match(r"^\(?([A-Z])\)?" + _LETTER_END, s)
    if m and OPTION_LETTERS.index(m.group(1)) < n:
        return OPTION_LETTERS.index(m.group(1))
    return None


def match_option(text: str, candidates: Sequence[str]) -> Optional[int]:
    """Map a span like "C. Clean Service", "(C)", "C" or the option text to an index."""
    s = _clean_item(text)
    if not s:
        return None
    n = len(candidates)
    idx = _letter_index(s, n)
    if idx is not None:
        return idx
    low = s.lower().rstrip(".")
    norm = [c.lower().strip().rstrip(".") for c in candidates]
    for i, cl in enumerate(norm):  # exact text wins
        if cl and cl == low:
            return i
    m = re.match(r"^\(?([A-Z])\)?\s+(.+)$", s)  # "C Clean Service"
    if m and OPTION_LETTERS.index(m.group(1)) < n and norm[OPTION_LETTERS.index(m.group(1))] == m.group(2).lower().rstrip("."):
        return OPTION_LETTERS.index(m.group(1))
    best, best_len = None, 0
    for i, cl in enumerate(norm):
        if cl and (re.search(r"(?<!\w)" + re.escape(cl) + r"(?!\w)", low) or (len(low) > 3 and low in cl)):
            if len(cl) > best_len:
                best, best_len = i, len(cl)
    return best


def parse_choice(text: str, candidates: Sequence[str]) -> Optional[int]:
    """Final choice from free text ("So, the final answer is [C. ...]", "Final answer: C")."""
    if not text:
        return None
    pats = [
        r"final answer\s*(?:is|:)?\s*:?\s*\[([^\]]+)\]",
        r"final answer\s*(?:is|:)?\s*:?\s*\**\s*(\(?[A-Z]\)?)" + _LETTER_END,
        r"answer\s*(?:is|:)\s*\[([^\]]+)\]",
        r"answer\s*(?:is|:)\s*\**\s*(\(?[A-Z]\)?)" + _LETTER_END,
        r"(?:option|choice)\s*(\(?[A-Z]\)?)" + _LETTER_END,
    ]
    for p in pats:
        ms = list(re.finditer(p, text, re.IGNORECASE))
        for m in reversed(ms):  # the last statement wins ("... so the final answer is")
            idx = match_option(m.group(1), candidates)
            if idx is not None:
                return idx
    line = parse_final_answer_line(text)
    if line:
        idx = match_option(line, candidates)
        if idx is not None:
            return idx
    stripped = text.strip()
    return match_option(stripped.split("\n")[0], candidates) if len(stripped) < 200 else None


def parse_ranking(text: str, candidates: Sequence[str]) -> Tuple[List[int], Optional[int]]:
    """Ranking ``[C] [A] [D] [B]`` and the final answer from the Step-4-B output."""
    order: List[int] = []
    if text:
        m = re.search(r"(?:descending order|ranking)[^\[]*((?:\[[^\]]+\][\s,>]*)+)", text, re.IGNORECASE)
        span = m.group(1) if m else ""
        for item in _BRACKET_RE.findall(span):
            idx = match_option(item, candidates)
            if idx is not None and idx not in order:
                order.append(idx)
    final = parse_choice(text, candidates)
    if final is None and order:
        final = order[0]
    return order, final


@dataclass
class GeneratedCandidate:
    text: str
    score: Optional[float]
    rationale: str


_CAND_HDR = re.compile(
    r"^\s*[\*#\->]*\s*(?:(?:answer|option|candidate)\s*\(?(?:[A-Za-z]|\d{1,2})\)?|\(?(?:[A-Z]|\d{1,2})[.)])"
    r"\s*\**\s*[:.)\-]?\s*\**\s*(?:(?:answer|option)\s*[:\-]\s*)?(.*)$",
    re.IGNORECASE,
)
_FIELD = re.compile(r"^\s*[\*(]*\s*(score|rationality(?: score)?|rationale|reason)\s*\**\s*[:=\-]", re.IGNORECASE)


def parse_open_candidates(text: str, k: int = 4) -> List[GeneratedCandidate]:
    """Step-4-Pre: "Answer A: <answer> | Score: 7 | Rationale: ..." (also numbered
    lists and multi-line "Score:" / "Rationale:" fields)."""
    if not text:
        return []
    blocks: List[List[str]] = []
    for line in text.splitlines():
        m = _CAND_HDR.match(line)
        if m and not _FIELD.match(line) and m.group(1).strip():
            blocks.append([m.group(1)])
        elif blocks and line.strip():
            blocks[-1].append(line)
    out: List[GeneratedCandidate] = []
    for b in blocks:
        segs = [x.strip() for x in "|".join(b).split("|") if x.strip()]
        ans_seg = re.split(r"\(?\b(?:score|rationality|rationale)\b", segs[0], flags=re.IGNORECASE)[0]
        ans = _clean_item(ans_seg)
        score: Optional[float] = None
        rat = ""
        for sg in segs[1:] + [segs[0][len(ans_seg):]]:
            f = _FIELD.match(sg)
            if not f:
                continue
            if f.group(1).lower().startswith(("score", "rationality")) and score is None:
                score = parse_score(sg)
            elif f.group(1).lower() in {"rationale", "reason"}:
                rat = sg[f.end():].strip()
        if score is None:  # never read numbers from the answer text itself
            score = parse_score(segs[0][len(ans_seg):] + " " + " ".join(segs[1:]))
        if ans:
            out.append(GeneratedCandidate(ans, score, rat))
    uniq: List[GeneratedCandidate] = []
    for c in out:
        if c.text.lower() not in {u.text.lower() for u in uniq}:
            uniq.append(c)
    return uniq[:k]


# ----------------------------------------------------------------- Step-5
def parse_verdict(text: str) -> Optional[bool]:
    """``True`` (answer verified), ``False`` (contradiction found) or ``None``."""
    if not text:
        return None
    m = re.findall(
        r"verdict\**\s*[:=\-]?\s*\**\s*(correct|incorrect|pass|fail|yes|no|true|false)\b", text, re.IGNORECASE
    )
    if m:
        return m[-1].lower() in {"correct", "pass", "yes", "true"}
    tail = re.sub(r"\bno contradictions?\b|\bdoes not contradict\b|\bnot contradict\w*", " consistent ", text.lower()[-400:])
    neg = re.search(
        r"\b(is|seems|appears)\s+(not\s+correct|incorrect|wrong|inconsistent)|\bcontradict(s|ed|ion)?\b|\bdoes not align\b|\bis not supported\b",
        tail,
    )
    pos = re.search(r"\b(is|remains)\s+(correct|verified|consistent|valid)\b|\bconsistent\b", tail)
    if neg and not pos:
        return False
    if pos and not neg:
        return True
    if pos and neg:
        return pos.start() > neg.start()
    return None


def parse_final_answer_line(text: str) -> Optional[str]:
    """"Final answer: <x>" for open-ended CoT baselines."""
    ms = re.findall(r"final answer\s*[:：]\s*(.+)", text or "", re.IGNORECASE)
    return _clean_item(ms[-1]) if ms else None
