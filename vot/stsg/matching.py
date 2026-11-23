"""Grounding free-form target phrases (Step-1 output) onto STSG nodes.

Used when the partial STSG of Step-2 is *extracted* from an existing STSG
(``step2_mode="match"``) and as a fallback when a generated partial STSG cannot
be parsed. MotionEpic itself grounds targets generatively.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from .schema import STSG

_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "of", "in", "on", "at", "to", "with", "and", "or",
    "his", "her", "their", "its", "my", "your", "our", "some", "any", "other", "another", "one", "two",
    "three", "several", "many", "video", "scene", "clip", "frame", "who", "which", "is", "are", "being",
}

SYNONYM_GROUPS: List[set] = [
    {"person", "man", "woman", "people", "boy", "girl", "child", "kid", "baby", "guy", "lady", "player",
     "human", "adult", "men", "women", "worker", "driver", "trainer", "someone", "pedestrian"},
    {"car", "vehicle", "truck", "van", "bus", "suv", "taxi", "jeep", "lorry"},
    {"bike", "bicycle", "motorcycle", "motorbike", "scooter"},
    {"dog", "puppy", "pet"},
    {"cat", "kitten"},
    {"cup", "mug", "glass"},
    {"sofa", "couch"},
    {"phone", "cellphone", "smartphone"},
    {"laptop", "computer", "notebook"},
    {"tv", "television", "screen", "monitor"},
    {"road", "street"},
    {"bag", "backpack", "handbag"},
]


def _singular(tok: str) -> str:
    if len(tok) > 3 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("es") and tok[-3] in "sxz":
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def normalize_phrase(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    return [_singular(t) for t in toks if t not in _STOPWORDS]


def _synonyms(tok: str) -> set:
    for g in SYNONYM_GROUPS:
        if tok in g:
            return g
    return {tok}


def phrase_category_score(phrase: str, category: str) -> float:
    pt, ct = normalize_phrase(phrase), normalize_phrase(category)
    if not pt or not ct:
        return 0.0
    if pt == ct:
        return 1.0
    p_head, c_head = pt[-1], ct[-1]
    if p_head == c_head:
        return 0.9
    if c_head in pt or p_head in ct:
        return 0.8
    if c_head in _synonyms(p_head):
        return 0.7
    inter = len(set(pt) & set(ct))
    return inter / len(set(pt) | set(ct))


def match_targets(
    targets: Sequence[str],
    stsg: STSG,
    threshold: float = 0.5,
    embed_fn: Optional[Callable[[List[str]], "Sequence[Sequence[float]]"]] = None,
    top_k: Optional[int] = None,
) -> Dict[str, List[str]]:
    """Map each target phrase to the object ids it refers to.

    Exact id mentions (``"car-1"``) win; otherwise categories are scored with
    lexical/synonym heuristics, optionally blended with cosine similarity from
    ``embed_fn`` (e.g. a sentence-transformer). Returns ``{target: [ids]}``
    (ids may occur in several frames).
    """
    ids_by_cat: Dict[str, List[str]] = {}
    all_ids: List[str] = []
    for f in stsg.frames:
        for o in f.objects:
            if o.id not in all_ids:
                all_ids.append(o.id)
            lst = ids_by_cat.setdefault(o.category, [])
            if o.id not in lst:
                lst.append(o.id)
    cats = list(ids_by_cat)
    emb_sim: Dict[str, Dict[str, float]] = {}
    if embed_fn is not None and cats and targets:
        vecs = embed_fn(list(targets) + cats)
        tv, cv = vecs[: len(targets)], vecs[len(targets):]

        def cos(a: Sequence[float], b: Sequence[float]) -> float:
            num = sum(x * y for x, y in zip(a, b))
            da = sum(x * x for x in a) ** 0.5
            db = sum(y * y for y in b) ** 0.5
            return num / (da * db) if da and db else 0.0

        for t, a in zip(targets, tv):
            emb_sim[t] = {c: cos(a, b) for c, b in zip(cats, cv)}

    out: Dict[str, List[str]] = {}
    for t in targets:
        mentioned = [i for i in all_ids if re.search(r"(?<![\w-])" + re.escape(i.lower()) + r"(?![\w-])", t.lower())]
        if mentioned:
            out[t] = mentioned
            continue
        scored = []
        for c in cats:
            s = phrase_category_score(t, c)
            if emb_sim:
                s = max(s, emb_sim[t][c])
            if s >= threshold:
                scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        if top_k:
            scored = scored[:top_k]
        ids: List[str] = []
        for _, c in scored:
            ids.extend(i for i in ids_by_cat[c] if i not in ids)
        out[t] = ids
    return out


def ground_targets(targets: Iterable[str], stsg: STSG, hops: int = 1, **kwargs) -> STSG:
    """Partial STSG containing the tracklets of all matched targets and their
    ``hops``-hop neighbour scenes."""
    mapping = match_targets(list(targets), stsg, **kwargs)
    seeds = [i for ids in mapping.values() for i in ids]
    return stsg.subgraph(seeds, hops=hops) if seeds else STSG(meta=dict(stsg.meta))
