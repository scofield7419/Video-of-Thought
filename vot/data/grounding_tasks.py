"""Fine-grained video-scene grounding-aware tuning data (paper Sec. 3.3).

From a (video, STSG) pair we build instruction-tuning samples for the five
grounding objectives:

* **L1** (coarse) - is the given STSG paired with the video? The STSG enters only
  through ``<stsg>`` graph tokens; negatives are STSGs of other videos or
  perturbed graphs (swapped categories / predicates, reversed time).
* **L2** (coarse) - generate the whole STSG expression of the video.
* **L3** (fine) - given action description(s), output the object tracklet(s), i.e.
  a partial STSG.
* **L4** (fine) - given key object(s), describe their temporal action(s) and
  output their tracklet(s).
* **L5** (fine) - given a box of an object in one frame, output its label and
  tracklet.

Frame alignment: every sample fixes the exact frames shown to the model
(``frames`` image paths or ``frame_times``) and the STSG is renumbered so that
"Frame k" is the k-th of those frames. Action descriptions come from external
annotations (e.g. Charades actions for Action Genome) or, when absent, from
contiguous spans of relation triplets in the STSG.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..stsg.matching import match_targets
from ..stsg.schema import STSG, CorefLink, FrameGraph, Triplet

TASKS = ("L1", "L2", "L3", "L4", "L5")

#: per-task frame caps: a full-video STSG expression (L2) grows fastest with the frame count
DEFAULT_TASK_MAX_FRAMES = {"L2": 8}

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d|[^\w\s]")


def approx_token_count(text: str) -> int:
    """Conservative LLaMA-style token estimate: every digit and punctuation mark is
    a token (STSG expressions are dominated by them), plus ~15% for sub-word splits."""
    return int(len(_TOKEN_RE.findall(text)) * 1.15) + 1

#: predicates that do not describe an action (kept in STSGs, skipped as descriptions)
NON_ACTION_PREDICATES = {
    "unsure", "not contacting", "not_contacting", "other relationship", "other_relationship",
    "not looking at", "not_looking_at",
}

TEMPLATES: Dict[str, List[str]] = {
    "L1": [
        "Is the spatial-temporal scene graph {stsg} paired with the video? Answer yes or no.",
        "Does the following STSG {stsg} correctly describe this video? Answer yes or no.",
        "Given the video and the scene graph {stsg}, decide whether they match. Answer yes or no.",
    ],
    "L2": [
        "Generate the spatial-temporal scene graph (STSG) expression of the video.",
        "Parse the video into its STSG expression, frame by frame, including the temporal coreference edges.",
        "Describe the video as a spatial-temporal scene graph: list the objects with boxes and the triplets of every frame.",
    ],
    "L3": [
        "Given the action description(s) {actions}, output the corresponding object tracklet(s) as a partial STSG expression.",
        "Ground the action(s) {actions} in the video by outputting the partial STSG of the involved objects.",
        "Which objects perform or undergo {actions}? Provide their tracklets as a partial STSG expression.",
    ],
    "L4": [
        "Given the key object(s) {objects}, describe the corresponding temporal action(s) and output the object tracklet(s).",
        "Describe what happens to {objects} over time in the video, then provide the tracklet(s) as a partial STSG expression.",
        "Track {objects}: explain their actions along the timeline and output the corresponding partial STSG.",
    ],
    "L5": [
        "In Frame {frame}, what is the object at {bbox}? Output the object label and its tracklet.",
        "Identify the object located at {bbox} in Frame {frame}, and provide its tracklet as a partial STSG expression.",
        "There is an object at {bbox} in Frame {frame}. Name it and output its tracklet.",
    ],
}


@dataclass
class ActionAnnotation:
    """External action label, e.g. a Charades action ``("Holding some clothes", 11.9, 21.2)``."""

    label: str
    start: Optional[float] = None
    end: Optional[float] = None
    objects: List[str] = field(default_factory=list)


@dataclass
class GroundingSource:
    """A (video, STSG) pair. ``frames``/``frame_times`` are aligned with ``stsg.frames``
    sorted by index (one entry per STSG frame)."""

    id: str
    stsg: STSG
    video: Optional[str] = None
    frames: Optional[List[str]] = None
    frame_times: Optional[List[float]] = None
    actions: List[ActionAnnotation] = field(default_factory=list)


@dataclass
class RelationSpan:
    subject: str
    predicate: str
    object: str
    subject_cat: str
    object_cat: str
    frames: List[int]  # renumbered frame indices
    keys: List[Tuple[int, str]]
    start: Optional[float] = None
    end: Optional[float] = None

    def describe(self, with_time: bool = True) -> str:
        pred = self.predicate.replace("_", " ")
        s = f"the {self.subject_cat} is {pred} the {self.object_cat}"
        if with_time:
            a, b = self.frames[0], self.frames[-1]
            s += f" in Frame {a}" if a == b else f" from Frame {a} to Frame {b}"
            if self.start is not None and self.end is not None:
                s += f" ({self.start:.1f}s-{self.end:.1f}s)"
        return s


# ----------------------------------------------------------------- helpers
def renumber(source: GroundingSource, max_frames: Optional[int] = None) -> Tuple[STSG, List[int]]:
    """Subsample to ``max_frames`` and renumber frames 1..K in time order.

    Returns the new STSG and the positions (0-based, in the source's sorted
    frame order) of the kept frames, used to select ``frames``/``frame_times``.
    """
    g = source.stsg
    order = sorted(f.index for f in g.frames)
    sub = g.subsample(max_frames) if max_frames else g.select_frames(order)
    kept = sorted(f.index for f in sub.frames)
    remap = {old: i + 1 for i, old in enumerate(kept)}
    pos = [order.index(k) for k in kept]
    times = source.frame_times
    frames = []
    for f in sorted(sub.frames, key=lambda x: x.index):
        nf = copy.deepcopy(f)
        nf.index = remap[f.index]
        if nf.timestamp is None and times is not None:
            nf.timestamp = times[order.index(f.index)]
        frames.append(nf)
    coref = [CorefLink(remap[c.src_frame], c.src, remap[c.dst_frame], c.dst) for c in sub.coref]
    return STSG(frames=frames, coref=coref, meta=dict(g.meta)), pos


def relation_spans(g: STSG, skip_non_actions: bool = True) -> List[RelationSpan]:
    """Contiguous runs of the same (track, predicate, track) relation."""
    tracks = g.tracks()
    track_of = {(fi, o.id): name for name, members in tracks.items() for fi, o in members}
    cat_of = {name: members[0][1].category for name, members in tracks.items()}
    order = sorted(f.index for f in g.frames)
    rank = {k: i for i, k in enumerate(order)}
    ts = {f.index: f.timestamp for f in g.frames}
    occ: Dict[Tuple[str, str, str], List[Tuple[int, Tuple[int, str], Tuple[int, str]]]] = {}
    for f in g.frames:
        for t in f.triplets:
            if skip_non_actions and t.predicate.lower() in NON_ACTION_PREDICATES:
                continue
            s, o = track_of.get((f.index, t.subject)), track_of.get((f.index, t.object))
            if s is None or o is None:
                continue
            occ.setdefault((s, t.predicate, o), []).append((f.index, (f.index, t.subject), (f.index, t.object)))
    spans: List[RelationSpan] = []
    for (s, p, o), items in occ.items():
        items.sort(key=lambda x: rank[x[0]])
        run: List = [items[0]]
        for it in items[1:] + [None]:
            if it is not None and rank[it[0]] == rank[run[-1][0]] + 1:
                run.append(it)
                continue
            frames = [r[0] for r in run]
            keys = [k for r in run for k in (r[1], r[2])]
            spans.append(RelationSpan(s, p, o, cat_of[s], cat_of[o], frames, keys, ts.get(frames[0]), ts.get(frames[-1])))
            if it is not None:
                run = [it]
    spans.sort(key=lambda x: (x.frames[0], x.subject, x.predicate, x.object))
    return spans


def _fmt_bbox(b) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in b) + "]"


def _restrict(g: STSG, frames: Sequence[int]) -> STSG:
    keep = set(frames)
    return g.select_frames([f.index for f in g.frames if f.index in keep])


def perturb_stsg(g: STSG, rng: random.Random, pool_categories: Sequence[str] = (), pool_predicates: Sequence[str] = ()) -> Tuple[STSG, str]:
    """Hard negative for L1: corrupt categories, predicates or temporal order."""
    kinds = []
    tracks = g.tracks()
    cats = sorted({m[0][1].category for m in tracks.values()})
    if len(tracks) >= 2 or pool_categories:
        kinds.append("category")
    if g.num_triplets and (len(g.predicates()) >= 2 or pool_predicates):
        kinds.append("predicate")
    if len(g.frames) >= 3:
        kinds.append("time")
    if not kinds:
        return g.copy(), "none"
    kind = rng.choice(kinds)
    h = g.copy()
    if kind == "category":
        names = sorted(tracks)
        cand = sorted(set(cats) | set(pool_categories))
        victim = rng.choice(names)
        old = tracks[victim][0][1].category
        new = rng.choice([c for c in cand if c != old] or [old + " (unrelated)"])
        keys = {(fi, o.id) for fi, o in tracks[victim]}
        rename: Dict[Tuple[int, str], str] = {}
        for f in h.frames:
            ids = set(f.object_ids())
            for o in f.objects:
                if (f.index, o.id) in keys:
                    nid = new
                    k = 1
                    while nid in ids and nid != o.id:
                        k += 1
                        nid = f"{new}-{k}"
                    rename[(f.index, o.id)] = nid
                    ids.add(nid)
                    o.id, o.label, o.track_id = nid, new, None
            f.triplets = [
                Triplet(rename.get((f.index, t.subject), t.subject), t.predicate, rename.get((f.index, t.object), t.object))
                for t in f.triplets
            ]
        h.coref = [CorefLink(c.src_frame, rename.get(c.src_key, c.src), c.dst_frame, rename.get(c.dst_key, c.dst)) for c in h.coref]
    elif kind == "predicate":
        preds = sorted(set(g.predicates()) | set(pool_predicates))
        for f in h.frames:
            f.triplets = [Triplet(t.subject, rng.choice([p for p in preds if p != t.predicate] or [t.predicate]), t.object) for t in f.triplets]
    else:
        # reverse the graph contents but keep each slot's index *and* timestamp,
        # otherwise decreasing timestamps would give the label away
        slots = sorted(h.frames, key=lambda x: x.index)
        contents = [copy.deepcopy(f) for f in slots][::-1]
        for slot, f in zip(slots, contents):
            f.index, f.timestamp = slot.index, slot.timestamp
        h.frames = contents
        h.coref = []
        h.infer_coref()
    return h, kind


# ----------------------------------------------------------------- builder
class GroundingTaskBuilder:
    """Turns :class:`GroundingSource` objects into SFT records (see ``vot.data.sft``)."""

    def __init__(
        self,
        tasks: Sequence[str] = TASKS,
        seed: int = 0,
        max_frames: int = 16,
        l1_negative_ratio: float = 0.5,
        l1_perturb_ratio: float = 0.5,
        max_actions: int = 2,
        max_objects: int = 2,
        precision: int = 1,
        task_max_frames: Optional[Dict[str, int]] = None,
        max_tokens: Optional[int] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        """``max_tokens`` bounds the text of a sample (instruction + target); longer
        L2 samples are rebuilt with fewer frames, other tasks are dropped. Leave room
        for the template and the video/STSG tokens (e.g. 3800 for a 4096 context)."""
        bad = set(tasks) - set(TASKS)
        if bad:
            raise ValueError(f"unknown tasks {bad}")
        self.tasks = list(tasks)
        self.seed = seed
        self.max_frames = max_frames
        self.l1_negative_ratio = l1_negative_ratio
        self.l1_perturb_ratio = l1_perturb_ratio
        self.max_actions = max_actions
        self.max_objects = max_objects
        self.precision = precision
        self.task_max_frames = dict(DEFAULT_TASK_MAX_FRAMES if task_max_frames is None else task_max_frames)
        self.max_tokens = max_tokens
        self.count_tokens = token_counter or approx_token_count
        self.dropped: Dict[str, int] = {}

    def _rng(self, sid: str, task: str) -> random.Random:
        return random.Random(f"{self.seed}:{sid}:{task}")

    def _expr(self, g: STSG) -> str:
        return g.to_expression(precision=self.precision)

    def _base(self, src: GroundingSource, pos: List[int], task: str, human: str, gpt: str) -> Dict:
        rec: Dict = {"id": f"{src.id}-{task}", "task": task}
        if src.video is not None:
            rec["video"] = src.video
        if src.frames is not None:
            rec["frames"] = [src.frames[i] for i in pos]
        if src.frame_times is not None:
            rec["frame_times"] = [src.frame_times[i] for i in pos]
        rec["conversations"] = [{"from": "human", "value": "<video>\n" + human}, {"from": "gpt", "value": gpt}]
        return rec

    def _fits(self, rec: Dict) -> bool:
        if not self.max_tokens:
            return True
        text = "\n".join(t["value"] for t in rec["conversations"])
        return self.count_tokens(text) <= self.max_tokens

    def build(self, src: GroundingSource, pool: Sequence[GroundingSource] = ()) -> List[Dict]:
        views: Dict[int, Tuple[STSG, List[int]]] = {}

        def view(cap: int) -> Tuple[STSG, List[int]]:
            if cap not in views:
                views[cap] = renumber(src, cap)
            return views[cap]

        if view(self.max_frames)[0].is_empty:
            return []
        out: List[Dict] = []
        for task in self.tasks:
            cap = min(self.max_frames, self.task_max_frames.get(task, self.max_frames))
            fn = getattr(self, f"_{task.lower()}")
            while True:
                g, pos = view(cap)
                rec = fn(src, g, pos, self._rng(src.id, task), pool)
                if rec is None or self._fits(rec):
                    break
                if task == "L2" and cap > 2:
                    cap //= 2
                    continue
                self.dropped[task] = self.dropped.get(task, 0) + 1
                rec = None
                break
            if rec is not None:
                out.append(rec)
        return out

    # ---- L1
    def _l1(self, src, g, pos, rng, pool) -> Optional[Dict]:
        prompt = rng.choice(TEMPLATES["L1"]).format(stsg="<stsg>")
        if rng.random() >= self.l1_negative_ratio:
            rec = self._base(src, pos, "L1", prompt, "Yes.")
            rec["stsg"] = g.to_dict()
            return rec
        others = [p for p in pool if p.id != src.id and not p.stsg.is_empty]
        if others and rng.random() >= self.l1_perturb_ratio:
            other = rng.choice(others)
            og, opos = renumber(other, self.max_frames)
            rec = self._base(src, pos, "L1", prompt, "No.")
            rec["stsg"] = og.to_dict()
            rec["negative"] = f"other:{other.id}"
            if other.video is not None:
                rec["stsg_video"] = other.video
            if other.frames is not None:
                rec["stsg_frames"] = [other.frames[i] for i in opos]
            if other.frame_times is not None:
                rec["stsg_frame_times"] = [other.frame_times[i] for i in opos]
            return rec
        pool_cats = sorted({c for p in pool for c in p.stsg.categories()})
        pool_preds = sorted({c for p in pool for c in p.stsg.predicates()})
        neg, kind = perturb_stsg(g, rng, pool_cats, pool_preds)
        if kind == "none":
            rec = self._base(src, pos, "L1", prompt, "Yes.")
            rec["stsg"] = g.to_dict()
            return rec
        rec = self._base(src, pos, "L1", prompt, "No.")
        rec["stsg"] = neg.to_dict()
        rec["negative"] = f"perturb:{kind}"
        return rec

    # ---- L2
    def _l2(self, src, g, pos, rng, pool) -> Dict:
        return self._base(src, pos, "L2", rng.choice(TEMPLATES["L2"]), self._expr(g))

    # ---- L3
    def _action_targets(self, src: GroundingSource, g: STSG, rng: random.Random) -> List[Tuple[str, STSG]]:
        items: List[Tuple[str, STSG]] = []
        persons = [o.id for f in g.frames for o in f.objects if o.category == "person"]
        for a in src.actions:
            frames = [f.index for f in g.frames if f.timestamp is None or a.start is None or a.end is None or a.start <= f.timestamp <= a.end]
            if not frames:
                continue
            ids = list(a.objects) or [i for ids in match_targets([a.label], g).values() for i in ids]
            ids = ids + persons
            sub = _restrict(g, frames).subgraph(ids, hops=0)
            if not sub.is_empty:
                items.append((a.label[0].lower() + a.label[1:], sub))
        if not items:
            for sp in relation_spans(g):
                sub = _restrict(g, sp.frames).subgraph(sp.keys, hops=0, follow_tracks=False)
                items.append((sp.describe(with_time=False), sub))
        rng.shuffle(items)
        return items[: max(1, rng.randint(1, self.max_actions))] if items else []

    def _l3(self, src, g, pos, rng, pool) -> Optional[Dict]:
        items = self._action_targets(src, g, rng)
        if not items:
            return None
        merged = STSG(meta=dict(g.meta))
        frames: Dict[int, FrameGraph] = {}
        for _, sub in items:
            for f in sub.frames:
                tgt = frames.setdefault(f.index, FrameGraph(index=f.index, timestamp=f.timestamp))
                for o in f.objects:
                    if tgt.get(o.id) is None:
                        tgt.objects.append(copy.copy(o))
                for t in f.triplets:
                    if t not in tgt.triplets:
                        tgt.triplets.append(t)
            merged.coref.extend(c for c in sub.coref if c not in merged.coref)
        merged.frames = [frames[k] for k in sorted(frames)]
        actions = ", ".join(f"[{a}]" for a, _ in items)
        human = rng.choice(TEMPLATES["L3"]).format(actions=actions)
        gpt = f"The partial STSG of {actions} is:\n{self._expr(merged)}"
        return self._base(src, pos, "L3", human, gpt)

    # ---- L4
    def _l4(self, src, g, pos, rng, pool) -> Optional[Dict]:
        spans = relation_spans(g)
        tracks = g.tracks()
        involved = sorted({s for sp in spans for s in (sp.subject, sp.object)})
        candidates = [t for t in involved if tracks[t][0][1].category != "person"] or involved or sorted(tracks)
        if not candidates:
            return None
        rng.shuffle(candidates)
        chosen = candidates[: rng.randint(1, min(self.max_objects, len(candidates)))]
        keys = [(fi, o.id) for t in chosen for fi, o in tracks[t]]
        names = ", ".join(f"[{tracks[t][0][1].category}]" for t in chosen)
        desc = [sp.describe() for sp in spans if sp.subject in chosen or sp.object in chosen]
        if desc:
            text = "; ".join(d[0].upper() + d[1:] for d in desc) + "."
        else:
            frames = sorted({fi for t in chosen for fi, _ in tracks[t]})
            text = f"{names} appear(s) from Frame {frames[0]} to Frame {frames[-1]} without notable interaction."
        sub = g.subgraph(keys, hops=1, follow_tracks=False)
        human = rng.choice(TEMPLATES["L4"]).format(objects=names)
        gpt = f"{text}\nThe tracklet(s) of {names}:\n{self._expr(sub)}"
        return self._base(src, pos, "L4", human, gpt)

    # ---- L5
    def _l5(self, src, g, pos, rng, pool) -> Optional[Dict]:
        cands = [(f.index, o) for f in g.frames for o in f.objects if o.bbox is not None and o.bbox[2] > 0 and o.bbox[3] > 0]
        if not cands:
            return None
        fi, obj = rng.choice(cands)
        sub = g.subgraph([(fi, obj.id)], hops=0, follow_tracks=True)
        human = rng.choice(TEMPLATES["L5"]).format(frame=fi, bbox=_fmt_bbox(obj.bbox))
        gpt = f'The object is [{obj.category}] ("{obj.id}"). Its tracklet is:\n{self._expr(sub)}'
        return self._base(src, pos, "L5", human, gpt)
