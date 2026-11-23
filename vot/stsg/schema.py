"""Spatial-Temporal Scene Graph (STSG) data model.

Follows Sec. 3.2 of the paper. A video is evenly sampled into frames; the k-th
sampled frame carries a scene graph ``G_k = (V_k, E_k)``:

* each node ``v_{k,i} = (c_i, f_i, b_i)`` is an object proposal with a category
  label ``c_i``, a neural representation ``f_i`` (filled lazily by a featurizer)
  and a bounding box ``b_i = (x, y, w, h)``;
* each predicate edge ``e_{k,i,j}`` is a ``subject - predicate - object`` triplet;
* temporal coreference edges ``e_{k-1 -> k}`` link the same object across
  adjacent frames, mimicking tracking.

Conventions used throughout the code base:

* Frame indices are **1-based** positions in the *sampled* clip ("Frame 1", ...),
  matching the STSG expressions in the paper (Fig. 3 / Appendix A).
* Bounding boxes are ``(x, y, w, h)`` with the top-left corner ``(x, y)``, all
  expressed in **percent of the frame width/height** (range ``[0, 100]``). This
  keeps expressions short and resolution independent; see :mod:`vot.stsg.geometry`
  for conversions.
* Object ids are unique within a frame. Several instances of one category are
  disambiguated with a numeric suffix (``man-1``, ``man-2``).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

BBox = Tuple[float, float, float, float]
NodeKey = Tuple[int, str]  # (frame index, object id)

_SUFFIX_RE = re.compile(r"[-_#]\d+$")


def category_of(object_id: str) -> str:
    """``"car-2" -> "car"``; ids without numeric suffix are returned unchanged."""
    return _SUFFIX_RE.sub("", object_id.strip()).strip()


@dataclass
class SGObject:
    """An object proposal (node) in a single-frame scene graph."""

    id: str
    bbox: Optional[BBox] = None
    label: Optional[str] = None
    score: Optional[float] = None
    track_id: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    feature: Any = field(default=None, repr=False, compare=False)

    @property
    def category(self) -> str:
        return self.label if self.label else category_of(self.id)

    @property
    def identity(self) -> str:
        """Cross-frame identity used when coreference edges are inferred."""
        return self.track_id if self.track_id else self.id

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id}
        if self.bbox is not None:
            d["bbox"] = [float(v) for v in self.bbox]
        if self.label is not None and self.label != category_of(self.id):
            d["label"] = self.label
        if self.score is not None:
            d["score"] = float(self.score)
        if self.track_id is not None and self.track_id != self.id:
            d["track_id"] = self.track_id
        if self.attributes:
            d["attributes"] = list(self.attributes)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SGObject":
        bbox = d.get("bbox")
        return cls(
            id=str(d["id"]),
            bbox=tuple(float(v) for v in bbox) if bbox is not None else None,  # type: ignore[arg-type]
            label=d.get("label"),
            score=d.get("score"),
            track_id=d.get("track_id"),
            attributes=list(d.get("attributes", [])),
        )


@dataclass(frozen=True)
class Triplet:
    """A predicate edge ``subject --predicate--> object`` inside one frame."""

    subject: str
    predicate: str
    object: str

    def to_list(self) -> List[str]:
        return [self.subject, self.predicate, self.object]


@dataclass
class FrameGraph:
    """Scene graph ``G_k = (V_k, E_k)`` of the k-th sampled frame."""

    index: int
    objects: List[SGObject] = field(default_factory=list)
    triplets: List[Triplet] = field(default_factory=list)
    timestamp: Optional[float] = None

    def object_ids(self) -> List[str]:
        return [o.id for o in self.objects]

    def get(self, object_id: str) -> Optional[SGObject]:
        for o in self.objects:
            if o.id == object_id:
                return o
        return None

    def neighbors(self, object_id: str) -> Set[str]:
        out: Set[str] = set()
        for t in self.triplets:
            if t.subject == object_id:
                out.add(t.object)
            elif t.object == object_id:
                out.add(t.subject)
        return out

    def validate(self) -> None:
        ids = self.object_ids()
        if len(ids) != len(set(ids)):
            raise ValueError(f"Frame {self.index}: duplicated object ids {ids}")

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "index": self.index,
            "objects": [o.to_dict() for o in self.objects],
            "triplets": [t.to_list() for t in self.triplets],
        }
        if self.timestamp is not None:
            d["timestamp"] = float(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FrameGraph":
        return cls(
            index=int(d["index"]),
            objects=[SGObject.from_dict(o) for o in d.get("objects", [])],
            triplets=[Triplet(*map(str, t)) for t in d.get("triplets", [])],
            timestamp=d.get("timestamp"),
        )


@dataclass(frozen=True)
class CorefLink:
    """Temporal coreference edge between two frames (normally ``k-1 -> k``)."""

    src_frame: int
    src: str
    dst_frame: int
    dst: str

    @property
    def src_key(self) -> NodeKey:
        return (self.src_frame, self.src)

    @property
    def dst_key(self) -> NodeKey:
        return (self.dst_frame, self.dst)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[NodeKey, NodeKey] = {}

    def add(self, x: NodeKey) -> None:
        self.parent.setdefault(x, x)

    def find(self, x: NodeKey) -> NodeKey:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: NodeKey, b: NodeKey) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic root: earliest frame wins
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


@dataclass
class STSG:
    """A spatial-temporal scene graph: a sequence of frame graphs + coreference edges."""

    frames: List[FrameGraph] = field(default_factory=list)
    coref: List[CorefLink] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[FrameGraph]:
        return iter(self.frames)

    @property
    def is_empty(self) -> bool:
        return all(len(f.objects) == 0 for f in self.frames)

    @property
    def num_nodes(self) -> int:
        return sum(len(f.objects) for f in self.frames)

    @property
    def num_triplets(self) -> int:
        return sum(len(f.triplets) for f in self.frames)

    def frame_indices(self) -> List[int]:
        return [f.index for f in self.frames]

    def frame(self, index: int) -> Optional[FrameGraph]:
        for f in self.frames:
            if f.index == index:
                return f
        return None

    def node(self, key: NodeKey) -> Optional[SGObject]:
        f = self.frame(key[0])
        return f.get(key[1]) if f is not None else None

    def categories(self) -> List[str]:
        seen: Dict[str, None] = {}
        for f in self.frames:
            for o in f.objects:
                seen.setdefault(o.category, None)
        return list(seen)

    def predicates(self) -> List[str]:
        seen: Dict[str, None] = {}
        for f in self.frames:
            for t in f.triplets:
                seen.setdefault(t.predicate, None)
        return list(seen)

    def copy(self) -> "STSG":
        return copy.deepcopy(self)

    def sort(self) -> "STSG":
        self.frames.sort(key=lambda f: f.index)
        self.coref.sort(key=lambda c: (c.src_frame, c.dst_frame, c.src, c.dst))
        return self

    def validate(self) -> None:
        """Raise ``ValueError`` on structural inconsistencies."""
        seen_idx: Set[int] = set()
        for f in self.frames:
            if f.index in seen_idx:
                raise ValueError(f"Duplicated frame index {f.index}")
            seen_idx.add(f.index)
            f.validate()
            ids = set(f.object_ids())
            for t in f.triplets:
                if t.subject not in ids or t.object not in ids:
                    raise ValueError(f"Frame {f.index}: triplet {t} references unknown object")
        for c in self.coref:
            if self.node(c.src_key) is None or self.node(c.dst_key) is None:
                raise ValueError(f"Coreference edge {c} references unknown node")

    def repair(self) -> "STSG":
        """Best-effort cleaning of model-generated graphs.

        Adds bbox-less nodes for objects that only appear in triplets, merges
        duplicated frames, drops duplicated objects/triplets and dangling
        coreference links. Returns ``self``.
        """
        merged: Dict[int, FrameGraph] = {}
        for f in self.frames:
            tgt = merged.setdefault(f.index, FrameGraph(index=f.index, timestamp=f.timestamp))
            known = set(tgt.object_ids())
            for o in f.objects:
                if o.id not in known:
                    tgt.objects.append(o)
                    known.add(o.id)
            for t in f.triplets:
                for oid in (t.subject, t.object):
                    if oid not in known:
                        tgt.objects.append(SGObject(id=oid))
                        known.add(oid)
                if t not in tgt.triplets:
                    tgt.triplets.append(t)
        self.frames = [merged[k] for k in sorted(merged)]
        uniq: Dict[CorefLink, None] = {}
        for c in self.coref:
            if self.node(c.src_key) is not None and self.node(c.dst_key) is not None:
                uniq.setdefault(c, None)
        self.coref = list(uniq)
        return self

    # ------------------------------------------------------- temporal structure
    def infer_coref(self, overwrite: bool = False) -> "STSG":
        """Link nodes with the same identity (``track_id`` or id) in consecutive frames."""
        if overwrite:
            self.coref = []
        existing = set(self.coref)
        ordered = sorted(self.frames, key=lambda f: f.index)
        for prev, cur in zip(ordered[:-1], ordered[1:]):
            prev_by_identity = {o.identity: o.id for o in prev.objects}
            linked_dst = {c.dst for c in self.coref if c.src_frame == prev.index and c.dst_frame == cur.index}
            for o in cur.objects:
                if o.id in linked_dst:
                    continue
                src = prev_by_identity.get(o.identity)
                if src is not None:
                    link = CorefLink(prev.index, src, cur.index, o.id)
                    if link not in existing:
                        self.coref.append(link)
                        existing.add(link)
        return self

    def _union_find(self) -> _UnionFind:
        uf = _UnionFind()
        for f in self.frames:
            for o in f.objects:
                uf.add((f.index, o.id))
        for c in self.coref:
            if self.node(c.src_key) is not None and self.node(c.dst_key) is not None:
                uf.union(c.src_key, c.dst_key)
        return uf

    def tracks(self) -> Dict[str, List[Tuple[int, SGObject]]]:
        """Group nodes into tracklets by following coreference edges.

        Returns ``{track name: [(frame index, node), ...]}`` ordered by time; the
        track name is the id of the node where the tracklet starts.
        """
        uf = self._union_find()
        groups: Dict[NodeKey, List[Tuple[int, SGObject]]] = {}
        for f in sorted(self.frames, key=lambda x: x.index):
            for o in f.objects:
                groups.setdefault(uf.find((f.index, o.id)), []).append((f.index, o))
        out: Dict[str, List[Tuple[int, SGObject]]] = {}
        for root, members in sorted(groups.items(), key=lambda kv: kv[0]):
            name = root[1]
            while name in out:  # two tracks starting with the same id
                name = name + "'"
            out[name] = members
        return out

    def track_of(self, key: NodeKey) -> List[NodeKey]:
        uf = self._union_find()
        root = uf.find(key)
        return [(f.index, o.id) for f in self.frames for o in f.objects if uf.find((f.index, o.id)) == root]

    # ------------------------------------------------------------- sub-graphs
    def subgraph(self, seeds: Iterable[Union[str, NodeKey]], hops: int = 0, follow_tracks: bool = True) -> "STSG":
        """Partial STSG around ``seeds``.

        ``seeds`` are object ids (matched in every frame) or ``(frame, id)`` keys.
        With ``follow_tracks`` the seeds are propagated along coreference edges,
        i.e. the result contains the full tracklets of the seeds. ``hops > 0``
        additionally keeps the h-hop neighbours of the seeds inside each frame
        ("neighbor scenes" in Step-3 of VoT), together with the triplets among
        the kept nodes.
        """
        seed_ids: Set[str] = set()
        seed_keys: Set[NodeKey] = set()
        for s in seeds:
            if isinstance(s, tuple):
                seed_keys.add((int(s[0]), str(s[1])))
            else:
                seed_ids.add(str(s))
        for f in self.frames:
            for o in f.objects:
                if o.id in seed_ids:
                    seed_keys.add((f.index, o.id))
        if follow_tracks and seed_keys:
            uf = self._union_find()
            roots = {uf.find(k) for k in seed_keys}
            for f in self.frames:
                for o in f.objects:
                    if uf.find((f.index, o.id)) in roots:
                        seed_keys.add((f.index, o.id))

        new_frames: List[FrameGraph] = []
        for f in self.frames:
            keep = {oid for (fi, oid) in seed_keys if fi == f.index}
            frontier = set(keep)
            for _ in range(max(0, hops)):
                nxt: Set[str] = set()
                for oid in frontier:
                    nxt |= f.neighbors(oid)
                frontier = nxt - keep
                keep |= nxt
            if not keep:
                continue
            new_frames.append(
                FrameGraph(
                    index=f.index,
                    timestamp=f.timestamp,
                    objects=[copy.copy(o) for o in f.objects if o.id in keep],
                    triplets=[t for t in f.triplets if t.subject in keep and t.object in keep],
                )
            )
        kept = {(f.index, oid) for f in new_frames for oid in f.object_ids()}
        coref = [c for c in self.coref if c.src_key in kept and c.dst_key in kept]
        return STSG(frames=new_frames, coref=coref, meta=dict(self.meta))

    def select_frames(self, indices: Iterable[int]) -> "STSG":
        """Keep only the given frame indices; coreference edges are re-linked
        across removed frames so tracklets stay connected."""
        idx = set(indices)
        keep_frames = [copy.deepcopy(f) for f in self.frames if f.index in idx]
        uf = self._union_find()
        out = STSG(frames=keep_frames, coref=[], meta=dict(self.meta))
        last_seen: Dict[NodeKey, NodeKey] = {}  # track root -> latest kept node of that track
        for cur in sorted(keep_frames, key=lambda f: f.index):
            seen_now: Dict[NodeKey, NodeKey] = {}
            for o in cur.objects:
                r = uf.find((cur.index, o.id))
                if r in last_seen:
                    src = last_seen[r]
                    out.coref.append(CorefLink(src[0], src[1], cur.index, o.id))
                seen_now.setdefault(r, (cur.index, o.id))
            last_seen.update(seen_now)
        return out

    def subsample(self, max_frames: int) -> "STSG":
        """Uniformly keep at most ``max_frames`` frames (first and last kept)."""
        n = len(self.frames)
        if max_frames <= 0 or n <= max_frames:
            return self.copy()
        ordered = sorted(self.frames, key=lambda f: f.index)
        if max_frames == 1:
            picks = [ordered[n // 2].index]
        else:
            picks = [ordered[round(i * (n - 1) / (max_frames - 1))].index for i in range(max_frames)]
        return self.select_frames(picks)

    def drop_empty_frames(self) -> "STSG":
        return self.select_frames([f.index for f in self.frames if f.objects])

    # ---------------------------------------------------------- serialization
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meta": dict(self.meta),
            "frames": [f.to_dict() for f in self.frames],
            "coref": [[c.src_frame, c.src, c.dst_frame, c.dst] for c in self.coref],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "STSG":
        return cls(
            frames=[FrameGraph.from_dict(f) for f in d.get("frames", [])],
            coref=[CorefLink(int(c[0]), str(c[1]), int(c[2]), str(c[3])) for c in d.get("coref", [])],
            meta=dict(d.get("meta", {})),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, s: str) -> "STSG":
        return cls.from_dict(json.loads(s))

    def to_expression(self, **kwargs: Any) -> str:
        from .expression import format_stsg

        return format_stsg(self, **kwargs)

    @classmethod
    def from_expression(cls, text: str, **kwargs: Any) -> "STSG":
        from .expression import parse_stsg

        return parse_stsg(text, **kwargs)

    @classmethod
    def from_frames(
        cls,
        frames: Sequence[Dict[str, Any]],
        infer_coref: bool = True,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "STSG":
        """Convenience constructor from ``[{"objects": {...}, "triplets": [...]}, ...]``.

        ``objects`` may be a ``{id: bbox}`` mapping or a list of object dicts.
        """
        fgs: List[FrameGraph] = []
        for i, fr in enumerate(frames, start=1):
            objs = fr.get("objects", [])
            if isinstance(objs, dict):
                obj_list = [SGObject(id=k, bbox=tuple(v) if v is not None else None) for k, v in objs.items()]  # type: ignore[arg-type]
            else:
                obj_list = [o if isinstance(o, SGObject) else SGObject.from_dict(o) for o in objs]
            fgs.append(
                FrameGraph(
                    index=int(fr.get("index", i)),
                    objects=obj_list,
                    triplets=[t if isinstance(t, Triplet) else Triplet(*t) for t in fr.get("triplets", [])],
                    timestamp=fr.get("timestamp"),
                )
            )
        g = cls(frames=fgs, meta=dict(meta or {}))
        return g.infer_coref() if infer_coref else g
