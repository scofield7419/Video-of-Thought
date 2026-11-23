"""Graph-level operations used by the reasoning engine."""

from __future__ import annotations

import copy
from typing import Optional

from .schema import STSG, FrameGraph


def union(a: STSG, b: STSG) -> STSG:
    """Node/edge union of two STSGs over the same sampled frames.

    Nodes are identified by ``(frame, id)``; attributes of ``a`` take precedence.
    """
    frames = {f.index: copy.deepcopy(f) for f in a.frames}
    for fb in b.frames:
        fa = frames.setdefault(fb.index, FrameGraph(index=fb.index, timestamp=fb.timestamp))
        ids = set(fa.object_ids())
        for o in fb.objects:
            if o.id not in ids:
                fa.objects.append(copy.copy(o))
                ids.add(o.id)
        for t in fb.triplets:
            if t not in fa.triplets:
                fa.triplets.append(t)
    coref = list(dict.fromkeys(list(a.coref) + list(b.coref)))
    out = STSG(frames=[frames[k] for k in sorted(frames)], coref=coref, meta={**b.meta, **a.meta})
    return out.repair()


def align_to_reference(partial: STSG, reference: STSG, copy_boxes: bool = True) -> STSG:
    """Snap a generated partial STSG onto a reference STSG.

    When the model selects nodes from a given full STSG (``step2_mode="select"``)
    it may re-type coordinates imprecisely; boxes of nodes that exist in the
    reference at the same frame are copied from it.
    """
    g = partial.copy()
    if not copy_boxes:
        return g
    for f in g.frames:
        rf = reference.frame(f.index)
        if rf is None:
            continue
        for o in f.objects:
            ro = rf.get(o.id)
            if ro is not None and ro.bbox is not None:
                o.bbox = ro.bbox
                o.label = o.label or ro.label
                o.track_id = o.track_id or ro.track_id
    if reference.meta:
        g.meta = {**reference.meta, **g.meta}
    return g


def neighbor_scene(partial: STSG, full: Optional[STSG], hops: int = 1) -> STSG:
    """Target tracklets plus their ``hops``-hop neighbour scenes (Step-3 input).

    Without a full STSG the generated partial graph is used as is (MotionEpic is
    trained to include interacting objects in the tracklet expression).
    """
    if full is None or full.is_empty or hops <= 0:
        return partial
    keys = [(f.index, oid) for f in partial.frames for oid in f.object_ids() if full.node((f.index, oid)) is not None]
    if not keys:
        return partial
    return union(partial, full.subgraph(keys, hops=hops))
