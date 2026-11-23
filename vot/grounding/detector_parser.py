"""STSG construction from an open-vocabulary detector + IoU tracker.

Pipeline per clip:

1. detect the query vocabulary (Step-1 targets + ``extra_vocabulary``) in every
   sampled frame with a zero-shot detector (default: Grounding DINO via
   ``transformers``), keeping boxes above ``box_threshold`` and applying
   per-label NMS;
2. link detections over time with Hungarian matching on IoU (per label, with a
   ``max_gap``-frame memory) -> temporal coreference edges;
3. add predicate edges between each object and its ``max_relations`` nearest
   neighbours, using geometric predicates (:func:`vot.stsg.geometry.spatial_predicate`)
   or a user-supplied ``relation_fn`` (e.g. a VLM relation classifier).

This yields precise boxes that API models cannot produce reliably, which is why
Step-2 can run in ``select``/``match`` mode on top of it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from ..stsg.geometry import iou, pixel_to_percent, spatial_predicate
from ..stsg.matching import normalize_phrase
from ..stsg.schema import STSG, CorefLink, FrameGraph, SGObject, Triplet
from ..video.io import VideoClip
from .base import STSGProvider

Detection = Tuple[str, float, Tuple[float, float, float, float]]  # (label, score, xyxy pixels)
DetectFn = Callable[[Image.Image, Sequence[str]], List[Detection]]
RelationFn = Callable[[Image.Image, SGObject, SGObject], Optional[str]]


def label_of(phrase: str) -> str:
    toks = normalize_phrase(phrase)
    return " ".join(toks) if toks else phrase.strip().lower()


class GroundingDinoDetector:
    """Zero-shot detector wrapper around ``AutoModelForZeroShotObjectDetection``."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: Optional[str] = None,
                 box_threshold: float = 0.35, text_threshold: float = 0.25) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def __call__(self, image: Image.Image, vocabulary: Sequence[str]) -> List[Detection]:
        text = " . ".join(v.lower() for v in vocabulary) + " ."
        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        post = self.processor.post_process_grounded_object_detection
        kw = dict(input_ids=inputs["input_ids"], text_threshold=self.text_threshold, target_sizes=[image.size[::-1]])
        try:
            res = post(outputs, threshold=self.box_threshold, **kw)[0]
        except TypeError:  # transformers < 4.51 used `box_threshold`
            res = post(outputs, box_threshold=self.box_threshold, **kw)[0]
        labels = res.get("text_labels", res.get("labels"))
        out: List[Detection] = []
        for lab, score, box in zip(labels, res["scores"].tolist(), res["boxes"].tolist()):
            lab = str(lab).strip()
            # map sub-phrase outputs back to the closest vocabulary entry
            best = max(vocabulary, key=lambda v: len(set(normalize_phrase(v)) & set(normalize_phrase(lab))))
            out.append((best if set(normalize_phrase(best)) & set(normalize_phrase(lab)) else lab, float(score), tuple(box)))
        return out


def _nms(dets: List[Detection], thr: float = 0.6) -> List[Detection]:
    from ..stsg.geometry import from_xyxy

    keep: List[Detection] = []
    for d in sorted(dets, key=lambda x: -x[1]):
        if all(d[0] != k[0] or iou(from_xyxy(d[2]), from_xyxy(k[2])) < thr for k in keep):
            keep.append(d)
    return keep


def _assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    if not cost or not cost[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except ImportError:  # greedy fallback
        pairs = sorted(((cost[i][j], i, j) for i in range(len(cost)) for j in range(len(cost[0]))))
        used_i, used_j, out = set(), set(), []
        for _, i, j in pairs:
            if i not in used_i and j not in used_j:
                out.append((i, j))
                used_i.add(i)
                used_j.add(j)
        return out


@dataclass
class _Track:
    tid: int
    label: str
    bbox: Tuple[float, float, float, float]
    last: int


class DetectorSTSGParser(STSGProvider):
    name = "detector"

    def __init__(
        self,
        detector: Optional[DetectFn] = None,
        extra_vocabulary: Sequence[str] = ("person",),
        max_objects: int = 12,
        iou_threshold: float = 0.3,
        max_gap: int = 2,
        max_relations: int = 3,
        relation_fn: Optional[RelationFn] = None,
        **detector_kwargs,
    ) -> None:
        self._detector = detector
        self._detector_kwargs = detector_kwargs
        self._lock = threading.Lock()
        self.extra_vocabulary = list(extra_vocabulary)
        self.max_objects = max_objects
        self.iou_threshold = iou_threshold
        self.max_gap = max_gap
        self.max_relations = max_relations
        self.relation_fn = relation_fn

    @property
    def detector(self) -> DetectFn:
        with self._lock:
            if self._detector is None:
                self._detector = GroundingDinoDetector(**self._detector_kwargs)
        return self._detector

    def __call__(self, clip: VideoClip, *, targets: Optional[Sequence[str]] = None, question=None, video_id=None) -> STSG:
        vocab: List[str] = []
        for v in list(targets or []) + self.extra_vocabulary:
            if v and label_of(v) not in {label_of(x) for x in vocab}:
                vocab.append(v)
        W, H = clip.size or (1, 1)

        per_frame: List[List[Tuple[str, float, Tuple[float, float, float, float]]]] = []
        for img in clip.frames:
            dets = _nms([(label_of(l), s, b) for l, s, b in self.detector(img, vocab)])[: self.max_objects]
            per_frame.append([(l, s, pixel_to_percent(b, W, H, xyxy=True)) for l, s, b in dets])

        tracks: List[_Track] = []
        assignment: List[List[int]] = []
        for k, dets in enumerate(per_frame):
            tids = [-1] * len(dets)
            for lab in {d[0] for d in dets}:
                di = [i for i, d in enumerate(dets) if d[0] == lab]
                active = [t for t in tracks if t.label == lab and k - t.last <= self.max_gap]
                cost = [[1.0 - iou(dets[i][2], t.bbox) for t in active] for i in di]
                for a, b in _assign(cost):
                    if cost[a][b] <= 1.0 - self.iou_threshold:
                        tids[di[a]] = active[b].tid
                        active[b].bbox, active[b].last = dets[di[a]][2], k
            for i, d in enumerate(dets):
                if tids[i] < 0:
                    t = _Track(len(tracks), d[0], d[2], k)
                    tracks.append(t)
                    tids[i] = t.tid
            assignment.append(tids)

        count: Dict[str, int] = {}
        for t in tracks:
            count[t.label] = count.get(t.label, 0) + 1
        seen: Dict[str, int] = {}
        name: Dict[int, str] = {}
        for t in tracks:
            if count[t.label] > 1:
                seen[t.label] = seen.get(t.label, 0) + 1
                name[t.tid] = f"{t.label}-{seen[t.label]}"
            else:
                name[t.tid] = t.label

        g = STSG(meta={"source": "detector", "video_id": video_id, "vocabulary": vocab})
        for k, (dets, tids) in enumerate(zip(per_frame, assignment), start=1):
            objs = [SGObject(id=name[t], bbox=tuple(round(v, 2) for v in d[2]), label=d[0], score=round(d[1], 3), track_id=name[t])
                    for d, t in zip(dets, tids)]
            trips: List[Triplet] = []
            for o in objs:
                others = sorted((x for x in objs if x.id != o.id), key=lambda x: _center_dist(o, x))[: self.max_relations]
                for x in others:
                    pred = self.relation_fn(clip.frames[k - 1], o, x) if self.relation_fn else spatial_predicate(o.bbox, x.bbox)  # type: ignore[arg-type]
                    if pred:
                        t = Triplet(o.id, pred, x.id)
                        if t not in trips:
                            trips.append(t)
            g.frames.append(FrameGraph(index=k, objects=objs, triplets=trips, timestamp=clip.timestamps[k - 1]))
        g.infer_coref()
        return g


def _center_dist(a: SGObject, b: SGObject) -> float:
    ax, ay = a.bbox[0] + a.bbox[2] / 2, a.bbox[1] + a.bbox[3] / 2  # type: ignore[index]
    bx, by = b.bbox[0] + b.bbox[2] / 2, b.bbox[1] + b.bbox[3] / 2  # type: ignore[index]
    return (ax - bx) ** 2 + (ay - by) ** 2


__all__ = ["CorefLink", "DetectorSTSGParser", "GroundingDinoDetector"]
