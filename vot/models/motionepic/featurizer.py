"""STSG -> padded tensors for the recurrent graph transformer.

The featurizer is the CPU-side half of STSG encoding: it lays an STSG out on a
``[T, N]`` grid, crops one RoI image per node from its frame (so that the frozen
CLIP tower can produce the node feature ``f_i`` on GPU inside the model), and
records labels, predicates, boxes, timestamps and coreference pointers. The
neural half lives in :meth:`MotionEpicForCausalLM.encode_stsg`.

Frame alignment: STSG frame ``k`` is the k-th frame of the clip it was produced
from. If both the STSG frame and the clip carry real timestamps, the temporally
closest clip frame is used instead (robust to annotations at other rates); clips
with nominal timestamps (``meta["timestamps_reliable"] = False``) are aligned by
index only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from ...stsg.geometry import percent_to_pixel
from ...stsg.schema import STSG, FrameGraph, SGObject
from ...video.io import VideoClip

CLIP_MEAN_RGB = (122, 116, 104)
#: ``VideoClip.meta`` key; ``False`` marks nominal timestamps that must not be used for alignment
TIMESTAMPS_RELIABLE = "timestamps_reliable"


@dataclass
class STSGInputs:
    """One featurized STSG (no batch dimension)."""

    node_roi: torch.Tensor  # [T, N] long: row in roi_pixels or -1
    node_label: torch.Tensor  # [T, N] long: index into labels or -1
    node_box: torch.Tensor  # [T, N, 4] float, xywh in [0, 1]
    node_mask: torch.Tensor  # [T, N] bool
    frame_time: torch.Tensor  # [T] float seconds
    coref_prev: torch.Tensor  # [T, N] long: node index in frame t-1 or -1
    edges: torch.Tensor  # [E, 3] long (t, subject, object)
    edge_pred: torch.Tensor  # [E] long: index into predicates
    labels: List[str]
    predicates: List[str]
    roi_pixels: Optional[torch.Tensor] = None  # [M, 3, H, W]
    node_ids: List[List[str]] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return self.node_mask.shape[0]

    @property
    def max_nodes(self) -> int:
        return self.node_mask.shape[1]


@dataclass
class STSGBatch:
    node_roi: torch.Tensor  # [B, T, N]
    node_label: torch.Tensor  # [B, T, N] -> phrases
    node_box: torch.Tensor  # [B, T, N, 4]
    node_mask: torch.Tensor  # [B, T, N]
    frame_time: torch.Tensor  # [B, T]
    frame_mask: torch.Tensor  # [B, T]
    coref_prev: torch.Tensor  # [B, T, N]
    edge_index: torch.Tensor  # [E, 4] (graph, frame, subject, object)
    edge_phrase: torch.Tensor  # [E] -> phrases
    phrases: List[str]
    roi_pixels: Optional[torch.Tensor] = None  # [M, 3, H, W]

    def __len__(self) -> int:
        return self.node_mask.shape[0]

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> "STSGBatch":
        def mv(x: Optional[torch.Tensor], cast: bool = False) -> Optional[torch.Tensor]:
            if x is None:
                return None
            return x.to(device=device, dtype=dtype) if (cast and dtype is not None) else x.to(device)

        return STSGBatch(
            node_roi=mv(self.node_roi),
            node_label=mv(self.node_label),
            node_box=mv(self.node_box),
            node_mask=mv(self.node_mask),
            frame_time=mv(self.frame_time),
            frame_mask=mv(self.frame_mask),
            coref_prev=mv(self.coref_prev),
            edge_index=mv(self.edge_index),
            edge_phrase=mv(self.edge_phrase),
            phrases=list(self.phrases),
            roi_pixels=mv(self.roi_pixels, cast=True),
        )


def square_pad(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w == h:
        return img
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), CLIP_MEAN_RGB)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas


def _node_priority(frame: FrameGraph) -> List[SGObject]:
    """Order nodes so that truncation to ``max_nodes`` drops the least informative ones."""
    in_rel = {x for t in frame.triplets for x in (t.subject, t.object)}
    order = {o.id: i for i, o in enumerate(frame.objects)}
    return sorted(frame.objects, key=lambda o: (o.id not in in_rel, o.bbox is None, order[o.id]))


class STSGFeaturizer:
    def __init__(
        self,
        image_processor=None,
        max_frames: int = 32,
        max_nodes: int = 24,
        roi_min_size: int = 8,
        default_fps: float = 8.0,
    ) -> None:
        self.image_processor = image_processor
        self.max_frames = max_frames
        self.max_nodes = max_nodes
        self.roi_min_size = roi_min_size
        self.default_fps = default_fps

    # ------------------------------------------------------------------ single
    def _frame_image(self, frame: FrameGraph, clip: Optional[VideoClip]) -> Tuple[Optional[Image.Image], Optional[float]]:
        if clip is None or len(clip) == 0:
            return None, None
        if frame.timestamp is not None and clip.timestamps and clip.meta.get(TIMESTAMPS_RELIABLE, True):
            j = min(range(len(clip)), key=lambda i: abs(clip.timestamps[i] - frame.timestamp))
            return clip.frames[j], clip.timestamps[j]
        if 1 <= frame.index <= len(clip):
            return clip.frames[frame.index - 1], clip.timestamps[frame.index - 1]
        return None, None

    def _crop(self, img: Image.Image, bbox) -> Image.Image:
        w, h = img.size
        x1, y1, x2, y2 = percent_to_pixel(bbox, w, h, xyxy=True)
        m = self.roi_min_size
        if x2 - x1 < m:
            cx = (x1 + x2) / 2
            x1, x2 = cx - m / 2, cx + m / 2
        if y2 - y1 < m:
            cy = (y1 + y2) / 2
            y1, y2 = cy - m / 2, cy + m / 2
        x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
        x2, y2 = min(w, max(x1 + 1, int(round(x2)))), min(h, max(y1 + 1, int(round(y2))))
        return square_pad(img.crop((x1, y1, x2, y2)))

    def encode(self, stsg: STSG, clip: Optional[VideoClip] = None) -> STSGInputs:
        g = stsg.subsample(self.max_frames)
        g = g.select_frames(sorted(g.frame_indices()))  # coref between layout-adjacent frames
        frames = sorted(g.frames, key=lambda f: f.index)
        t_len = max(1, len(frames))
        n_len = max(1, min(self.max_nodes, max((len(f.objects) for f in frames), default=1)))

        node_roi = torch.full((t_len, n_len), -1, dtype=torch.long)
        node_label = torch.full((t_len, n_len), -1, dtype=torch.long)
        node_box = torch.zeros(t_len, n_len, 4)
        node_mask = torch.zeros(t_len, n_len, dtype=torch.bool)
        frame_time = torch.zeros(t_len)
        coref_prev = torch.full((t_len, n_len), -1, dtype=torch.long)
        labels: Dict[str, int] = {}
        preds: Dict[str, int] = {}
        edges: List[Tuple[int, int, int]] = []
        edge_pred: List[int] = []
        crops: List[Image.Image] = []
        layout: List[Dict[str, int]] = []
        node_ids: List[List[str]] = []

        for t, fr in enumerate(frames):
            img, ts = self._frame_image(fr, clip)
            if fr.timestamp is not None:
                frame_time[t] = fr.timestamp
            elif ts is not None:
                frame_time[t] = ts
            else:
                frame_time[t] = (fr.index - 1) / self.default_fps
            kept = _node_priority(fr)[:n_len]
            pos: Dict[str, int] = {}
            for n, o in enumerate(kept):
                pos[o.id] = n
                node_mask[t, n] = True
                node_label[t, n] = labels.setdefault(o.category, len(labels))
                if o.bbox is not None:
                    node_box[t, n] = torch.tensor([v / 100.0 for v in o.bbox])
                    if img is not None and self.image_processor is not None:
                        node_roi[t, n] = len(crops)
                        crops.append(self._crop(img, o.bbox))
            for tr in fr.triplets:
                if tr.subject in pos and tr.object in pos and tr.subject != tr.object:
                    edges.append((t, pos[tr.subject], pos[tr.object]))
                    edge_pred.append(preds.setdefault(tr.predicate, len(preds)))
            layout.append(pos)
            node_ids.append([o.id for o in kept])

        index_of = {fr.index: t for t, fr in enumerate(frames)}
        for c in g.coref:
            ts_, td = index_of.get(c.src_frame), index_of.get(c.dst_frame)
            if ts_ is None or td is None or td != ts_ + 1:
                continue
            si, di = layout[ts_].get(c.src), layout[td].get(c.dst)
            if si is not None and di is not None:
                coref_prev[td, di] = si

        roi_pixels = None
        if crops:
            roi_pixels = self.image_processor(images=crops, return_tensors="pt")["pixel_values"]
        return STSGInputs(
            node_roi=node_roi,
            node_label=node_label,
            node_box=node_box,
            node_mask=node_mask,
            frame_time=frame_time,
            coref_prev=coref_prev,
            edges=torch.tensor(edges, dtype=torch.long).view(-1, 3),
            edge_pred=torch.tensor(edge_pred, dtype=torch.long),
            labels=list(labels),
            predicates=list(preds),
            roi_pixels=roi_pixels,
            node_ids=node_ids,
        )

    # ------------------------------------------------------------------- batch
    @staticmethod
    def collate(items: Sequence[STSGInputs]) -> STSGBatch:
        b = len(items)
        t_len = max(1, max((x.num_frames for x in items), default=1))
        n_len = max(1, max((x.max_nodes for x in items), default=1))
        node_roi = torch.full((b, t_len, n_len), -1, dtype=torch.long)
        node_label = torch.full((b, t_len, n_len), -1, dtype=torch.long)
        node_box = torch.zeros(b, t_len, n_len, 4)
        node_mask = torch.zeros(b, t_len, n_len, dtype=torch.bool)
        frame_time = torch.zeros(b, t_len)
        frame_mask = torch.zeros(b, t_len, dtype=torch.bool)
        coref_prev = torch.full((b, t_len, n_len), -1, dtype=torch.long)
        phrases: Dict[str, int] = {}
        edge_index, edge_phrase, rois = [], [], []
        roi_offset = 0
        for i, x in enumerate(items):
            t, n = x.num_frames, x.max_nodes
            lab_map = torch.tensor([phrases.setdefault(s, len(phrases)) for s in x.labels] or [0], dtype=torch.long)
            pred_map = torch.tensor([phrases.setdefault(s, len(phrases)) for s in x.predicates] or [0], dtype=torch.long)
            node_label[i, :t, :n] = torch.where(x.node_label >= 0, lab_map[x.node_label.clamp(min=0)], -1)
            node_roi[i, :t, :n] = torch.where(x.node_roi >= 0, x.node_roi + roi_offset, -1)
            node_box[i, :t, :n] = x.node_box
            node_mask[i, :t, :n] = x.node_mask
            frame_time[i, :t] = x.frame_time
            frame_mask[i, :t] = True
            coref_prev[i, :t, :n] = x.coref_prev
            if x.edges.numel():
                edge_index.append(torch.cat([torch.full((len(x.edges), 1), i, dtype=torch.long), x.edges], dim=1))
                edge_phrase.append(pred_map[x.edge_pred])
            if x.roi_pixels is not None:
                rois.append(x.roi_pixels)
                roi_offset += x.roi_pixels.shape[0]
        return STSGBatch(
            node_roi=node_roi,
            node_label=node_label,
            node_box=node_box,
            node_mask=node_mask,
            frame_time=frame_time,
            frame_mask=frame_mask,
            coref_prev=coref_prev,
            edge_index=torch.cat(edge_index) if edge_index else torch.zeros(0, 4, dtype=torch.long),
            edge_phrase=torch.cat(edge_phrase) if edge_phrase else torch.zeros(0, dtype=torch.long),
            phrases=list(phrases),
            roi_pixels=torch.cat(rois) if rois else None,
        )
