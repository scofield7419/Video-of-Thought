"""Bounding-box utilities. Boxes are ``(x, y, w, h)`` in percent unless noted."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

BBox = Tuple[float, float, float, float]


def to_xyxy(b: BBox) -> BBox:
    x, y, w, h = b
    return (x, y, x + w, y + h)


def from_xyxy(b: Sequence[float]) -> BBox:
    x1, y1, x2, y2 = b
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def pixel_to_percent(b: Sequence[float], width: int, height: int, xyxy: bool = False) -> BBox:
    """Convert a pixel box (``xywh`` or ``xyxy``) to percent ``xywh``."""
    if xyxy:
        b = from_xyxy(b)
    x, y, w, h = b
    return (100.0 * x / width, 100.0 * y / height, 100.0 * w / width, 100.0 * h / height)


def percent_to_pixel(b: BBox, width: int, height: int, xyxy: bool = False) -> BBox:
    x, y, w, h = b
    px = (x * width / 100.0, y * height / 100.0, w * width / 100.0, h * height / 100.0)
    return to_xyxy(px) if xyxy else px


def area(b: BBox) -> float:
    return max(0.0, b[2]) * max(0.0, b[3])


def center(b: BBox) -> Tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def intersection(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def iou(a: Optional[BBox], b: Optional[BBox]) -> float:
    if a is None or b is None:
        return 0.0
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def containment(inner: BBox, outer: BBox) -> float:
    """Fraction of ``inner`` covered by ``outer``."""
    a = area(inner)
    return intersection(inner, outer) / a if a > 0 else 0.0


def spatial_predicate(subj: BBox, obj: BBox, near_thresh: float = 5.0) -> str:
    """A coarse geometric predicate between two boxes (used by the detector-based
    STSG parser when no semantic relation model is available)."""
    if containment(subj, obj) > 0.85:
        return "inside"
    if containment(obj, subj) > 0.85:
        return "containing"
    if iou(subj, obj) > 0.05:
        return "overlapping"
    sx, sy = center(subj)
    ox, oy = center(obj)
    dx, dy = ox - sx, oy - sy
    gap_x = max(0.0, max(subj[0], obj[0]) - min(subj[0] + subj[2], obj[0] + obj[2]))
    gap_y = max(0.0, max(subj[1], obj[1]) - min(subj[1] + subj[3], obj[1] + obj[3]))
    if max(gap_x, gap_y) < near_thresh:
        return "next to"
    if abs(dx) >= abs(dy):
        return "on the left of" if dx > 0 else "on the right of"
    return "above" if dy > 0 else "below"
