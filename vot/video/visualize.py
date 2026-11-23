"""Draw STSG boxes, ids and triplets on sampled frames (for inspecting Step-2 tracklets)."""

from __future__ import annotations

import colorsys
import hashlib
import os
from typing import List, Optional

from PIL import Image, ImageDraw

from ..stsg.geometry import percent_to_pixel
from ..stsg.schema import STSG
from .io import VideoClip


def _color(key: str) -> tuple:
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16) / 0xFFFFFF
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_frame(img: Image.Image, stsg: STSG, k: int, show_triplets: bool = True) -> Image.Image:
    out = img.convert("RGB").copy()
    f = stsg.frame(k)
    if f is None:
        return out
    d = ImageDraw.Draw(out)
    W, H = out.size
    for o in f.objects:
        if o.bbox is None:
            continue
        x1, y1, x2, y2 = percent_to_pixel(o.bbox, W, H, xyxy=True)
        c = _color(o.identity)
        d.rectangle([x1, y1, x2, y2], outline=c, width=max(2, W // 200))
        d.rectangle([x1, max(0, y1 - 14), x1 + 7 * len(o.id) + 6, y1], fill=c)
        d.text((x1 + 3, max(0, y1 - 13)), o.id, fill=(0, 0, 0))
    if show_triplets and f.triplets:
        lines = [f"{t.subject} -{t.predicate}-> {t.object}" for t in f.triplets[:8]]
        d.rectangle([0, H - 14 * len(lines) - 6, W, H], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            d.text((4, H - 14 * (len(lines) - i) - 2), line, fill=(255, 255, 255))
    d.text((4, 4), f"Frame {k}", fill=(255, 255, 0))
    return out


def render_stsg(clip: VideoClip, stsg: STSG, out_dir: Optional[str] = None, grid_cols: int = 4) -> List[Image.Image]:
    """Annotated frames; optionally saved as ``frame_XX.jpg`` plus a ``grid.jpg`` overview."""
    frames = [draw_frame(img, stsg, k) for k, img in enumerate(clip.frames, start=1)]
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for k, im in enumerate(frames, start=1):
            im.save(os.path.join(out_dir, f"frame_{k:02d}.jpg"), quality=90)
        if frames:
            w, h = frames[0].size
            s = min(1.0, 480 / w)
            tw, th = int(w * s), int(h * s)
            rows = (len(frames) + grid_cols - 1) // grid_cols
            grid = Image.new("RGB", (tw * min(grid_cols, len(frames)), th * rows))
            for i, im in enumerate(frames):
                grid.paste(im.resize((tw, th)), ((i % grid_cols) * tw, (i // grid_cols) * th))
            grid.save(os.path.join(out_dir, "grid.jpg"), quality=90)
    return frames
