"""Video decoding and frame sampling.

The paper samples frames uniformly at 8 fps for MotionEpic. Long videos would
exceed any context budget, so sampling is ``fps``-based with a ``max_frames`` cap:
when ``duration * fps > max_frames`` the clip is covered by ``max_frames``
uniformly spaced frames instead.

Decoders are tried in the order decord -> PyAV -> OpenCV. A directory of image
files or a list of PIL images is also accepted.
"""

from __future__ import annotations

import base64
import io
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class VideoClip:
    """Sampled frames of a video. ``frames[k-1]`` is "Frame k" in STSG expressions."""

    frames: List[Image.Image]
    timestamps: List[float]
    source: Optional[str] = None
    native_fps: Optional[float] = None
    duration: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.frames) != len(self.timestamps):
            raise ValueError("frames and timestamps must have the same length")

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> Optional[tuple]:
        return self.frames[0].size if self.frames else None

    def frame(self, k: int) -> Image.Image:
        """1-based access, consistent with STSG frame numbering."""
        return self.frames[k - 1]

    def subsample(self, max_frames: int) -> "VideoClip":
        n = len(self.frames)
        if max_frames <= 0 or n <= max_frames:
            return self
        idx = uniform_indices(n, max_frames)
        return VideoClip(
            frames=[self.frames[i] for i in idx],
            timestamps=[self.timestamps[i] for i in idx],
            source=self.source,
            native_fps=self.native_fps,
            duration=self.duration,
            meta={**self.meta, "subsampled_from": n, "kept_positions": [i + 1 for i in idx]},
        )

    def resized(self, max_side: int) -> "VideoClip":
        """Downscale frames so that the longer side is at most ``max_side`` pixels."""
        out = []
        for f in self.frames:
            w, h = f.size
            s = max_side / max(w, h)
            out.append(f.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC) if s < 1 else f)
        return VideoClip(out, list(self.timestamps), self.source, self.native_fps, self.duration, dict(self.meta))

    def fingerprint(self) -> str:
        """Cheap identifier for caching: source path + sampled timestamps + size."""
        import hashlib

        h = hashlib.sha1()
        if self.source:
            h.update(os.path.abspath(self.source).encode())
            h.update(",".join(f"{t:.3f}" for t in self.timestamps).encode())
            h.update(str(self.size).encode())
        else:
            for f in self.frames:
                h.update(f.tobytes())
        return h.hexdigest()


def uniform_indices(n: int, k: int) -> List[int]:
    """``k`` indices that split ``range(n)`` into equal segments (segment centres)."""
    if k >= n:
        return list(range(n))
    return [min(n - 1, int((i + 0.5) * n / k)) for i in range(k)]


def sample_timestamps(
    duration: float,
    fps: float,
    max_frames: Optional[int] = None,
    start: Optional[float] = None,
    end: Optional[float] = None,
) -> List[float]:
    """Timestamps (seconds) to sample from ``[start, end)`` at ``fps``.

    At least one frame is returned. If ``fps * (end - start)`` exceeds
    ``max_frames``, ``max_frames`` segment-centre timestamps are used instead.
    """
    s = max(0.0, start or 0.0)
    e = duration if end is None else min(end, duration)
    if e <= s:
        e = min(duration, s + 1e-3) if duration > s else s + 1e-3
    span = e - s
    n = max(1, int(math.floor(span * fps + 1e-6)))
    if max_frames and n > max_frames:
        n = max_frames
        step = span / n
        return [s + (i + 0.5) * step for i in range(n)]
    step = 1.0 / fps
    # centre the grid inside the span
    offset = (span - (n - 1) * step) / 2.0 if n > 1 else span / 2.0
    return [s + offset + i * step for i in range(n)]


def _from_image_dir(
    path: str, fps: float, max_frames: Optional[int], frames_fps: float, start: Optional[float] = None, end: Optional[float] = None
) -> VideoClip:
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise FileNotFoundError(f"No image frames found in {path}")
    duration = len(files) / frames_fps
    ts = sample_timestamps(duration, fps, max_frames, start, end)
    idx = sorted({min(len(files) - 1, int(t * frames_fps)) for t in ts})
    frames = [Image.open(os.path.join(path, files[i])).convert("RGB") for i in idx]
    return VideoClip(frames, [i / frames_fps for i in idx], path, frames_fps, duration)


def _decord(path: str, ts_fn) -> VideoClip:
    import decord  # type: ignore

    vr = decord.VideoReader(path, ctx=decord.cpu(0))
    native = float(vr.get_avg_fps()) or 30.0
    total = len(vr)
    duration = total / native
    ts = ts_fn(duration)
    idx = sorted({min(total - 1, int(round(t * native))) for t in ts})
    arr = vr.get_batch(idx).asnumpy()
    frames = [Image.fromarray(a).convert("RGB") for a in arr]
    return VideoClip(frames, [i / native for i in idx], path, native, duration)


def _pyav(path: str, ts_fn) -> VideoClip:
    import av  # type: ignore

    with av.open(path) as container:
        stream = container.streams.video[0]
        native = float(stream.average_rate) if stream.average_rate else 30.0
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / 1e6
        else:
            duration = (stream.frames or 0) / native
        targets = ts_fn(duration)
        t0 = float(stream.start_time * stream.time_base) if stream.start_time is not None and stream.time_base is not None else 0.0
        frames, stamps, ti = [], [], 0
        for n_dec, frame in enumerate(container.decode(stream)):
            t = float(frame.pts * frame.time_base) - t0 if frame.pts is not None else n_dec / native
            while ti < len(targets) and t + 0.5 / native >= targets[ti]:
                if not stamps or stamps[-1] != t:
                    frames.append(frame.to_image().convert("RGB"))
                    stamps.append(t)
                ti += 1
            if ti >= len(targets):
                break
    return VideoClip(frames, stamps, path, native, duration)


def _opencv(path: str, ts_fn) -> VideoClip:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"OpenCV cannot open {path}")
    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / native
    idx = sorted({min(total - 1, int(round(t * native))) for t in ts_fn(duration)})
    frames, stamps = [], []
    want, pos = set(idx), 0
    while pos <= idx[-1]:
        ok, fr = cap.read()
        if not ok:
            break
        if pos in want:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            stamps.append(pos / native)
        pos += 1
    cap.release()
    return VideoClip(frames, stamps, path, native, duration)


def load_video(
    source: Union[str, Sequence[Image.Image], VideoClip],
    fps: float = 8.0,
    max_frames: Optional[int] = 32,
    start: Optional[float] = None,
    end: Optional[float] = None,
    backend: str = "auto",
    frames_fps: float = 1.0,
) -> VideoClip:
    """Decode and sample a video.

    Args:
        source: video file, directory of frame images, list of PIL images or a
            :class:`VideoClip` (returned subsampled to ``max_frames``).
        fps: target sampling rate (the paper uses 8 fps for MotionEpic).
        max_frames: hard cap on the number of sampled frames.
        start, end: optional segment in seconds (e.g. STAR / VLEP question spans).
        backend: ``"auto"``, ``"decord"``, ``"pyav"`` or ``"opencv"``.
        frames_fps: frame rate of an image directory / image list input.
    """
    if isinstance(source, VideoClip):
        return source.subsample(max_frames) if max_frames else source
    if not isinstance(source, str):
        imgs = [im.convert("RGB") for im in source]
        duration = len(imgs) / frames_fps
        ts = sample_timestamps(duration, fps, max_frames, start, end)
        idx = sorted({min(len(imgs) - 1, int(t * frames_fps)) for t in ts})
        return VideoClip([imgs[i] for i in idx], [i / frames_fps for i in idx], None, frames_fps, duration)
    if os.path.isdir(source):
        return _from_image_dir(source, fps, max_frames, frames_fps, start, end)
    if not os.path.exists(source):
        raise FileNotFoundError(source)

    ts_fn = lambda duration: sample_timestamps(duration, fps, max_frames, start, end)  # noqa: E731
    order = {"auto": ["decord", "pyav", "opencv"], "decord": ["decord"], "pyav": ["pyav"], "opencv": ["opencv"]}[backend]
    errors = []
    for name in order:
        fn = {"decord": _decord, "pyav": _pyav, "opencv": _opencv}[name]
        try:
            clip = fn(source, ts_fn)
            if len(clip) == 0:
                raise IOError("no frames decoded")
            clip.meta.update({"sample_fps": fps, "max_frames": max_frames, "start": start, "end": end})
            return clip
        except ImportError as e:
            errors.append(f"{name}: not installed ({e})")
        except Exception as e:  # pragma: no cover - decoder specific failures
            errors.append(f"{name}: {e}")
    raise RuntimeError(f"Could not decode {source}: " + "; ".join(errors))


def encode_image(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Base64 string of an image (for API backends)."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
