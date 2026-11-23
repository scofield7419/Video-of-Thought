"""Textual STSG expressions (paper Fig. 3 and Appendix A).

MotionEpic reads and *generates* STSGs as text, e.g.::

    Frame 1: {
        Objects: [{"car-1": [0.0, 13.4, 7.0, 8.1]}, {"street": [9.0, 13.5, 16.5, 15.0]}],
        Triplets: [("car-1", "on the left", "street")]
    }
    Frame 2: {
        ...
    }
    Temporal Coreference Edge: [
        {Frame 1 -> 2: [("car-1", "car-1"), ("street", "street")]}
    ]

:func:`format_stsg` renders an :class:`~vot.stsg.schema.STSG` in this grammar and
:func:`parse_stsg` reads it back. The parser is deliberately tolerant because
it also consumes model generations: it accepts single/double quotes, ``[]`` or
``()`` for triplets, ``->``/``→``/``to`` in coreference headers, markdown code
fences, missing sections and a JSON variant (``{"frames": [...]}``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .schema import STSG, CorefLink, FrameGraph, SGObject, Triplet

STSG_FORMAT_SPEC = """\
An STSG expression lists one scene graph per sampled frame, followed by the temporal coreference edges:
Frame <k>: {
    Objects: [{"<object id>": [x, y, w, h]}, ...],
    Triplets: [("<subject id>", "<predicate>", "<object id>"), ...]
}
...
Temporal Coreference Edge: [
    {Frame <k-1> -> <k>: [("<object id in frame k-1>", "<same object's id in frame k>"), ...]},
    ...
]
Rules: <k> is the 1-based frame number shown with each frame. Object ids are short category names, \
with a numeric suffix to separate instances of the same category (e.g. "man-1", "man-2"); keep the \
same id for the same physical object across frames. [x, y, w, h] is the bounding box: top-left corner \
(x, y), width w and height h, all in percent (0-100) of the frame width/height, one decimal place. \
Triplets only use ids listed in the same frame."""


# --------------------------------------------------------------------- format
def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _fmt_num(v: float, precision: int) -> str:
    return f"{v:.{precision}f}"


def format_frame(frame: FrameGraph, precision: int = 1, compact: bool = False, with_timestamp: bool = False) -> str:
    objs = []
    for o in frame.objects:
        if o.bbox is None:
            objs.append("{" + _q(o.id) + ": null}")
        else:
            objs.append("{" + _q(o.id) + ": [" + ", ".join(_fmt_num(v, precision) for v in o.bbox) + "]}")
    trips = ["(" + ", ".join(_q(x) for x in t.to_list()) + ")" for t in frame.triplets]
    header = f"Frame {frame.index}"
    if with_timestamp and frame.timestamp is not None:
        header += f" (t={frame.timestamp:.2f}s)"
    objects = "Objects: [" + ", ".join(objs) + "]"
    triplets = "Triplets: [" + ", ".join(trips) + "]"
    if compact:
        return f"{header}: {{{objects}, {triplets}}}"
    return f"{header}: {{\n    {objects},\n    {triplets}\n}}"


def format_coref(coref: List[CorefLink], compact: bool = False) -> str:
    groups: Dict[Tuple[int, int], List[CorefLink]] = {}
    for c in coref:
        groups.setdefault((c.src_frame, c.dst_frame), []).append(c)
    items = []
    for (a, b), links in sorted(groups.items()):
        pairs = ", ".join(f"({_q(c.src)}, {_q(c.dst)})" for c in links)
        items.append(f"{{Frame {a} -> {b}: [{pairs}]}}")
    if compact:
        return "Temporal Coreference Edge: [" + ", ".join(items) + "]"
    body = ",\n".join("    " + it for it in items)
    return "Temporal Coreference Edge: [\n" + body + ("\n" if body else "") + "]"


def format_stsg(
    stsg: STSG,
    precision: int = 1,
    compact: bool = False,
    with_coref: bool = True,
    with_timestamp: bool = False,
    max_frames: Optional[int] = None,
) -> str:
    """Render an STSG in the expression grammar of the paper."""
    g = stsg.subsample(max_frames) if max_frames else stsg
    frames = sorted(g.frames, key=lambda f: f.index)
    parts = [format_frame(f, precision, compact, with_timestamp) for f in frames]
    if with_coref and g.coref:
        parts.append(format_coref(g.coref, compact))
    return "\n".join(parts)


# ---------------------------------------------------------------------- parse
_FENCE_RE = re.compile(r"```[a-zA-Z]*\n?|```")
_FRAME_HDR_RE = re.compile(r"Frame\s*(\d+)\s*(?:\([^)]*\))?\s*\**\s*:?\s*\**\s*\{", re.IGNORECASE)
_COREF_HDR_RE = re.compile(r"Temporal\s+Coreference\s+Edges?\s*\**\s*:?\s*\**\s*\[", re.IGNORECASE)
_COREF_ITEM_RE = re.compile(
    r"Frame\s*(\d+)\s*(?:->|→|=>|to)\s*(?:Frame\s*)?(\d+)\s*:\s*\[", re.IGNORECASE
)
_STR = r"""["'“”‘’]([^"'“”‘’]+?)["'“”‘’]"""
_NUM = r"-?\d+(?:\.\d+)?"
_OBJ_RE = re.compile(_STR + r"\s*:\s*(\[[^\]]*\]|null|None)")
_TRIPLET_RE = re.compile(r"[\(\[]\s*" + _STR + r"\s*,\s*" + _STR + r"\s*,\s*" + _STR + r"\s*[\)\]]")
_PAIR_RE = re.compile(r"[\(\[]\s*" + _STR + r"\s*,\s*" + _STR + r"\s*[\)\]]")
_BARE_STR_RE = re.compile(_STR)
_OBJECTS_KEY_RE = re.compile(r"Objects\s*:\s*\[", re.IGNORECASE)
_TRIPLETS_KEY_RE = re.compile(r"Triplets\s*:\s*\[", re.IGNORECASE)


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> Tuple[str, int]:
    """Return the content between ``text[start] == open_ch`` and its matching
    closing bracket (or the rest of the string if unbalanced) and the end index."""
    assert text[start] == open_ch
    depth = 0
    in_str: Optional[str] = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if ch == in_str and text[i - 1] != "\\":
                in_str = None
            continue
        if ch == '"':
            in_str = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
    return text[start + 1 :], len(text)


def _parse_bbox(raw: str, scale: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float, float, float]]:
    nums = [float(x) for x in re.findall(_NUM, raw)]
    if len(nums) != 4:
        return None
    x, y, w, h = nums
    if scale is not None:
        sw, sh = scale
        x, w = x * sw, w * sw
        y, h = y * sh, h * sh
    clamp = lambda v: min(100.0, max(0.0, v))  # noqa: E731
    x, y = clamp(x), clamp(y)
    w, h = max(0.0, min(w, 100.0 - x)), max(0.0, min(h, 100.0 - y))
    return (round(x, 3), round(y, 3), round(w, 3), round(h, 3))


def _section(block: str, key_re: re.Pattern) -> Optional[str]:
    m = key_re.search(block)
    if not m:
        return None
    content, _ = _balanced(block, m.end() - 1, "[", "]")
    return content


def _parse_frame_block(index: int, block: str, scale: Optional[Tuple[float, float]]) -> FrameGraph:
    frame = FrameGraph(index=index)
    objects_txt = _section(block, _OBJECTS_KEY_RE)
    seen = set()
    if objects_txt is not None:
        matched = list(_OBJ_RE.finditer(objects_txt))
        if matched:
            for m in matched:
                oid = m.group(1).strip()
                if oid in seen:
                    continue
                seen.add(oid)
                raw = m.group(2)
                frame.objects.append(SGObject(id=oid, bbox=_parse_bbox(raw, scale) if raw.startswith("[") else None))
        else:
            for m in _BARE_STR_RE.finditer(objects_txt):
                oid = m.group(1).strip()
                if oid not in seen:
                    seen.add(oid)
                    frame.objects.append(SGObject(id=oid))
    triplets_txt = _section(block, _TRIPLETS_KEY_RE)
    if triplets_txt is not None:
        for m in _TRIPLET_RE.finditer(triplets_txt):
            t = Triplet(m.group(1).strip(), m.group(2).strip(), m.group(3).strip())
            if t not in frame.triplets:
                frame.triplets.append(t)
    return frame


def _json_frame_index(v: Any, default: int) -> int:
    m = re.search(r"\d+", str(v)) if v is not None else None
    return int(m.group()) if m else default


def _json_objects(objs: Any) -> List[Dict[str, Any]]:
    if isinstance(objs, dict):
        objs = [{"id": k, "bbox": v} for k, v in objs.items()]
    if not isinstance(objs, (list, tuple)):
        return []
    out = []
    for o in objs:
        if isinstance(o, str):
            out.append({"id": o})
        elif isinstance(o, dict) and "id" not in o and len(o) == 1:  # {"man-1": [x, y, w, h]}
            (k, v), = o.items()
            out.append({"id": k, "bbox": v})
        elif isinstance(o, dict) and "id" in o:
            box = o.get("bbox", o.get("box"))
            out.append({"id": o["id"], "bbox": box, "label": o.get("label")})
    for o in out:
        b = o.get("bbox")
        if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(x, (int, float)) for x in b)):
            o["bbox"] = None
    return out


def _json_triplets(trips: Any) -> List[List[str]]:
    out = []
    for t in trips or []:
        if isinstance(t, dict):
            t = [t.get("subject", t.get("s")), t.get("predicate", t.get("p", t.get("relation"))), t.get("object", t.get("o"))]
        if isinstance(t, (list, tuple)) and len(t) == 3 and all(isinstance(x, str) for x in t):
            out.append(list(t))
    return out


def _parse_json_variant(text: str) -> Optional[STSG]:
    s = text.strip()
    if not (s.startswith("{") and '"frames"' in s):
        return None
    try:
        data = json.loads(s)
        frames = [
            {
                "index": _json_frame_index(fr.get("index", fr.get("frame")), i),
                "objects": _json_objects(fr.get("objects")),
                "triplets": _json_triplets(fr.get("triplets")),
            }
            for i, fr in enumerate(data.get("frames", []), start=1)
            if isinstance(fr, dict)
        ]
        coref = [c for c in data.get("coref", []) if isinstance(c, (list, tuple)) and len(c) == 4]
        return STSG.from_dict({"frames": frames, "coref": coref})
    except (ValueError, TypeError, AttributeError, KeyError):
        return None


def _rescale(g: STSG, scale: Tuple[float, float]) -> STSG:
    sw, sh = scale
    for f in g.frames:
        for o in f.objects:
            if o.bbox is not None:
                x, y, w, h = o.bbox
                o.bbox = _parse_bbox(f"{x * sw} {y * sh} {w * sw} {h * sh}", None)
    return g


_BOX_RE = re.compile(r"\[\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\]")


def detect_box_units(text: str) -> str:
    """Guess the unit of the boxes in ``text``: ``"percent"``, ``"pixel"`` or ``"unit"`` ([0, 1])."""
    vals = [float(v) for m in _BOX_RE.finditer(text) for v in m.groups()]
    if not vals:
        return "percent"
    hi = max(vals)
    if hi > 100.5:
        return "pixel"
    if hi <= 1.0 and any(0 < v < 1 for v in vals):
        return "unit"
    return "percent"


def parse_stsg(
    text: str,
    infer_missing_coref: bool = True,
    pixel_size: Optional[Tuple[int, int]] = None,
    repair: bool = True,
    image_size: Optional[Tuple[int, int]] = None,
) -> STSG:
    """Parse an STSG expression (possibly embedded in free text).

    Args:
        text: model output or serialized expression.
        infer_missing_coref: if the text has no coreference section, link
            objects that keep the same id in consecutive frames.
        pixel_size: ``(width, height)`` if the boxes are known to be in pixels;
            they are converted to percent. When ``None`` boxes are assumed to be
            in percent unless ``image_size`` is given.
        repair: run :meth:`STSG.repair` (adds nodes referenced only by
            triplets, removes duplicates/dangling links).
        image_size: ``(width, height)`` of the frames the model saw. Enables unit
            auto-detection: boxes in pixels or in ``[0, 1]`` are converted to percent
            (models do not always follow the requested convention).

    Returns an empty STSG when nothing can be parsed.
    """
    text = _FENCE_RE.sub("", text or "")
    if pixel_size is None and image_size is not None:
        units = detect_box_units(text)
        if units == "pixel":
            pixel_size = image_size
        elif units == "unit":
            pixel_size = (1, 1)
    scale = None
    if pixel_size is not None:
        scale = (100.0 / pixel_size[0], 100.0 / pixel_size[1])

    js = _parse_json_variant(text)
    if js is not None:
        if scale is not None:
            _rescale(js, scale)
        if repair:
            js.repair()
        return js.infer_coref() if infer_missing_coref and not js.coref else js

    frames: List[FrameGraph] = []
    pos = 0
    while True:
        m = _FRAME_HDR_RE.search(text, pos)
        if not m:
            break
        block, end = _balanced(text, m.end() - 1, "{", "}")
        frames.append(_parse_frame_block(int(m.group(1)), block, scale))
        pos = end

    # the coreference section follows the frame blocks; earlier mentions are prose
    coref_m = _COREF_HDR_RE.search(text, pos if frames else 0)
    coref: List[CorefLink] = []
    if coref_m:
        coref_txt, _ = _balanced(text, coref_m.end() - 1, "[", "]")
        for m in _COREF_ITEM_RE.finditer(coref_txt):
            content, _ = _balanced(coref_txt, m.end() - 1, "[", "]")
            a, b = int(m.group(1)), int(m.group(2))
            for p in _PAIR_RE.finditer(content):
                coref.append(CorefLink(a, p.group(1).strip(), b, p.group(2).strip()))

    g = STSG(frames=frames, coref=coref)
    if repair:
        g.repair()
    if infer_missing_coref and not coref_m:
        g.infer_coref()
    return g


def extract_bracketed(text: str) -> List[str]:
    """Return the contents of top-level ``[...]`` spans that are not STSG lists."""
    out: List[str] = []
    i = 0
    while i < len(text):
        if text[i] == "[":
            content, end = _balanced(text, i, "[", "]")
            out.append(content.strip())
            i = end
        else:
            i += 1
    return out


def stsg_summary(stsg: STSG) -> Dict[str, Any]:
    return {
        "frames": len(stsg.frames),
        "nodes": stsg.num_nodes,
        "triplets": stsg.num_triplets,
        "coref": len(stsg.coref),
        "categories": stsg.categories(),
    }
