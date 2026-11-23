"""Convert Action Genome annotations into per-video STSGs (vot.stsg format).

Inputs (from https://www.actiongenome.org, see the AG toolkit):
    annotations/person_bbox.pkl                    {"VID.mp4/000123.png": {"bbox": [[x1,y1,x2,y2],...], "bbox_size": (W,H), ...}}
    annotations/object_bbox_and_relationship.pkl   {"VID.mp4/000123.png": [{"class", "bbox": (x,y,w,h), "visible",
                                                     "attention_relationship", "spatial_relationship",
                                                     "contacting_relationship", "metadata": {"set": ...}}, ...]}
    frames/VID.mp4/000123.png                      (dumped with the AG toolkit; used for frame alignment)

Output: one JSONL line per video::

    {"video_id": "VID", "split": "train", "video": "VID.mp4", "frames": ["VID.mp4/000123.png", ...],
     "frame_numbers": [123, ...], "frame_times": [4.1, ...] | null, "stsg": {...STSG.to_dict()...}}

Every relation is person -> object (AG annotates human-object interactions);
``frame_times`` is filled when ``--videos_dir`` lets us read each video's fps.

    python scripts/data/convert_action_genome.py --ag_root data/action_genome --out data/action_genome/stsg.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vot.stsg.geometry import pixel_to_percent  # noqa: E402
from vot.stsg.schema import STSG, FrameGraph, SGObject, Triplet  # noqa: E402

REL_KEYS = ("attention_relationship", "spatial_relationship", "contacting_relationship")


def object_id(cls: str) -> str:
    """AG classes such as ``cup/glass/bottle`` -> id ``cup``; the full name is kept as label."""
    return cls.split("/")[0].strip().replace(" ", "_")


def object_label(cls: str) -> str:
    return cls.replace("/", " or ").replace("_", " ")


def predicate(rel: str) -> str:
    return rel.replace("_", " ").strip()


def probe_fps(path: str) -> Optional[float]:
    try:
        import decord  # type: ignore

        return float(decord.VideoReader(path).get_avg_fps())
    except Exception:
        pass
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps else None
    except Exception:
        return None


def convert(person: Dict, objects: Dict, videos_dir: Optional[str] = None, min_frames: int = 1) -> List[Dict]:
    by_video: Dict[str, List[str]] = defaultdict(list)
    for key in objects:
        by_video[key.split("/")[0]].append(key)
    out = []
    for vid_file, keys in sorted(by_video.items()):
        keys.sort(key=lambda k: int(os.path.splitext(k.split("/")[1])[0]))
        fps = probe_fps(os.path.join(videos_dir, vid_file)) if videos_dir else None
        frames, frame_files, numbers, split = [], [], [], None
        for k in keys:
            num = int(os.path.splitext(k.split("/")[1])[0])
            pinfo = person.get(k, {})
            size = pinfo.get("bbox_size")
            if size is None:
                continue
            w, h = size
            fg = FrameGraph(index=len(frames) + 1, timestamp=(num / fps) if fps else None)
            pboxes = pinfo.get("bbox")
            pbox = None
            if pboxes is not None and len(pboxes) > 0:
                pbox = tuple(round(float(v), 2) for v in pixel_to_percent([float(x) for x in pboxes[0]], float(w), float(h), xyxy=True))
            rels = []
            for o in objects[k]:
                split = (o.get("metadata") or {}).get("set", split)
                if not o.get("visible", True) or o.get("bbox") is None:
                    continue
                oid = object_id(o["class"])
                if fg.get(oid) is not None:
                    continue
                fg.objects.append(SGObject(id=oid, label=object_label(o["class"]),
                                           bbox=tuple(round(float(v), 2) for v in pixel_to_percent([float(x) for x in o["bbox"]], float(w), float(h)))))
                for rk in REL_KEYS:
                    for r in o.get(rk) or []:
                        rels.append(Triplet("person", predicate(r), oid))
            if pbox is not None or rels:
                fg.objects.insert(0, SGObject(id="person", bbox=pbox))
            fg.triplets = list(dict.fromkeys(rels))
            if not fg.objects:
                continue
            frames.append(fg)
            frame_files.append(k)
            numbers.append(num)
        if len(frames) < min_frames:
            continue
        stsg = STSG(frames=frames, meta={"video_id": vid_file.split(".")[0], "source": "action_genome"}).infer_coref()
        out.append({
            "video_id": vid_file.split(".")[0],
            "split": split,
            "video": vid_file,
            "frames": frame_files,
            "frame_numbers": numbers,
            "frame_times": [f.timestamp for f in frames] if fps else None,
            "stsg": stsg.to_dict(),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ag_root", required=True, help="Action Genome root containing annotations/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos_dir", default=None, help="Charades videos, to compute timestamps from fps")
    ap.add_argument("--split", default=None, choices=[None, "train", "test"])
    ap.add_argument("--min_frames", type=int, default=2)
    args = ap.parse_args()
    ann = os.path.join(args.ag_root, "annotations")
    with open(os.path.join(ann, "person_bbox.pkl"), "rb") as f:
        person = pickle.load(f)
    with open(os.path.join(ann, "object_bbox_and_relationship.pkl"), "rb") as f:
        objects = pickle.load(f)
    rows = convert(person, objects, args.videos_dir, args.min_frames)
    if args.split:
        rows = [r for r in rows if r["split"] == args.split]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_frames = sum(len(r["frames"]) for r in rows)
    print(f"wrote {len(rows)} videos / {n_frames} annotated frames to {args.out}")


if __name__ == "__main__":
    main()
