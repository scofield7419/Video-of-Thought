#!/usr/bin/env python
"""Convert an official VideoQA benchmark release into the unified JSONL format.

Examples::

    python scripts/data/prepare_videoqa.py --benchmark nextqa --raw data/raw/nextqa --split val \
        --out data/nextqa/val.jsonl
    python scripts/data/prepare_videoqa.py --benchmark activitynet_qa \
        --raw data/raw/activitynet-qa/dataset --split test --out data/activitynet_qa/test.jsonl
    python scripts/data/prepare_videoqa.py --benchmark socialiq --raw data/raw/socialiq/qa_val.jsonl \
        --split val --out data/socialiq/val.jsonl --opt ways=2,4
    python scripts/data/prepare_videoqa.py --benchmark nextqa --raw data/nextqa/val.jsonl \
        --check-videos data/raw/nextqa/NExTVideo
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vot.data.benchmarks import BENCHMARKS, get_benchmark  # noqa: E402
from vot.data.videoqa import resolve_video, save_samples  # noqa: E402


def parse_opts(opts: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for o in opts:
        if "=" not in o:
            raise SystemExit(f"--opt expects key=value, got '{o}'")
        k, v = o.split("=", 1)
        if "," in v:
            out[k] = [int(x) if x.isdigit() else x for x in v.split(",") if x]
        elif v.isdigit():
            out[k] = int(v)
        elif v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True, help=f"one of {sorted(BENCHMARKS)}")
    ap.add_argument("--raw", required=True, nargs="+", help="raw annotation file(s) or directory")
    ap.add_argument("--split", default=None, help="split name (default: benchmark default)")
    ap.add_argument("--out", default=None, help="output .jsonl (omit to only print statistics)")
    ap.add_argument("--opt", action="append", default=[], help="extra converter kwargs, key=value")
    ap.add_argument("--check-videos", default=None, metavar="VIDEO_ROOT", help="report samples whose video is missing")
    args = ap.parse_args()

    spec = get_benchmark(args.benchmark)
    split = args.split or spec.default_split
    raw = args.raw[0] if len(args.raw) == 1 else args.raw
    samples = spec.convert(raw, split=split, **parse_opts(args.opt))

    cats = Counter(s.category for s in samples)
    labeled = sum(s.answer is not None for s in samples)
    print(f"[{spec.name}/{split}] {len(samples)} samples ({labeled} labeled), {len({s.video for s in samples})} videos")
    for k, v in sorted(cats.items(), key=lambda kv: str(kv[0])):
        print(f"  {k}: {v}")

    if args.check_videos:
        missing = [s for s in samples if resolve_video(s, args.check_videos) is None]
        print(f"  missing videos: {len(missing)} / {len(samples)}")
        for s in missing[:10]:
            print(f"    {s.qid}: {s.video}")

    if args.out:
        n = save_samples(samples, args.out)
        print(f"wrote {n} samples -> {args.out}")


if __name__ == "__main__":
    main()
