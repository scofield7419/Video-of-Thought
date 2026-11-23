"""Command line interface: ``vot run | eval | stsg``.

Examples::

    vot run --video demo.mp4 --question "Why is the man running?" \\
        --options "He is late" "He is chased" "He is exercising" --backend openai:gpt-4o

    vot eval --benchmark nextqa --data data/nextqa/val.jsonl --video-root data/nextqa/videos \\
        --backend anthropic:claude-opus-5-5 --method vot --workers 8 --out results/nextqa_vot.jsonl

    vot stsg --video demo.mp4 --provider vlm --backend openai:gpt-4o --out demo_stsg.json --viz viz/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Optional


def _kv(items: Optional[List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for it in items or []:
        k, _, v = it.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", default="openai:gpt-4o", help="openai:<model> | anthropic:<model> | motionepic:<ckpt> | mock")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible server URL (vLLM, DashScope, Gemini, ...)")
    p.add_argument("--backend-arg", action="append", metavar="KEY=VALUE", help="extra backend kwargs (JSON values)")
    p.add_argument("--cache", default=None, help="SQLite response cache path")


def _add_method_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", default="vot", choices=["vot", "cot", "direct"])
    p.add_argument("--config", default=None, help="YAML with VoTConfig fields (see configs/vot/)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override VoTConfig fields")
    p.add_argument("--stsg-provider", default=None, help="vlm | detector[:<hf model id>] | static:<dir of <video_id>.json>")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)


def _build(args: argparse.Namespace):
    from .backends import build_backend
    from .grounding import build_provider
    from .reasoning import METHODS, VoTConfig

    kw = _kv(args.backend_arg)
    if args.base_url:
        kw["base_url"] = args.base_url
    model = build_backend(args.backend, cache=args.cache, **kw)
    cfg = VoTConfig.from_yaml(args.config) if args.config else VoTConfig()
    over = _kv(getattr(args, "set", None))
    if args.fps is not None:
        over["fps"] = args.fps
    if args.max_frames is not None:
        over["max_frames"] = args.max_frames
    if over:
        cfg = VoTConfig.from_dict({**cfg.to_dict(), **over})
    provider = build_provider(args.stsg_provider, model=model) if args.stsg_provider else None
    return METHODS[args.method](model, cfg, stsg_provider=provider), model


def cmd_run(args: argparse.Namespace) -> int:
    from .stsg import STSG

    method, model = _build(args)
    stsg = None
    if args.stsg:
        with open(args.stsg) as f:
            stsg = STSG.from_dict(json.load(f))
    import os

    video_id = args.video_id or os.path.splitext(os.path.basename(os.path.normpath(args.video)))[0]
    res = method.run(args.video, args.question, args.options or None, stsg=stsg, video_id=video_id, start=args.start, end=args.end)
    print(res.pretty())
    if args.verbose:
        for s in res.trace:
            print(f"\n===== {s.step} (round {s.round}) =====\n--- prompt ---\n{s.prompt}\n--- response ---\n{s.response}")
    if args.out:
        with open(args.out, "w") as f:
            f.write(res.to_json(indent=2))
        print(f"\nfull trace written to {args.out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .data.benchmarks import get_benchmark
    from .data.videoqa import load_samples
    from .eval.metrics import format_report
    from .eval.runner import run_benchmark

    method, model = _build(args)
    samples = load_samples(args.data)
    if args.limit:
        samples = samples[: args.limit]
    bench = get_benchmark(args.benchmark).name if args.benchmark else None
    report = run_benchmark(
        method, samples, args.out, video_root=args.video_root, benchmark=bench, workers=args.workers,
        resume=not args.no_resume, save_trace=args.save_trace,
    )
    print(format_report(report))
    print(json.dumps(report.get("run", {}).get("usage", {})))
    return 0


def cmd_stsg(args: argparse.Namespace) -> int:
    from .backends import build_backend
    from .grounding import build_provider
    from .video import load_video

    clip = load_video(args.video, fps=args.fps or 1.0, max_frames=args.max_frames or 16)
    kw = _kv(args.backend_arg)
    if args.base_url:
        kw["base_url"] = args.base_url
    model = build_backend(args.backend, cache=args.cache, **kw) if args.provider == "vlm" else None
    provider = build_provider(args.provider, model=model)
    g = provider(clip, targets=args.targets)
    print(g.to_expression())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(g.to_dict(), f, indent=2)
    if args.viz:
        from .video.visualize import render_stsg

        render_stsg(clip, g, args.viz)
        print(f"visualisation written to {args.viz}/grid.jpg")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="vot", description="Video-of-Thought reasoning (ICML 2024)")
    ap.add_argument("--log-level", default="WARNING")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="answer one question about one video")
    p.add_argument("--video", required=True)
    p.add_argument("--question", required=True)
    p.add_argument("--options", nargs="*", default=None, help="candidate answers (omit for open-ended QA)")
    p.add_argument("--stsg", default=None, help="STSG json of the sampled clip")
    p.add_argument("--video-id", default=None)
    p.add_argument("--start", type=float, default=None)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--out", default=None, help="write the full result + trace as JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="print every prompt and response")
    _add_backend_args(p)
    _add_method_args(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("eval", help="run a benchmark from a unified JSONL file")
    p.add_argument("--data", required=True, help="unified VideoQA JSONL (scripts/data/prepare_videoqa.py)")
    p.add_argument("--benchmark", default=None, help="nextqa | star | vlep | intentqa | socialiq | causalvidqa | msrvtt_qa | activitynet_qa")
    p.add_argument("--video-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--save-trace", action="store_true")
    _add_backend_args(p)
    _add_method_args(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("stsg", help="parse the STSG of a video with a provider")
    p.add_argument("--video", required=True)
    p.add_argument("--provider", default="vlm", help="vlm | detector[:<hf model id>]")
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--viz", default=None, help="directory for annotated frames")
    _add_backend_args(p)
    p.set_defaults(fn=cmd_stsg)

    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
