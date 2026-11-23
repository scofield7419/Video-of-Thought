"""Benchmark runner: run a reasoning method over a unified VideoQA file,
stream predictions to JSONL (resumable) and compute the benchmark report."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence

from ..data.videoqa import VideoQASample, resolve_video
from .metrics import evaluate, format_report

logger = logging.getLogger(__name__)


def _load_done(path: str) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:  # e.g. a line cut by an interrupted run
                    continue
                if not rec.get("error"):
                    done[str(rec["qid"])] = rec
    return done


def run_benchmark(
    method: Any,
    samples: Sequence[VideoQASample],
    out_path: str,
    video_root: Optional[str] = None,
    benchmark: Optional[str] = None,
    workers: int = 1,
    resume: bool = True,
    save_trace: bool = False,
    progress: bool = True,
) -> Dict[str, Any]:
    """Run ``method`` (a :class:`~vot.reasoning.VideoOfThought` or baseline) on ``samples``.

    Each finished instance is appended to ``out_path`` as
    ``{"qid", "pred", "pred_text", "verified", "usage", ...}``; with ``resume``
    already-finished qids are skipped. Returns the metrics report, which is also
    written next to ``out_path`` (``.report.json`` / ``.report.md``).
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    done = _load_done(out_path) if resume else {}
    if not resume and os.path.exists(out_path):
        os.remove(out_path)
    todo = [s for s in samples if str(s.qid) not in done]
    lock = threading.Lock()
    t0 = time.time()
    n_err = 0

    def work(s: VideoQASample) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"qid": s.qid}
        try:
            video = resolve_video(s, video_root)
            if video is None:
                raise FileNotFoundError(f"video {s.video!r} not found under {video_root!r}")
            res = method.run(
                video, s.question, s.candidates if s.is_multi_choice else None,
                video_id=os.path.splitext(os.path.basename(s.video))[0], start=s.start, end=s.end,
            )
            rec.update(
                pred=res.prediction,
                pred_text=res.answer_text,
                candidates=res.candidates if not s.is_multi_choice else None,
                scores=res.scores,
                targets=res.targets,
                verified=res.verified,
                rounds=res.rounds,
                usage=res.usage,
            )
            if save_trace:
                rec["result"] = res.to_dict(with_trace=True)
        except Exception as e:  # keep the run alive; failures are retried on resume
            rec.update(pred=None, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(limit=3))
        return rec

    iterator: Any
    with open(out_path, "a") as fout, ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(work, s) for s in todo]
        iterator = as_completed(futures)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(futures), desc=type(method).__name__)
            except ImportError:
                pass
        for fut in iterator:
            rec = fut.result()
            if rec.get("error"):
                n_err += 1
                logger.warning("qid=%s failed: %s", rec["qid"], rec["error"])
            with lock:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            if not rec.get("error"):
                done[str(rec["qid"])] = rec

    preds = [{"qid": q, "pred": r.get("pred")} for q, r in done.items()]
    report = evaluate(list(samples), preds, benchmark=benchmark)
    report["run"] = {
        "method": type(method).__name__,
        "backend": method.model.describe() if hasattr(method, "model") else None,
        "config": method.cfg.to_dict() if hasattr(method, "cfg") else None,
        "errors_this_run": n_err,
        "seconds_this_run": round(time.time() - t0, 1),
        "usage": _sum_usage(done.values()),
    }
    base = os.path.splitext(out_path)[0]
    with open(base + ".report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    with open(base + ".report.md", "w") as f:
        f.write(format_report(report))
    return report


def _sum_usage(records: Any) -> Dict[str, int]:
    tot: Dict[str, int] = {}
    for r in records:
        for k, v in (r.get("usage") or {}).items():
            tot[k] = tot.get(k, 0) + int(v or 0)
    return tot


def load_predictions(path: str) -> List[Dict[str, Any]]:
    return list(_load_done(path).values())
