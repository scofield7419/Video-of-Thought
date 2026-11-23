"""Build the L1-L5 grounding-aware tuning corpus (paper Sec. 3.3) from STSG JSONL files.

Input: JSONL rows with ``video_id``, ``stsg`` and frame alignment (``frames`` and/or
``frame_times``; optional ``video``), e.g. the output of convert_action_genome.py or
of an external dynamic scene-graph parser run on WebVid.

Optional Charades action annotations (for Action Genome videos) give L3 real action
descriptions with time spans; otherwise actions are derived from relation spans.

Rows with ``"split": "test"`` are skipped unless ``--include_test`` is given, so benchmark
test videos never leak into grounding tuning. Samples whose text exceeds ``--max_tokens``
are shortened (L2: fewer frames) or dropped.

    python scripts/data/build_grounding_corpus.py --stsg data/action_genome/stsg_train.jsonl \\
        --out data/sft/grounding_ag.jsonl --charades_csv Charades_v1_train.csv \\
        --charades_classes Charades_v1_classes.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vot.data.grounding_tasks import TASKS, ActionAnnotation, GroundingSource, GroundingTaskBuilder  # noqa: E402
from vot.stsg.schema import STSG  # noqa: E402


def load_charades(csv_path: str, classes_path: str) -> Dict[str, List[ActionAnnotation]]:
    names = {}
    with open(classes_path) as f:
        for line in f:
            if line.strip():
                cid, name = line.strip().split(" ", 1)
                names[cid] = name
    out: Dict[str, List[ActionAnnotation]] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            acts = []
            for item in (row.get("actions") or "").split(";"):
                parts = item.split()
                if len(parts) == 3 and parts[0] in names:
                    acts.append(ActionAnnotation(names[parts[0]], float(parts[1]), float(parts[2])))
            out[row["id"]] = acts
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stsg", nargs="+", required=True, help="STSG JSONL file(s)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--max_frames", type=int, default=16, help="frames per sample (L1, L3-L5)")
    ap.add_argument("--l2_max_frames", type=int, default=8, help="frames per full-STSG (L2) sample")
    ap.add_argument("--max_tokens", type=int, default=3800,
                    help="text budget per sample (model_max_length minus video/STSG tokens and template); 0 = off")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer for exact counts (default: conservative estimate)")
    ap.add_argument("--include_test", action="store_true", help="keep rows with split == 'test'")
    ap.add_argument("--repeats", type=int, default=1, help="samples per task and video (different seeds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--charades_csv", default=None)
    ap.add_argument("--charades_classes", default=None)
    ap.add_argument("--pool_size", type=int, default=2000, help="videos used as L1 negative pool")
    args = ap.parse_args()

    actions = load_charades(args.charades_csv, args.charades_classes) if args.charades_csv else {}
    counter = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=False)
        counter = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    sources: List[GroundingSource] = []
    skipped_test = 0
    for path in args.stsg:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("split") == "test" and not args.include_test:
                    skipped_test += 1
                    continue
                vid = r.get("video_id") or r.get("id")
                acts = actions.get(vid, []) if r.get("frame_times") else []
                sources.append(GroundingSource(
                    id=vid, stsg=STSG.from_dict(r["stsg"]), video=r.get("video"),
                    frames=r.get("frames"), frame_times=r.get("frame_times"), actions=acts,
                ))
    rng = random.Random(args.seed)
    pool = rng.sample(sources, min(args.pool_size, len(sources)))
    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for rep in range(args.repeats):
            builder = GroundingTaskBuilder(
                tasks=args.tasks, seed=args.seed + rep, max_frames=args.max_frames,
                task_max_frames={"L2": args.l2_max_frames}, max_tokens=args.max_tokens or None, token_counter=counter,
            )
            for src in sources:
                for rec in builder.build(src, pool):
                    if args.repeats > 1:
                        rec["id"] = f"{rec['id']}-r{rep}"
                    f.write(json.dumps(rec) + "\n")
                    n += 1
            for task, k in sorted(builder.dropped.items()):
                print(f"[repeat {rep}] dropped {k} {task} samples over the token budget")
    print(f"wrote {n} grounding samples ({', '.join(args.tasks)}) from {len(sources)} videos to {args.out}"
          + (f" (skipped {skipped_test} test-split videos)" if skipped_test else ""))


if __name__ == "__main__":
    main()
