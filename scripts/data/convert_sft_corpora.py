"""Convert stage-1 / stage-3 corpora into vot.data.sft records.

* ``webvid``: WebVid CSV (``videoid, name, page_dir, ...``) -> caption records
  ``<video> + "Describe the video briefly." -> caption``; video path ``{page_dir}/{videoid}.mp4``.
* ``videochatgpt``: Video-ChatGPT instruction JSON (list of ``{"video_id", "q", "a"}``).
* ``videochat``: VideoChat instruction JSON (list of ``{"video", "QA": [{"i", "q", "a"}, ...]}``).

    python scripts/data/convert_sft_corpora.py webvid --src results_10M_train.csv --out data/sft/webvid_caption.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random

CAPTION_PROMPTS = [
    "Describe the video briefly.",
    "Give a short caption of the video.",
    "What is happening in the video? Answer in one sentence.",
    "Summarize the visual content of the video.",
]


def webvid(src: str, max_rows: int, seed: int):
    rng = random.Random(seed)
    with open(src) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if max_rows and i >= max_rows:
                break
            yield {
                "id": f"webvid-{row['videoid']}",
                "video": os.path.join(row.get("page_dir", ""), f"{row['videoid']}.mp4"),
                "conversations": [
                    {"from": "human", "value": "<video>\n" + rng.choice(CAPTION_PROMPTS)},
                    {"from": "gpt", "value": row["name"].strip()},
                ],
            }


def videochatgpt(src: str, max_rows: int, seed: int, ext: str = ".mp4"):
    with open(src) as f:
        data = json.load(f)
    for i, r in enumerate(data):
        if max_rows and i >= max_rows:
            break
        vid = r["video_id"]
        yield {
            "id": f"vcg-{i}",
            "video": vid if os.path.splitext(vid)[1] else vid + ext,
            "conversations": [{"from": "human", "value": "<video>\n" + r["q"].strip()}, {"from": "gpt", "value": r["a"].strip()}],
        }


def videochat(src: str, max_rows: int, seed: int):
    with open(src) as f:
        data = json.load(f)
    for i, r in enumerate(data):
        if max_rows and i >= max_rows:
            break
        conv = []
        for j, qa in enumerate(r["QA"]):
            q = " ".join(x for x in (qa.get("i", ""), qa.get("q", "")) if x).strip()
            conv.append({"from": "human", "value": ("<video>\n" if j == 0 else "") + q})
            conv.append({"from": "gpt", "value": qa["a"].strip()})
        yield {"id": f"videochat-{i}", "video": r["video"], "conversations": conv}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["webvid", "videochatgpt", "videochat"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    fn = {"webvid": webvid, "videochatgpt": videochatgpt, "videochat": videochat}[args.kind]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for rec in fn(args.src, args.max_rows, args.seed):
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
