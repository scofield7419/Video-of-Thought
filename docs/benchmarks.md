# VideoQA Benchmarks

The paper evaluates VoT on eight VideoQA benchmarks (Sec. 5.1):

* **Fine-tuning setting:** VLEP, STAR, IntentQA, Social-IQ, Causal-VidQA, NExT-QA
* **Zero-shot setting:** MSR-VTT-QA, ActivityNet-QA (plus NExT-QA and STAR)

All benchmarks are converted into one unified JSONL format
(`vot.data.videoqa.VideoQASample`), one question per line:

```json
{"qid": "4010069381_6", "video": "0004/4010069381", "question": "why did the boy smile",
 "candidates": ["...", "...", "...", "...", "..."], "answer": 2, "category": "CW",
 "start": null, "end": null, "benchmark": "nextqa", "meta": {}}
```

| Field | Meaning |
|---|---|
| `qid` | unique question id (predictions are keyed by it) |
| `video` | video file name / relative path, resolved with `vot.data.videoqa.resolve_video(sample, video_root)` |
| `candidates` | answer options (multiple choice); `null` for open-ended QA |
| `answer` | option index (MC), list of indices (multi-answer), string (open-ended), `null` (unlabeled) |
| `category` | question type used for the per-category breakdown |
| `start`, `end` | optional temporal segment in seconds (STAR, VLEP, Social-IQ) |
| `meta` | benchmark-specific extras (e.g. Causal-VidQA answer/reason linkage) |

## Converting a benchmark

```bash
python scripts/data/prepare_videoqa.py --benchmark nextqa --raw data/raw/nextqa --split val \
    --out data/nextqa/val.jsonl --check-videos data/raw/nextqa/NExTVideo
```

`--raw` accepts the annotation file(s) or the directory that contains them;
already-converted JSONL files are passed through unchanged. Extra converter
options are given as `--opt key=value` (e.g. `--opt video_map=map_vid_vidorID.json`,
`--opt ways=2,4`, `--opt reason_with_answer=true`).

From Python:

```python
from vot.data.benchmarks import get_benchmark

spec = get_benchmark("nextqa")
samples = spec.convert("data/raw/nextqa", split="val")
report = spec.evaluate(samples, [{"qid": s.qid, "pred": 0} for s in samples])
```

## Benchmarks

Download links point to the official repositories. Videos must be obtained
following each dataset's license; they are not redistributed here.

### NExT-QA — `nextqa`
* Source: <https://github.com/doc-doc/NExT-QA>
* Raw files: `{train,val,test}.csv` (`video, frame_count, width, height, question, answer, qid, type, a0..a4`),
  `map_vid_vidorID.json` (video id → `<folder>/<vidor id>` under `NExTVideo/`). If the map file sits next to
  the csv it is used automatically.
* 5 options. Categories: `CW, CH` → Causal; `TN, TC, TP` → Temporal; `DL, DC, DO` → Descriptive.
* Metrics: **Acc@All, Acc@C, Acc@T, Acc@D**.

### STAR — `star`
* Source: <https://github.com/csbobby/STAR_Benchmark> (videos: Charades)
* Raw files: `STAR_{train,val,test}.json` with `question_id, video_id, question, answer, choices, start, end`.
  `answer` is the text of the correct choice; the index is recovered by matching it against `choices`.
* 4 options. Category = prefix of `question_id`.
* Metrics: **Int., Seq., Pre., Fea.**, their mean and overall accuracy.

### VLEP — `vlep`
* Source: <https://github.com/jayleicn/VideoLanguageFuturePred>
* Raw files: `vlep_{train,dev,test}_release.jsonl` with `example_id, vid_name, ts, events, answer`.
* 2 options (the two future events). VLEP has no question text, so the fixed question
  *"Which event is more likely to happen next in the video?"* is used; `ts` becomes `start`/`end`.
* Metric: accuracy.

### IntentQA — `intentqa`
* Source: <https://github.com/JoannaLiXY/IntentQA>
* Assumed raw files: NExT-QA-style `{train,val,test}.csv` (`video_id` or `video`, `question`, `answer`, `qid`,
  `type`, `a0..a4`).
* 5 options. Categories: `CW` → Why, `CH` → How, `TP`/`TN`/`TP&TN` → Before&After.
* Metrics: Why, How, Before&After, **Total** (the paper reports Total).

### Social-IQ — `socialiq`
* Source: Social-IQ 1.0 is distributed through the CMU Multimodal SDK
  (<https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK>). Its raw QA files are not tabular; convert them
  first into the intermediate JSONL below (one line per question):
  ```json
  {"qid": "xyz_q3", "video": "xyz", "question": "...", "correct": ["..."], "incorrect": ["..."], "start": null, "end": null}
  ```
* The converter enumerates **2-way** items (every correct × incorrect pair) and **4-way** items (every correct
  answer with 3 incorrect answers). Option order is shuffled deterministically per item.
  `--opt max_per_question=N` caps the items per question and way.
* The Social-IQ 2.0 format (`qid, q, a0..a3, answer_idx, vid_name, ts`;
  <https://github.com/abwilf/Social-IQ-2.0-Challenge>) is also accepted and yields 4-way items.
* Metrics: **2-way, 4-way** accuracy.

### Causal-VidQA — `causalvidqa`
* Source: <https://github.com/bcmi/Causal-VidQA> (videos: Kinetics-700 clips)
* Assumed raw layout: `QA/<video_id>/text.json`, `QA/<video_id>/answer.json` and `split/{train,val,test}.pkl`
  (video-id lists; `.txt`/`.json` lists are accepted too, or pass `--opt split_file=...`).
* 4 options. Descriptive and explanatory questions give one sample each. Predictive and counterfactual
  questions give two linked samples (`meta.part = answer | reason`, shared `meta.group`). By default the reason
  question does not reveal the gold answer; `--opt reason_with_answer=true` gives the oracle-answer variant.
* Metrics: **Acc@D, Acc@E, Acc@P (A, R, AR), Acc@C (A, R, AR)**, where AR requires both answer and reason to be
  correct, and Acc@All = mean(D, E, P-AR, C-AR).

### MSR-VTT-QA — `msrvtt_qa`
* Source: <https://github.com/xudejing/video-question-answering>
* Raw files: `{train,val,test}_qa.json` with `id, video_id, question, answer, category_id`; videos `video<id>.mp4`.
* Open-ended. Category = first word of the question (what/who/how/when/where/...).

### ActivityNet-QA — `activitynet_qa`
* Source: <https://github.com/MILVLG/activitynet-qa>
* Raw files: `dataset/{split}_q.json` (`video_name, question_id, question`) and `dataset/{split}_a.json`
  (`question_id, answer, type`). Both `<name>` and `v_<name>` video spellings are tried when resolving videos.
* Open-ended. Category = answer type (motion, spatial, temporal, yes_no, color, object, location, number, other).

### Open-ended scoring

`vot.eval.metrics` reports normalized exact match (**EM**) and token **F1** for open-ended benchmarks. The
zero-shot numbers of video MLLMs are usually obtained with an LLM judge following the Video-ChatGPT protocol
(<https://github.com/mbzuai-oryx/Video-ChatGPT>): the judge returns yes/no plus a 0–5 score, and accuracy is the
fraction of "yes". Plug in any chat model:

```python
from vot.eval.metrics import evaluate, make_llm_judge

judge = make_llm_judge(lambda system, user: my_llm(system, user))  # returns the judge's raw text
report = evaluate(samples, predictions, benchmark="msrvtt_qa", judge=judge)  # adds "Acc" (judge) and "Score"
```

## Reference results from the paper

Accuracy (%) of MotionEpic reported in the paper, for reference when reproducing.

**Table 1** (fine-tuning)

| Setting | VLEP | STAR Int. | STAR Seq. | STAR Pre. | STAR Fea. | IntentQA | Social-IQ 2-Way | Social-IQ 4-Way |
|---|---|---|---|---|---|---|---|---|
| MotionEpic + CoT | 68.2 | 66.8 | 69.6 | 60.6 | 57.4 | 66.1 | 71.7 | 43.0 |
| MotionEpic + VoT | **73.4** | **71.5** | **72.6** | **66.6** | **62.7** | **70.8** | **72.8** | **45.0** |

**Table 2** (Causal-VidQA, fine-tuning)

| Setting | Acc@D | Acc@E | Acc@P A | Acc@P R | Acc@P AR | Acc@C A | Acc@C R | Acc@C AR |
|---|---|---|---|---|---|---|---|---|
| MotionEpic + CoT | 78.5 | 77.2 | 70.1 | 70.8 | 52.4 | 71.2 | 69.1 | 55.0 |
| MotionEpic + VoT | **81.2** | **83.0** | **74.3** | **73.7** | **54.7** | **74.5** | **73.8** | **58.6** |

**Table 3** (NExT-QA, fine-tuning)

| Setting | Acc@All | Acc@C | Acc@T | Acc@D |
|---|---|---|---|---|
| MotionEpic + CoT | 72.2 | 73.4 | 69.1 | 80.7 |
| MotionEpic + VoT | **76.0** | **75.8** | **74.6** | **83.3** |

**Table 4** (zero-shot)

| Setting | MSR-VTT | ActivityNet | NExT-QA | STAR | Avg. |
|---|---|---|---|---|---|
| MotionEpic + CoT | 63.1 | 50.0 | 61.9 | 56.5 | 57.8 |
| MotionEpic + VoT | **66.2** | **54.6** | **66.5** | **61.7** | **62.3** |
| &nbsp;&nbsp;w/o Verify-G | 63.6 | 51.4 | 62.0 | 59.1 | 59.0 |
| &nbsp;&nbsp;w/o Verify-C | 65.1 | 53.4 | 62.8 | 58.8 | 60.1 |
