import csv
import json
import os
import pickle
import subprocess
import sys

import pytest

from vot.data.benchmarks import BENCHMARKS, get_benchmark
from vot.data.videoqa import VideoQASample, load_samples, resolve_video, save_samples

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def _write_jsonl(path, recs):
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_sample_roundtrip(tmp_path):
    s = VideoQASample("q1", "v1", "What?", ["x", "y"], 1, "CW", 1.5, 3.0, "nextqa", {"k": 1})
    assert s.is_multi_choice and s.answer_text == "y" and s.answer_letter == "B"
    assert VideoQASample.from_dict(s.to_dict()) == s
    oe = VideoQASample("q2", "v2", "Who?", answer="man")
    assert not oe.is_multi_choice and oe.answer_text == "man"
    multi = VideoQASample("q3", "v3", "Which?", ["a", "b", "c"], [0, 2])
    assert multi.is_multi_answer and multi.answer_text == ["a", "c"]
    p = tmp_path / "s.jsonl"
    assert save_samples([s, oe, multi], str(p)) == 3
    assert load_samples(str(p)) == [s, oe, multi]


def test_resolve_video(tmp_path):
    root = tmp_path / "videos"
    (root / "0004").mkdir(parents=True)
    (root / "0004" / "123.mp4").write_bytes(b"")
    (root / "v_abc.mkv").write_bytes(b"")
    (root / "frames_dir").mkdir()
    assert resolve_video("0004/123", str(root)) == str(root / "0004" / "123.mp4")
    assert resolve_video("123", str(root)) == str(root / "0004" / "123.mp4")  # recursive search
    assert resolve_video(str(root / "0004" / "123.mp4")) == str(root / "0004" / "123.mp4")
    assert resolve_video("frames_dir", str(root)) == str(root / "frames_dir")
    s = VideoQASample("q", "abc", "?", meta={"video_alts": ["v_abc"]})
    assert resolve_video(s, str(root)) == str(root / "v_abc.mkv")
    assert resolve_video("nope", str(root)) is None


def test_registry_complete():
    assert set(BENCHMARKS) == {
        "nextqa", "star", "vlep", "intentqa", "socialiq", "causalvidqa", "msrvtt_qa", "activitynet_qa"
    }
    assert get_benchmark("NExT-QA").name == "nextqa"
    assert get_benchmark("msrvtt").task == "oe"
    with pytest.raises(KeyError):
        get_benchmark("unknown")


def test_nextqa(tmp_path):
    rows = [
        dict(video="4010069381", frame_count=100, width=640, height=480, question="why did the boy smile",
             answer=2, qid=6, type="CW", a0="x", a1="y", a2="z", a3="u", a4="w"),
        dict(video="4010069381", frame_count=100, width=640, height=480, question="what is in his hand",
             answer=0, qid=7, type="DO", a0="cup", a1="ball", a2="pen", a3="phone", a4="book"),
    ]
    _write_csv(tmp_path / "val.csv", rows)
    _write_json(str(tmp_path / "map_vid_vidorID.json"), {"4010069381": "0004/4010069381"})
    samples = get_benchmark("nextqa").convert(str(tmp_path), split="val")
    assert [s.qid for s in samples] == ["4010069381_6", "4010069381_7"]
    assert samples[0].video == "0004/4010069381" and samples[0].answer == 2 and samples[0].category == "CW"
    assert samples[1].candidates[0] == "cup" and len(samples[1].candidates) == 5
    # unified passthrough
    save_samples(samples, str(tmp_path / "u.jsonl"))
    assert get_benchmark("nextqa").convert(str(tmp_path / "u.jsonl")) == samples


def test_intentqa(tmp_path):
    rows = [dict(video_id="9", question="why", answer=1, qid=0, type="TP&TN", a0="a", a1="b", a2="c", a3="d", a4="e")]
    _write_csv(tmp_path / "test.csv", rows)
    (s,) = get_benchmark("intentqa").convert(str(tmp_path / "test.csv"), split="test")
    assert s.video == "9" and s.answer == 1 and s.category == "TP&TN" and s.benchmark == "intentqa"


def test_star(tmp_path):
    recs = [
        {"question_id": "Interaction_T1_13", "video_id": "6H78U", "question": "What did the person do?",
         "answer": "Took the cup.", "start": 11.1, "end": 19.6,
         "choices": [{"choice_id": 1, "choice": "Took the cup."}, {"choice_id": 0, "choice": "Ate the food."},
                     {"choice_id": 2, "choice": "Opened the door."}, {"choice_id": 3, "choice": "Sat on the bed."}]},
        {"question_id": "Feasibility_T4_1", "video_id": "AB12", "question": "What can he do next?",
         "start": 0, "end": 5, "choices": [{"choice_id": i, "choice": str(i)} for i in range(4)]},
    ]
    _write_json(str(tmp_path / "STAR_val.json"), recs)
    a, b = get_benchmark("star").convert(str(tmp_path), split="val")
    assert a.candidates[0] == "Ate the food." and a.answer == 1  # sorted by choice_id
    assert a.category == "Interaction" and a.start == 11.1 and a.end == 19.6
    assert b.answer is None and b.category == "Feasibility"


def test_vlep(tmp_path):
    _write_jsonl(tmp_path / "vlep_dev_release.jsonl",
                 [{"example_id": 7, "vid_name": "friends_s01", "ts": [3.0, 8.5], "events": ["e0", "e1"], "answer": 1}])
    (s,) = get_benchmark("vlep").convert(str(tmp_path), split="dev")
    assert s.qid == "7" and s.candidates == ["e0", "e1"] and s.answer == 1 and (s.start, s.end) == (3.0, 8.5)


def test_causalvidqa(tmp_path):
    qa = tmp_path / "QA" / "vid1"
    text = {
        "descriptive": {"question": "What is he doing?", "answer": ["a", "b", "c", "d"]},
        "explanatory": {"question": "Why?", "answer": ["a", "b", "c", "d"]},
        "predictive": {"question": "What next?", "answer": ["a", "b", "c", "d"], "reason": ["r0", "r1", "r2", "r3"]},
        "counterfactual": {"question": "What if?", "answer": ["a", "b", "c", "d"], "reason": ["r0", "r1", "r2", "r3"]},
    }
    ans = {"descriptive": {"answer": 1}, "explanatory": {"answer": 0},
           "predictive": {"answer": 2, "reason": 3}, "counterfactual": {"answer": 0, "reason": 1}}
    _write_json(str(qa / "text.json"), text)
    _write_json(str(qa / "answer.json"), ans)
    os.makedirs(tmp_path / "split")
    with open(tmp_path / "split" / "test.pkl", "wb") as f:
        pickle.dump(["vid1"], f)
    samples = get_benchmark("causalvidqa").convert(str(tmp_path), split="test")
    assert len(samples) == 6
    by = {s.qid: s for s in samples}
    assert by["vid1_descriptive"].answer == 1 and by["vid1_descriptive"].category == "D"
    assert by["vid1_predictive_answer"].answer == 2 and by["vid1_predictive_reason"].answer == 3
    assert by["vid1_predictive_reason"].candidates == ["r0", "r1", "r2", "r3"]
    assert by["vid1_counterfactual_reason"].meta["group"] == "vid1_counterfactual"
    assert "a" not in by["vid1_predictive_reason"].question.split()  # gold answer not leaked
    oracle = get_benchmark("causalvidqa").convert(str(tmp_path), split="test", reason_with_answer=True)
    assert "The answer is: c" in {s.qid: s for s in oracle}["vid1_predictive_reason"].question


def test_socialiq(tmp_path):
    rec = {"qid": "q1", "video": "vidA", "question": "How does she feel?",
           "correct": ["happy", "glad", "joyful", "pleased"], "incorrect": ["sad", "angry", "bored"]}
    _write_jsonl(tmp_path / "val.jsonl", [rec])
    conv = get_benchmark("socialiq").convert
    samples = conv(str(tmp_path / "val.jsonl"), split="val")
    two = [s for s in samples if s.category == "2-way"]
    four = [s for s in samples if s.category == "4-way"]
    assert len(two) == 4 * 3 and len(four) == 4 * 1
    for s in samples:
        assert s.candidates[s.answer] in rec["correct"]
        assert sum(c in rec["correct"] for c in s.candidates) == 1
    assert [s.to_dict() for s in conv(str(tmp_path / "val.jsonl"), split="val")] == [s.to_dict() for s in samples]
    assert len({tuple(s.candidates).index(s.candidates[s.answer]) for s in two}) > 1  # answer position varies
    capped = conv(str(tmp_path / "val.jsonl"), split="val", ways=[2], max_per_question=5)
    assert len(capped) == 5 and all(s.category == "2-way" for s in capped)
    # Social-IQ 2.0 style
    _write_jsonl(tmp_path / "siq2.jsonl",
                 [{"qid": "z", "q": "Why?", "a0": "x", "a1": "y", "a2": "z", "a3": "w", "answer_idx": 2,
                   "vid_name": "abc", "ts": [1, 2]}])
    (s2,) = conv(str(tmp_path / "siq2.jsonl"))
    assert s2.answer == 2 and s2.category == "4-way" and s2.video == "abc"


def test_msrvtt_qa(tmp_path):
    _write_json(str(tmp_path / "test_qa.json"),
                [{"id": 5, "video_id": 7010, "question": "what is a man doing?", "answer": "cooking", "category_id": 2}])
    (s,) = get_benchmark("msrvtt_qa").convert(str(tmp_path), split="test")
    assert s.video == "video7010" and s.answer == "cooking" and s.category == "what" and not s.is_multi_choice


def test_activitynet_qa(tmp_path):
    d = tmp_path / "dataset"
    _write_json(str(d / "test_q.json"), [{"video_name": "abc123", "question_id": "abc123_1", "question": "is it day"}])
    _write_json(str(d / "test_a.json"), [{"question_id": "abc123_1", "answer": "yes", "type": 3}])
    (s,) = get_benchmark("activitynet_qa").convert(str(tmp_path), split="test")
    assert s.answer == "yes" and s.category == "yes_no" and s.meta["video_alts"] == ["v_abc123"]
    (s2,) = get_benchmark("activitynet_qa").convert([str(d / "test_a.json"), str(d / "test_q.json")], split="test")
    assert s2 == s


def test_prepare_script(tmp_path):
    _write_jsonl(tmp_path / "vlep_dev_release.jsonl",
                 [{"example_id": 1, "vid_name": "v", "ts": [0, 1], "events": ["a", "b"], "answer": 0}])
    out = tmp_path / "out" / "dev.jsonl"
    res = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "data", "prepare_videoqa.py"),
         "--benchmark", "vlep", "--raw", str(tmp_path), "--out", str(out)],
        capture_output=True, text=True, env={**os.environ, "KMP_DUPLICATE_LIB_OK": "TRUE"},
    )
    assert res.returncode == 0, res.stderr
    assert len(load_samples(str(out))) == 1
