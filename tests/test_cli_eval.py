import json

from PIL import Image

from vot.cli import main


def test_eval_end_to_end_with_mock(tmp_path):
    vids = tmp_path / "videos"
    for v in ["v1", "v2"]:
        d = vids / v
        d.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (32, 24), (i * 30, 0, 0)).save(d / f"{i:03d}.jpg")
    data = tmp_path / "val.jsonl"
    rows = [
        {"qid": "q1", "video": "v1", "question": "Why?", "candidates": ["a", "b", "c"], "answer": 0, "category": "CW", "benchmark": "nextqa"},
        {"qid": "q2", "video": "v2", "question": "What next?", "candidates": ["a", "b", "c"], "answer": 1, "category": "TN", "benchmark": "nextqa"},
        {"qid": "q3", "video": "missing", "question": "?", "candidates": ["a", "b"], "answer": 0, "category": "DL", "benchmark": "nextqa"},
    ]
    data.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "res" / "pred.jsonl"
    for method in ["vot", "cot"]:
        rc = main(["eval", "--data", str(data), "--benchmark", "nextqa", "--video-root", str(vids), "--out", str(out),
                   "--backend", "mock", "--method", method, "--workers", "2", "--no-resume", "--set", "max_frames=2"])
        assert rc == 0
        recs = [json.loads(l) for l in out.read_text().splitlines()]
        assert {r["qid"] for r in recs} == {"q1", "q2", "q3"}
        assert [r for r in recs if r["qid"] == "q3"][0]["error"].startswith("FileNotFoundError")
        report = json.loads((tmp_path / "res" / "pred.report.json").read_text())
        assert report["n"] == 3 and "Acc@All" in report["metrics"]
    assert (tmp_path / "res" / "pred.report.md").exists()


def test_run_cli(tmp_path, capsys, truck_stsg):
    d = tmp_path / "v"
    d.mkdir()
    for i in range(3):
        Image.new("RGB", (32, 24)).save(d / f"{i:03d}.jpg")
    g = tmp_path / "g.json"
    g.write_text(json.dumps(truck_stsg.to_dict()))
    out = tmp_path / "trace.json"
    rc = main(["run", "--video", str(d), "--question", "What is the truck doing?", "--options", "parking", "collecting garbage",
               "--backend", "mock", "--stsg", str(g), "--out", str(out), "--max-frames", "3"])
    assert rc == 0 and "Answer" in capsys.readouterr().out
    assert json.loads(out.read_text())["trace"]


def test_runner_skips_truncated_line(tmp_path):
    from vot.eval.runner import load_predictions

    p = tmp_path / "p.jsonl"
    p.write_text(json.dumps({"qid": "a", "pred": 1}) + "\n" + '{"qid": "b", "pr')
    assert [r["qid"] for r in load_predictions(str(p))] == ["a"]
