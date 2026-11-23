import importlib.util
import json
import os
import pickle

from _tiny_motionepic import kitchen_stsg

from vot.data.grounding_tasks import (
    ActionAnnotation,
    GroundingSource,
    GroundingTaskBuilder,
    perturb_stsg,
    relation_spans,
    renumber,
)
from vot.stsg import STSG, parse_stsg

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _src(sid="vid1", times=True):
    g = kitchen_stsg()
    return GroundingSource(
        id=sid, stsg=g, video=f"{sid}.mp4",
        frames=[f"{sid}.mp4/{i:06d}.png" for i in range(3)],
        frame_times=[1.0, 2.0, 3.0] if times else None,
    )


def test_relation_spans_merge_consecutive_frames():
    spans = relation_spans(kitchen_stsg())
    by_pred = {s.predicate: s for s in spans}
    assert by_pred["holding"].frames == [2, 3]
    assert by_pred["looking at"].frames == [1]
    assert by_pred["holding"].describe(with_time=False) == "the person is holding the cup"


def test_renumber_subsamples_and_aligns_frames():
    src = _src()
    src.stsg = STSG.from_dict({**src.stsg.to_dict(), "frames": [dict(f, index=f["index"] * 10) for f in src.stsg.to_dict()["frames"]], "coref": []})
    src.stsg.infer_coref()
    g, pos = renumber(src, max_frames=2)
    assert [f.index for f in g.frames] == [1, 2]
    assert pos == [0, 2]
    assert [f.timestamp for f in g.frames] == [1.0, 3.0]
    assert all(c.dst_frame == c.src_frame + 1 for c in g.coref)


def test_builder_produces_all_tasks_deterministically():
    pool = [_src("vid1"), _src("vid2")]
    b = GroundingTaskBuilder(seed=3)
    recs = b.build(pool[0], pool)
    assert [r["task"] for r in recs] == ["L1", "L2", "L3", "L4", "L5"]
    assert recs == GroundingTaskBuilder(seed=3).build(pool[0], pool)
    for r in recs:
        human, gpt = r["conversations"][0]["value"], r["conversations"][1]["value"]
        assert human.startswith("<video>\n")
        assert r["frames"] == pool[0].frames and r["frame_times"] == [1.0, 2.0, 3.0]
        if r["task"] == "L1":
            assert "<stsg>" in human and gpt in {"Yes.", "No."} and "stsg" in r
        if r["task"] == "L2":
            ref = [{k: v for k, v in f.items() if k != "timestamp"} for f in renumber(pool[0])[0].to_dict()["frames"]]
            assert parse_stsg(gpt).to_dict()["frames"] == ref
        if r["task"] in {"L3", "L4", "L5"}:
            assert not parse_stsg(gpt).is_empty
    l5 = recs[-1]
    assert "Frame" in l5["conversations"][0]["value"] and "The object is [" in l5["conversations"][1]["value"]


def test_l1_negatives_cover_other_and_perturbed():
    pool = [_src(f"v{i}") for i in range(4)]
    kinds = set()
    for seed in range(40):
        rec = GroundingTaskBuilder(tasks=["L1"], seed=seed, l1_negative_ratio=1.0).build(pool[0], pool)[0]
        assert rec["conversations"][1]["value"] == "No."
        kinds.add(rec["negative"].split(":")[0])
        if rec["negative"].startswith("other"):
            assert rec["stsg_frames"][0].startswith(rec["negative"].split(":")[1])
    assert kinds == {"other", "perturb"}


def test_perturbations_change_graph():
    import random

    g = kitchen_stsg()
    seen = set()
    for s in range(30):
        h, kind = perturb_stsg(g, random.Random(s), ["dog", "sofa"], ["sitting on"])
        seen.add(kind)
        h.validate()
        assert h.to_dict() != g.to_dict()
    assert seen == {"category", "predicate", "time"}


def test_external_actions_drive_l3():
    src = _src()
    src.actions = [ActionAnnotation("Holding a cup", 1.5, 3.5)]
    rec = GroundingTaskBuilder(tasks=["L3"], seed=0, max_actions=1).build(src)[0]
    assert "[holding a cup]" in rec["conversations"][0]["value"]
    g = parse_stsg(rec["conversations"][1]["value"])
    assert [f.index for f in g.frames] == [2, 3]
    assert set(g.categories()) == {"person", "cup"}


def test_action_genome_conversion(tmp_path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    person = {
        "AB12.mp4/000010.png": {"bbox": [[10, 20, 110, 220]], "bbox_size": (200, 400)},
        "AB12.mp4/000020.png": {"bbox": [], "bbox_size": (200, 400)},
    }
    objects = {
        "AB12.mp4/000010.png": [
            {"class": "cup/glass/bottle", "bbox": (50, 100, 20, 40), "visible": True,
             "attention_relationship": ["looking_at"], "spatial_relationship": ["in_front_of"],
             "contacting_relationship": ["holding"], "metadata": {"set": "train"}},
            {"class": "table", "bbox": None, "visible": False, "metadata": {"set": "train"}},
        ],
        "AB12.mp4/000020.png": [
            {"class": "cup/glass/bottle", "bbox": (60, 100, 20, 40), "visible": True,
             "attention_relationship": ["not_looking_at"], "spatial_relationship": [], "contacting_relationship": ["holding"],
             "metadata": {"set": "train"}},
        ],
    }
    for name, obj in (("person_bbox.pkl", person), ("object_bbox_and_relationship.pkl", objects)):
        with open(ann / name, "wb") as f:
            pickle.dump(obj, f)
    spec = importlib.util.spec_from_file_location("conv_ag", os.path.join(ROOT, "scripts/data/convert_action_genome.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = mod.convert(person, objects)
    assert len(rows) == 1
    r = rows[0]
    json.dumps(r)
    g = STSG.from_dict(r["stsg"])
    g.validate()
    f1, f2 = g.frames
    assert f1.get("person").bbox == (5.0, 5.0, 50.0, 50.0)
    assert f1.get("cup").bbox == (25.0, 25.0, 10.0, 10.0) and f1.get("cup").label == "cup or glass or bottle"
    assert {t.predicate for t in f1.triplets} == {"looking at", "in front of", "holding"}
    assert f2.get("person").bbox is None
    assert ("person", "person") in {(c.src, c.dst) for c in g.coref}
    assert r["frames"] == ["AB12.mp4/000010.png", "AB12.mp4/000020.png"] and r["split"] == "train"
