"""Regression tests for the MotionEpic review findings (items 1-9)."""

import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from _tiny_motionepic import kitchen_stsg, tiny_clip, tiny_model  # noqa: E402
from PIL import Image  # noqa: E402

from vot.data.grounding_tasks import GroundingSource, GroundingTaskBuilder, perturb_stsg  # noqa: E402
from vot.data.sft import SFTCollator, SFTDataset  # noqa: E402
from vot.models.motionepic import IGNORE_INDEX, vicuna_v1  # noqa: E402
from vot.stsg import STSG  # noqa: E402
from vot.video import VideoClip  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _conv(user, reply=None):
    c = vicuna_v1()
    c.append("USER", user)
    c.append("ASSISTANT", reply)
    return c


def _long_stsg(n_frames=16, n_obj=6):
    frames = []
    for k in range(n_frames):
        objs = {f"obj-{i}": [10.0 * i + 0.1 * k, 5.0 + k, 7.3, 8.9] for i in range(n_obj)}
        trips = [(f"obj-{i}", "next to", f"obj-{i + 1}") for i in range(n_obj - 1)]
        frames.append({"objects": objs, "triplets": trips, "timestamp": 0.5 * k})
    return STSG.from_frames(frames)


# 1 -------------------------------------------------------------------------
def test_video_frames_are_square_padded_not_center_cropped():
    _, proc = tiny_model()
    img = Image.new("RGB", (480, 270), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 100, 270))  # object in x in [0, 20.8%]
    clip = VideoClip([img], [0.0])
    px = proc.process_video(clip)[0]
    assert px.shape[-1] == px.shape[-2]
    left = px[0, :, : px.shape[-1] // 6]  # left sixth of the padded square
    assert float(left.max()) > 1.5  # the white stripe is still visible


# 2 -------------------------------------------------------------------------
def test_zero3_is_rejected(tmp_path):
    from vot.train.train import check_supported, zero_stage

    z3, z2 = tmp_path / "z3.json", tmp_path / "z2.json"
    z3.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    z2.write_text(json.dumps({"zero_optimization": {"stage": 2}}))
    assert zero_stage(SimpleNamespace(deepspeed=str(z2))) == 2
    assert zero_stage(SimpleNamespace(deepspeed=None)) == 0
    check_supported(SimpleNamespace(deepspeed=str(z2)))
    with pytest.raises(ValueError, match="ZeRO-3"):
        check_supported(SimpleNamespace(deepspeed=str(z3)))
    with pytest.raises(ValueError, match="ZeRO-3"):
        check_supported(SimpleNamespace(deepspeed={"zero_optimization": {"stage": 3}}))
    assert not os.path.exists(os.path.join(ROOT, "configs", "deepspeed", "zero3.json"))


# 3 -------------------------------------------------------------------------
def test_time_reversal_negative_keeps_slot_timestamps():
    g = kitchen_stsg()
    for f, t in zip(g.frames, [4.1, 4.6, 5.1]):
        f.timestamp = t
    for s in range(50):
        h, kind = perturb_stsg(g, random.Random(s))
        if kind == "time":
            break
    assert kind == "time"
    ordered = sorted(h.frames, key=lambda f: f.index)
    assert [f.timestamp for f in ordered] == [4.1, 4.6, 5.1]
    assert ordered[0].object_ids() == sorted(g.frames, key=lambda f: f.index)[-1].object_ids()


# 4 -------------------------------------------------------------------------
def test_inference_truncation_keeps_prompt_tail():
    model, proc = tiny_model()
    model.eval()
    inp = proc.build_inputs(_conv("<video>\n" + " ".join(["describe the cup"] * 20)), [tiny_clip(4)], [])
    model.model_max_length = 40
    emb, am, pos, _ = model.prepare_multimodal_inputs(inp["input_ids"], inp["attention_mask"], None, inp["videos"], inp["video_mask"], None, padding_side="left")
    assert emb.shape[1] == 40
    tail = model.get_input_embeddings()(inp["input_ids"][0, -3:])
    assert torch.allclose(emb[0, -3:], tail)
    ids, labels = proc.encode_conversation(_conv("<video>\n" + " ".join(["describe the cup"] * 20), "yes ."))
    vids, vmask = proc.collate_videos([proc.process_video(tiny_clip(4))])
    emb2, _, _, lab2 = model.prepare_multimodal_inputs(ids[None], None, labels[None], vids, vmask, None)
    head = model.get_input_embeddings()(ids[:2])
    assert emb2.shape[1] == 40 and torch.allclose(emb2[0, :2], head)  # training keeps the head


# 5 -------------------------------------------------------------------------
def test_l2_uses_fewer_frames_and_length_budget():
    src = GroundingSource(id="v", stsg=_long_stsg(), frame_times=[0.5 * k for k in range(16)])
    recs = {r["task"]: r for r in GroundingTaskBuilder(seed=0).build(src)}
    assert len(recs["L2"]["frame_times"]) == 8 and len(recs["L5"]["frame_times"]) == 16
    b = GroundingTaskBuilder(seed=0, max_tokens=1700)  # 8-frame L2 ~3.3k tokens, 4-frame ~1.6k
    recs = {r["task"]: r for r in b.build(src)}
    assert len(recs["L2"]["frame_times"]) == 4
    assert set(recs) == {"L1", "L2", "L5"} and b.dropped == {"L3": 1, "L4": 1}
    assert all(b._fits(r) for r in recs.values())
    tiny = GroundingTaskBuilder(seed=0, max_tokens=5)
    assert tiny.build(src) == [] and sum(tiny.dropped.values()) == 5


def test_collator_masks_cut_reply_and_budgets_multimodal_tokens():
    _, proc = tiny_model()
    cfg = proc.cfg
    ids, labels = proc.encode_conversation(_conv("<video>\ndescribe the cup", " ".join(["the person is holding the cup ."] * 5)))
    item = {"input_ids": ids, "labels": labels, "videos": [proc.process_video(tiny_clip(2))], "stsgs": []}
    budget = len(ids) + cfg.num_query_tokens - 1
    full = SFTCollator(proc, max_length=budget)([item])
    assert torch.equal(full["labels"][0], labels)  # fits exactly: untouched
    cut = SFTCollator(proc, max_length=budget - 3)
    t_ids, t_lab = cut.truncate(ids, labels)
    assert len(t_ids) + cfg.num_query_tokens - 1 == budget - 3
    assert (t_lab == IGNORE_INDEX).all()  # the cut reply is fully masked (no EOS-less target)
    ok_ids, ok_lab = proc.encode_conversation(_conv("<video>\ndescribe the cup", "yes ."))
    ok = {"input_ids": ok_ids, "labels": ok_lab, "videos": [proc.process_video(tiny_clip(2))], "stsgs": []}
    batch = cut([item, ok])
    assert batch["input_ids"].shape[0] == 1 and batch["videos"].shape[0] == 1  # unsupervised sample dropped


def test_collator_masks_only_the_cut_turn():
    _, proc = tiny_model()
    c = vicuna_v1()
    c.append("USER", "describe the cup")
    c.append("ASSISTANT", "yes .")
    c.append("USER", "is the cup on table")
    c.append("ASSISTANT", " ".join(["no ."] * 10))
    ids, labels = proc.encode_conversation(c)
    t_ids, t_lab = SFTCollator(proc, max_length=len(ids) - 4).truncate(ids, labels)
    sup = t_lab[t_lab != IGNORE_INDEX]
    assert proc.tokenizer.decode(sup).replace(" ", "") == "yes.</s>"


def test_all_masked_batch_gives_zero_loss():
    model, proc = tiny_model()
    model.train()
    ids, labels = proc.encode_conversation(_conv("describe the cup", "yes ."))
    out = model(input_ids=ids[None], labels=torch.full_like(labels, IGNORE_INDEX)[None])
    assert out.loss.item() == 0.0


# 6 -------------------------------------------------------------------------
def test_trainables_are_fp32_without_deepspeed(tmp_path, monkeypatch):
    from transformers import TrainingArguments

    import vot.models.motionepic.builder as builder
    from vot.train.arguments import DataArguments, ModelArguments
    from vot.train.train import build_model

    m, p = tiny_model()
    m.to(torch.bfloat16)
    monkeypatch.setattr(builder, "build_motionepic", lambda cfg, **kw: (m, p))
    targs = TrainingArguments(output_dir=str(tmp_path), report_to=[], use_cpu=True)
    model, proc = build_model(ModelArguments(train_stsg_encoder=True), DataArguments(), targs)
    trainable = [q for q in model.parameters() if q.requires_grad]
    assert trainable and all(q.dtype == torch.float32 for q in trainable)
    assert model.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
    model.train()
    clip, g = tiny_clip(3), kitchen_stsg()
    ids, labels = proc.encode_conversation(_conv("<video>\n<stsg>\nis the video paired", "yes ."))
    vids, vmask = proc.collate_videos([proc.process_video(clip)])
    stsg = proc.featurizer.collate([proc.featurizer.encode(g, clip)])
    out = model(input_ids=ids[None], labels=labels[None], videos=vids, video_mask=vmask, stsg=stsg)
    out.loss.backward()
    assert model.video_projector.proj.weight.grad.dtype == torch.float32


# 7 -------------------------------------------------------------------------
def test_corpus_builder_skips_test_split(tmp_path):
    rows = [
        {"video_id": "tr1", "split": "train", "frame_times": [0.0, 1.0, 2.0], "stsg": kitchen_stsg().to_dict()},
        {"video_id": "te1", "split": "test", "frame_times": [0.0, 1.0, 2.0], "stsg": kitchen_stsg().to_dict()},
    ]
    src = tmp_path / "stsg.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    script = os.path.join(ROOT, "scripts", "data", "build_grounding_corpus.py")

    def run(*extra):
        out = tmp_path / "out.jsonl"
        subprocess.run([sys.executable, script, "--stsg", str(src), "--out", str(out), *extra], check=True, capture_output=True)
        return {json.loads(line)["id"].split("-")[0] for line in out.read_text().splitlines()}

    assert run() == {"tr1"}
    assert run("--include_test") == {"tr1", "te1"}


# 8 -------------------------------------------------------------------------
def test_every_stsg_parameter_gets_a_gradient():
    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=True)
    model.train()
    assert not hasattr(model.stsg_encoder.layers[-1], "o_e")
    assert hasattr(model.stsg_encoder.layers[0], "o_e")
    nobox = STSG.from_frames([{"objects": {"person": None, "cup": None}}, {"objects": {"person": None}}])
    for g in (nobox, kitchen_stsg()):
        model.zero_grad()
        clip = tiny_clip(3)
        ids, labels = proc.encode_conversation(_conv("<video>\n<stsg>\nis the video paired", "yes ."))
        vids, vmask = proc.collate_videos([proc.process_video(clip)])
        stsg = proc.featurizer.collate([proc.featurizer.encode(g, clip)])
        model(input_ids=ids[None], labels=labels[None], videos=vids, video_mask=vmask, stsg=stsg).loss.backward()
        missing = [n for n, q in model.stsg_encoder.named_parameters() if q.grad is None]
        assert not missing, missing


# 9 -------------------------------------------------------------------------
def test_frames_without_times_align_by_index(tmp_path):
    _, proc = tiny_model()
    files = []
    for i in range(3):
        Image.new("RGB", (64, 48), (120 * i, 0, 0)).save(tmp_path / f"{i}.png")
        files.append(f"{i}.png")
    g = kitchen_stsg()
    for f, t in zip(g.frames, [4.1, 4.6, 5.1]):  # real annotation times, clip has none
        f.timestamp = t
    ds = SFTDataset([], proc, frames_root=str(tmp_path))
    rec = {"id": "x", "frames": files, "stsg": g.to_dict(),
           "conversations": [{"from": "human", "value": "<video>\n<stsg>\nis the video paired"}, {"from": "gpt", "value": "yes ."}]}
    clip = ds._clip(rec)
    assert clip.meta.get("timestamps_reliable") is False
    feat = proc.featurizer
    picked = [feat._frame_image(fr, clip)[0] for fr in sorted(g.frames, key=lambda f: f.index)]
    assert [im.getpixel((0, 0))[0] for im in picked] == [0, 120, 240]
    x = ds.build(rec)["stsgs"][0]
    reds = x.roi_pixels[:, 0].mean(dim=(1, 2))
    per_frame = [float(reds[x.node_roi[t][x.node_mask[t]]].mean()) for t in range(3)]
    assert per_frame[0] < per_frame[1] < per_frame[2]
    timed = VideoClip(clip.frames, [4.1, 4.6, 5.1])  # real timestamps: matched by time
    assert feat._frame_image(g.frames[2], timed)[0] is clip.frames[2]
