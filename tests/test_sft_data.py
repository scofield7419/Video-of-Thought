import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from _tiny_motionepic import kitchen_stsg, tiny_model  # noqa: E402
from PIL import Image  # noqa: E402

from vot.data.grounding_tasks import GroundingSource, GroundingTaskBuilder  # noqa: E402
from vot.data.sft import SFTCollator, SFTDataset, load_records  # noqa: E402
from vot.models.motionepic import IGNORE_INDEX  # noqa: E402


@pytest.fixture
def corpus(tmp_path):
    frames_root = tmp_path / "frames"
    srcs = []
    for v in ("vidA", "vidB"):
        d = frames_root / f"{v}.mp4"
        d.mkdir(parents=True)
        files = []
        for i in range(3):
            Image.new("RGB", (64, 48), (40 * i, 80, 120)).save(d / f"{i:06d}.png")
            files.append(f"{v}.mp4/{i:06d}.png")
        srcs.append(GroundingSource(id=v, stsg=kitchen_stsg(), frames=files, frame_times=[0.0, 1.0, 2.0]))
    recs = []
    for seed in range(3):
        for s in srcs:
            recs.extend(GroundingTaskBuilder(seed=seed).build(s, srcs))
    caption = {"id": "cap", "frames": srcs[0].frames,
               "conversations": [{"from": "human", "value": "describe the video"}, {"from": "gpt", "value": "a person holding a cup ."}]}
    recs.append(caption)
    path = tmp_path / "sft.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return str(path), str(frames_root), recs


def test_load_records_and_dataset_items(corpus):
    path, frames_root, recs = corpus
    assert len(load_records(path)) == len(recs)
    _, proc = tiny_model()
    ds = SFTDataset(path, proc, frames_root=frames_root)
    for i in range(len(ds)):
        item = ds[i]
        rec = recs[i]
        assert len(item["videos"]) == 1 and item["videos"][0].shape[0] == 3
        assert len(item["stsgs"]) == (1 if rec.get("task") == "L1" else 0)
        assert (item["labels"] != IGNORE_INDEX).sum() > 0
    cap = ds[len(ds) - 1]  # caption record without placeholder gets <video> prepended
    assert len(cap["videos"]) == 1


def test_collator_and_forward_backward(corpus):
    path, frames_root, recs = corpus
    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=True)
    model.train()
    ds = SFTDataset(path, proc, frames_root=frames_root)
    l1 = [i for i, r in enumerate(recs) if r.get("task") == "L1"][:2]
    l2 = [i for i, r in enumerate(recs) if r.get("task") == "L2"][:1]
    batch = SFTCollator(proc)([ds[i] for i in l1 + l2])
    assert batch["videos"].shape[0] == 3 and batch["stsg"].node_mask.shape[0] == 2
    out = model(**batch)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.stsg_encoder.out[0].weight.grad.abs().sum() > 0


def test_collator_truncation_drops_cut_placeholders(corpus):
    path, frames_root, recs = corpus
    _, proc = tiny_model()
    ds = SFTDataset(path, proc, frames_root=frames_root)
    l1 = [i for i, r in enumerate(recs) if r.get("task") == "L1"][0]
    batch = SFTCollator(proc, max_length=1)([ds[l1]])
    assert "videos" not in batch and "stsg" not in batch


def test_trainer_saves_only_trainables(corpus, tmp_path):
    from transformers import TrainingArguments

    from vot.models.motionepic.builder import load_motionepic
    from vot.train.trainer import MotionEpicTrainer

    path, frames_root, recs = corpus
    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=True)
    before = {k: v.clone() for k, v in model.mm_trainable_state_dict().items()}
    ds = SFTDataset(path, proc, frames_root=frames_root)
    out_dir = tmp_path / "ckpt"
    args = TrainingArguments(
        output_dir=str(out_dir), per_device_train_batch_size=2, max_steps=2, learning_rate=1e-3,
        report_to=[], save_strategy="no", remove_unused_columns=False, use_cpu=True, logging_steps=1,
    )
    trainer = MotionEpicTrainer(model=model, args=args, train_dataset=ds, data_collator=SFTCollator(proc), mm_lr=5e-3)
    trainer.train()
    lrs = sorted({g["lr"] for g in trainer.optimizer.param_groups})
    assert len(lrs) == 1  # only multimodal params are trainable here
    trainer.save_model(str(out_dir))
    assert sorted(os.listdir(out_dir)) >= ["mm_trainables.safetensors", "motionepic_config.json"]
    after = model.mm_trainable_state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)
    other, _ = tiny_model(seed=5)
    loaded, _ = load_motionepic(str(out_dir), llm=other, tokenizer=proc.tokenizer, torch_dtype=torch.float32, load_vision_weights=False)
    for k, v in after.items():
        assert torch.equal(v, loaded.mm_trainable_state_dict()[k])


def test_train_cli_config_parsing(tmp_path):
    from vot.train.train import parse_args

    cfg = tmp_path / "c.yaml"
    cfg.write_text("data_path: [a.jsonl, b.jsonl]\ntrain_stsg_encoder: true\nlearning_rate: 3.0e-4\noutput_dir: out\n")
    m, d, t = parse_args(["--config", str(cfg), "--learning_rate", "1e-5"])
    assert d.data_path == ["a.jsonl", "b.jsonl"] and m.train_stsg_encoder and t.learning_rate == 1e-5
    assert t.remove_unused_columns is False
    bad = tmp_path / "bad.yaml"
    bad.write_text("learning_rat: 1\n")
    with pytest.raises(ValueError):
        parse_args(["--config", str(bad)])


def test_shipped_stage_configs_parse():
    from vot.train.train import parse_args

    root = os.path.join(os.path.dirname(__file__), "..", "configs", "train")
    for name in sorted(os.listdir(root)):
        # GPU-only flags (bf16/tf32) are switched off to parse on CPU CI
        m, d, t = parse_args(["--config", os.path.join(root, name), "--bf16", "False", "--tf32", "False"])
        assert d.data_path and t.output_dir
    from vot.models.motionepic import MotionEpicConfig

    cfg = MotionEpicConfig.from_yaml(os.path.join(root, "..", "model", "motionepic.yaml"))
    assert cfg.stsg_num_layers == 6 and cfg.stsg_hidden_size == 768 and cfg.sample_fps == 8.0


def test_build_model_chains_stages(tmp_path, monkeypatch):
    pytest.importorskip("peft")
    import vot.models.motionepic.builder as builder
    from vot.train.arguments import DataArguments, ModelArguments
    from vot.train.train import build_model

    model_a, proc_a = tiny_model(seed=11)
    models = iter([(model_a, proc_a), tiny_model(seed=12), tiny_model(seed=13)])
    monkeypatch.setattr(builder, "build_motionepic", lambda cfg, **kw: next(models))
    from transformers import TrainingArguments

    targs = TrainingArguments(output_dir=str(tmp_path / "o"), report_to=[], use_cpu=True)
    m1, _ = build_model(ModelArguments(lora_enable=True, lora_r=4, lora_alpha=8, train_stsg_encoder=True), DataArguments(), targs)
    with torch.no_grad():
        builder.unwrap(m1).video_projector.proj.weight.add_(1.0)
    builder.save_motionepic(m1, str(tmp_path / "stage2"))

    m2, _ = build_model(ModelArguments(init_from=str(tmp_path / "stage2"), lora_enable=True, train_stsg_encoder=True), DataArguments(), targs)
    names = [n for n, p in m2.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in names) and any("stsg_encoder" in n for n in names)
    ref = builder.unwrap(m1).video_projector.proj.weight
    assert torch.equal(builder.unwrap(m2).video_projector.proj.weight, ref)

    m3, _ = build_model(ModelArguments(init_from=str(tmp_path / "stage2"), lora_enable=False), DataArguments(), targs)
    assert not hasattr(m3, "peft_config")
    assert all(("video_projector" in n) for n, p in m3.named_parameters() if p.requires_grad)
