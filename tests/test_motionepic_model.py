import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from _tiny_motionepic import kitchen_stsg, tiny_clip, tiny_model  # noqa: E402

from vot.models.motionepic import IGNORE_INDEX, STSG_TOKEN_INDEX, VIDEO_TOKEN_INDEX, vicuna_v1  # noqa: E402


def _conv(user, reply=None):
    c = vicuna_v1()
    c.append("USER", user)
    c.append("ASSISTANT", reply)
    return c


def test_processor_masks_instructions_and_inserts_sentinels():
    _, proc = tiny_model()
    ids, labels = proc.encode_conversation(_conv("<video>\n<stsg>\nis the video paired", "yes."))
    assert (ids == VIDEO_TOKEN_INDEX).sum() == 1 and (ids == STSG_TOKEN_INDEX).sum() == 1
    sup = labels[labels != IGNORE_INDEX]
    assert sup[-1] == proc.tokenizer.eos_token_id
    assert proc.tokenizer.decode(sup[:-1]).replace(" ", "") == "yes."
    assert ids[0] == proc.tokenizer.bos_token_id
    prompt = proc.encode_prompt(_conv("<video>\ndescribe"))
    assert prompt[-1] == proc.tokenizer.convert_tokens_to_ids(":")


def test_splicing_lengths_labels_and_padding():
    model, proc = tiny_model()
    model.eval()
    clip, g = tiny_clip(4), kitchen_stsg()
    a_ids, a_lab = proc.encode_conversation(_conv("<video>\n<stsg>\nis the video paired", "yes."))
    b_ids, b_lab = proc.encode_conversation(_conv("describe the cup", "the person is holding the cup ."))
    ids = proc.pad([a_ids, b_ids], proc.tokenizer.pad_token_id)
    labels = proc.pad([a_lab, b_lab], IGNORE_INDEX)
    mask = proc.pad([torch.ones_like(a_ids), torch.ones_like(b_ids)], 0)
    videos, vmask = proc.collate_videos([proc.process_video(clip)])
    stsg = proc.featurizer.collate([proc.featurizer.encode(g, clip)])
    kv, ks = model.video_projector.num_tokens, model.stsg_encoder.num_tokens
    for side in ("right", "left"):
        emb, am, pos, lab = model.prepare_multimodal_inputs(ids, mask, labels, videos, vmask, stsg, padding_side=side)
        la, lb = len(a_ids) - 2 + kv + ks, len(b_ids)
        assert emb.shape[:2] == (2, max(la, lb))
        assert am.sum(1).tolist() == [la, lb]
        assert (lab[0] != IGNORE_INDEX).sum() == (a_lab != IGNORE_INDEX).sum()
        assert (lab[1] != IGNORE_INDEX).sum() == (b_lab != IGNORE_INDEX).sum()
        if side == "right":
            vpos = int((a_ids == VIDEO_TOKEN_INDEX).nonzero())
            assert torch.allclose(emb[0, vpos : vpos + kv], model.encode_video(videos, vmask)[0], atol=1e-5)
            assert (lab[0, vpos : vpos + kv + ks] == IGNORE_INDEX).all()
            assert pos[1, :lb].tolist() == list(range(lb))
        else:
            assert am[1, : max(la, lb) - lb].sum() == 0
            assert pos[1, -lb:].tolist() == list(range(lb))


def test_placeholder_count_mismatch_raises():
    model, proc = tiny_model()
    ids, _ = proc.encode_conversation(_conv("<video>\ndescribe", "ok"))
    with pytest.raises(ValueError):
        model.prepare_multimodal_inputs(ids[None], None, None, None, None, None)


def test_train_step_updates_only_trainables():
    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=True)
    model.train()
    clip, g = tiny_clip(4), kitchen_stsg()
    ids, labels = proc.encode_conversation(_conv("<video>\n<stsg>\nis the video paired", "yes."))
    videos, vmask = proc.collate_videos([proc.process_video(clip)])
    stsg = proc.featurizer.collate([proc.featurizer.encode(g, clip)])
    out = model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None], labels=labels[None],
                videos=videos, video_mask=vmask, stsg=stsg)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.video_projector.proj.weight.grad is not None
    assert model.stsg_encoder.out[0].weight.grad is not None
    assert model.model.layers[0].self_attn.q_proj.weight.grad is None
    assert all(p.grad is None for p in model.vision_tower.parameters())


def test_text_only_batch_still_touches_trainables():
    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=True)
    model.train()
    ids, labels = proc.encode_conversation(_conv("describe the cup", "the cup ."))
    out = model(input_ids=ids[None], attention_mask=torch.ones_like(ids)[None], labels=labels[None])
    out.loss.backward()
    grad = model.stsg_encoder.out[0].weight.grad
    assert grad is not None and torch.count_nonzero(grad) == 0


def test_generate_returns_new_tokens_only():
    model, proc = tiny_model()
    model.eval()
    clip, g = tiny_clip(3), kitchen_stsg()
    inputs = proc.build_inputs(_conv("<video>\n<stsg>\ndescribe"), [clip], [(g, clip)])
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=proc.tokenizer.pad_token_id,
                         eos_token_id=proc.tokenizer.eos_token_id)
    assert out.shape[0] == 1 and 1 <= out.shape[1] <= 5


def test_save_and_load_roundtrip(tmp_path):
    from vot.models.motionepic.builder import load_motionepic, save_motionepic

    model, proc = tiny_model(seed=1)
    save_motionepic(model, str(tmp_path))
    other, _ = tiny_model(seed=2)
    loaded, _ = load_motionepic(str(tmp_path), llm=other, tokenizer=proc.tokenizer, torch_dtype=torch.float32,
                                load_vision_weights=False)
    for k, v in model.mm_trainable_state_dict().items():
        assert torch.equal(v, loaded.mm_trainable_state_dict()[k]), k


def test_lora_wrapping_and_save(tmp_path):
    pytest.importorskip("peft")
    from vot.models.motionepic.builder import LORA_TARGET_REGEX, apply_lora, load_motionepic, save_motionepic

    model, proc = tiny_model()
    model.requires_grad_(False)
    model.set_mm_trainable(projector=True, stsg_encoder=False)
    peft_model = apply_lora(model, r=4, alpha=8)
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert any("lora_A" in n for n in names)
    assert any("video_projector" in n for n in names)
    assert not any("stsg_encoder" in n for n in names)
    assert not any("vision_tower" in n for n in names)
    assert LORA_TARGET_REGEX
    save_motionepic(peft_model, str(tmp_path))
    assert (tmp_path / "lora" / "adapter_config.json").exists()
    base, _ = tiny_model(seed=3)
    loaded, _ = load_motionepic(str(tmp_path), llm=base, tokenizer=proc.tokenizer, torch_dtype=torch.float32,
                                load_vision_weights=False)
    assert not hasattr(loaded, "peft_config")


def test_backend_renders_messages_and_generates():
    from vot.backends.base import GenerationConfig, Message, STSGPart, VideoPart
    from vot.backends.motionepic import MotionEpicBackend

    model, proc = tiny_model()
    be = MotionEpicBackend(model=model, processor=proc)
    clip, g = tiny_clip(3), kitchen_stsg()
    msgs = [
        Message.system("you are an expert"),
        Message.user("<Video>", VideoPart(clip), "</Video>", "describe the cup"),
        Message.assistant("the person is holding the cup"),
        Message.user(STSGPart(g), "describe what happens"),
    ]
    conv, clips, stsgs = be.render(msgs)
    prompt = conv.get_prompt()
    assert prompt.startswith("you are an expert USER:")
    assert prompt.count("<video>") == 1 and prompt.count("<stsg>") == 1
    assert "Frame 1:" in prompt and prompt.endswith("ASSISTANT:")
    assert len(clips) == 1 and stsgs[0][1] is clips[0]
    r = be.generate(msgs, GenerationConfig(max_new_tokens=4))
    assert isinstance(r.text, str) and r.completion_tokens <= 4
    be_text = MotionEpicBackend(model=model, processor=proc, stsg_mode="text")
    conv, _, stsgs = be_text.render(msgs)
    assert not stsgs and "<stsg>" not in conv.get_prompt()
    assert be.capabilities.native_stsg and be.capabilities.knows_stsg_format
