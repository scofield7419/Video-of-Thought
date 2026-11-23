"""Tiny random MotionEpic for offline tests (no downloads)."""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WORDS = (
    "a chat between curious user and an artificial intelligence assistant the gives helpful detailed polite answers "
    "to user's questions USER ASSISTANT : is the spatial-temporal scene graph paired with video answer yes or no "
    "yes. no. frame objects triplets temporal coreference edge person cup table holding looking at on in front of "
    "describe what happens provide tracklet output label object partial stsg of describe . , [ ] ( ) { } \" < > / "
    "STSG generate expression"
).split()

TINY_VISION = dict(
    hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
    image_size=32, patch_size=8, projection_dim=16,
)


def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    vocab = {t: i for i, t in enumerate(specials)}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    for i in range(200):
        vocab.setdefault(str(i), len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")


def tiny_config(**overrides):
    from vot.models.motionepic import MotionEpicConfig

    kw = dict(
        vision_config=TINY_VISION, phrase_encoder="hash", image_size=32,
        qformer_hidden_size=32, qformer_num_heads=2, qformer_intermediate_size=64,
        frame_qformer_layers=2, video_qformer_layers=1, num_frame_query_tokens=4, num_query_tokens=6, max_frames=8,
        stsg_feature_dim=16, stsg_hidden_size=32, stsg_num_layers=2, stsg_num_heads=4, stsg_num_tokens=5,
        stsg_resampler_layers=1, max_stsg_frames=8, max_stsg_nodes=6, stsg_dropout=0.0, roi_min_size=4,
    )
    kw.update(overrides)
    return MotionEpicConfig(**kw)


def tiny_model(seed: int = 0, **overrides):
    import torch
    from transformers import LlamaConfig

    from vot.models.motionepic.builder import build_motionepic
    from vot.models.motionepic.modeling import MotionEpicForCausalLM

    torch.manual_seed(seed)
    tok = tiny_tokenizer()
    lcfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=48, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    llm = MotionEpicForCausalLM(lcfg)
    model, proc = build_motionepic(tiny_config(**overrides), torch_dtype=torch.float32, llm=llm, tokenizer=tok, load_vision_weights=False)
    return model, proc


def tiny_clip(n: int = 4, size=(64, 48)):
    from PIL import Image

    from vot.video import VideoClip

    return VideoClip([Image.new("RGB", size, (30 * i, 100, 160)) for i in range(n)], [0.5 * i for i in range(n)])


def kitchen_stsg():
    from vot.stsg import STSG

    return STSG.from_frames(
        [
            {"objects": {"person": [10, 10, 30, 80], "cup": [50, 50, 10, 10], "table": [40, 60, 50, 30]},
             "triplets": [("person", "looking at", "cup"), ("cup", "on", "table")]},
            {"objects": {"person": [12, 10, 30, 80], "cup": [35, 40, 10, 10], "table": [40, 60, 50, 30]},
             "triplets": [("person", "holding", "cup")]},
            {"objects": {"person": [14, 10, 30, 80], "cup": [30, 30, 10, 10]},
             "triplets": [("person", "holding", "cup")]},
        ]
    )
