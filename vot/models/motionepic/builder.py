"""Build, save and load MotionEpic.

Checkpoint layout (only what training changes is stored)::

    <ckpt>/
        motionepic_config.json        # MotionEpicConfig (incl. base LLM / CLIP ids)
        mm_trainables.safetensors     # video_projector.* + stsg_encoder.*
        lora/                         # PEFT LoRA adapter of the LLM (optional)
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple, Union

import torch

from .configuration import LORA_DIRNAME, TRAINABLES_NAME, MotionEpicConfig
from .modeling import MotionEpicForCausalLM
from .processor import MotionEpicProcessor

logger = logging.getLogger(__name__)

#: LoRA only on the LLM blocks (CLIP / Q-Former also contain q_proj-like names)
LORA_TARGET_REGEX = r".*model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def _dtype(d: Union[str, torch.dtype, None]) -> Optional[torch.dtype]:
    if d is None or isinstance(d, torch.dtype):
        return d
    return {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "float16": torch.float16,
            "fp32": torch.float32, "float32": torch.float32}[d]


def unwrap(model) -> MotionEpicForCausalLM:
    """Strip PEFT / DDP wrappers."""
    m = getattr(model, "module", model)
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


def build_motionepic(
    cfg: MotionEpicConfig,
    torch_dtype: Union[str, torch.dtype, None] = torch.bfloat16,
    llm: Optional[MotionEpicForCausalLM] = None,
    tokenizer=None,
    load_vision_weights: bool = True,
    attn_implementation: Optional[str] = None,
    model_max_length: int = 4096,
) -> Tuple[MotionEpicForCausalLM, MotionEpicProcessor]:
    """Load the LLM (unless ``llm`` is given), attach the multimodal modules and
    create the matching processor. Multimodal weights are freshly initialised
    (plus optional BLIP-2 Q-Former init)."""
    dtype = _dtype(torch_dtype)
    if llm is None:
        kw = {"torch_dtype": dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        llm = MotionEpicForCausalLM.from_pretrained(cfg.llm_name_or_path, **kw)
    llm.attach_multimodal(cfg, load_vision_weights=load_vision_weights, dtype=dtype)
    if cfg.blip2_qformer_init:
        llm.video_projector.load_blip2_frame_qformer(cfg.blip2_qformer_init)
        llm.video_projector.to(dtype=dtype or llm.dtype)
    llm.model_max_length = model_max_length
    processor = MotionEpicProcessor.from_config(cfg, tokenizer=tokenizer, model_max_length=model_max_length)
    return llm, processor


def apply_lora(model: MotionEpicForCausalLM, r: int = 128, alpha: int = 256, dropout: float = 0.05):
    """Wrap the LLM with LoRA; projector / STSG-encoder trainability is preserved."""
    from peft import LoraConfig, get_peft_model

    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    lcfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGET_REGEX, bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lcfg)
    base = unwrap(peft_model)
    for n, p in base.named_parameters():
        if "lora_" not in n and flags.get(n, False):
            p.requires_grad_(True)
    return peft_model


def load_lora(model: MotionEpicForCausalLM, path: str, is_trainable: bool = False):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, path, is_trainable=is_trainable)


def save_motionepic(model, output_dir: str, state_dict=None) -> None:
    """Save config + trainable multimodal weights + LoRA adapter (if any).

    ``state_dict`` may be supplied by the trainer (e.g. gathered under ZeRO-3).
    """
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)
    base = unwrap(model)
    base.mm_config.save(output_dir)
    if state_dict is None:
        state_dict = base.mm_trainable_state_dict()
    sd = {}
    for k, v in state_dict.items():
        k = k.replace("base_model.model.", "")
        if k.startswith(("video_projector.", "stsg_encoder.")):
            sd[k] = v.detach().cpu().contiguous()
    save_file(sd, os.path.join(output_dir, TRAINABLES_NAME))
    inner = getattr(model, "module", model)
    if hasattr(inner, "peft_config"):
        lora_dir = os.path.join(output_dir, LORA_DIRNAME)
        if state_dict is not None and any("lora_" in k for k in state_dict):
            inner.save_pretrained(lora_dir, state_dict=state_dict)
        else:
            inner.save_pretrained(lora_dir)


def load_motionepic(
    checkpoint: str,
    llm_name_or_path: Optional[str] = None,
    vision_name_or_path: Optional[str] = None,
    torch_dtype: Union[str, torch.dtype, None] = torch.bfloat16,
    device: Optional[Union[str, torch.device]] = None,
    merge_lora: bool = True,
    is_trainable: bool = False,
    tokenizer=None,
    llm: Optional[MotionEpicForCausalLM] = None,
    load_vision_weights: bool = True,
    model_max_length: int = 4096,
):
    """Load a MotionEpic checkpoint directory (see module docstring).

    Returns ``(model, processor)``. With ``merge_lora`` (inference) the adapter is
    merged into the LLM weights; with ``is_trainable`` it stays a trainable
    PEFT model (to continue training in a later stage).
    """
    from safetensors.torch import load_file

    cfg = MotionEpicConfig.load(checkpoint)
    if llm_name_or_path:
        cfg.llm_name_or_path = llm_name_or_path
    if vision_name_or_path:
        cfg.vision_name_or_path = vision_name_or_path
    cfg.blip2_qformer_init = None
    model, processor = build_motionepic(
        cfg, torch_dtype=torch_dtype, llm=llm, tokenizer=tokenizer,
        load_vision_weights=load_vision_weights, model_max_length=model_max_length,
    )
    wpath = os.path.join(checkpoint, TRAINABLES_NAME)
    if os.path.exists(wpath):
        missing = model.load_mm_state_dict(load_file(wpath), strict=False)
        if missing:
            logger.warning("%d multimodal tensors missing from %s", len(missing), wpath)
    else:
        logger.warning("No %s in %s: multimodal modules are randomly initialised", TRAINABLES_NAME, checkpoint)
    lora_dir = os.path.join(checkpoint, LORA_DIRNAME)
    if os.path.isdir(lora_dir):
        model = load_lora(model, lora_dir, is_trainable=is_trainable)
        if merge_lora and not is_trainable:
            model = model.merge_and_unload()
    if device is not None:
        model.to(device)
    if not is_trainable:
        model.eval()
    return model, processor
