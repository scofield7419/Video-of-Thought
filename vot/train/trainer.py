"""HF Trainer that saves only what MotionEpic trains (projector, STSG encoder, LoRA)."""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import torch
from transformers import Trainer

from ..models.motionepic.builder import save_motionepic

logger = logging.getLogger(__name__)

MM_KEYS = ("video_projector", "stsg_encoder")


class MotionEpicTrainer(Trainer):
    def __init__(self, *args, mm_lr: Optional[float] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mm_lr = mm_lr

    def create_optimizer(self):
        """Separate learning rate for the multimodal modules (``mm_lr``)."""
        if self.optimizer is not None or self.mm_lr is None:
            return super().create_optimizer()
        decay = set(self.get_decay_parameter_names(self.model))
        groups: Dict[str, Dict] = {}
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            is_mm = any(k in n for k in MM_KEYS)
            key = f"{'mm' if is_mm else 'llm'}_{'decay' if n in decay else 'nodecay'}"
            g = groups.setdefault(key, {"params": [], "weight_decay": self.args.weight_decay if n in decay else 0.0})
            if is_mm:
                g["lr"] = self.mm_lr
            g["params"].append(p)
        cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = cls(list(groups.values()), **kwargs)
        return self.optimizer

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None) -> None:
        """Resume (non-DeepSpeed) from our checkpoint layout instead of full weights."""
        from safetensors.torch import load_file

        from ..models.motionepic.builder import unwrap
        from ..models.motionepic.configuration import LORA_DIRNAME, TRAINABLES_NAME

        model = model or self.model
        path = os.path.join(resume_from_checkpoint, TRAINABLES_NAME)
        if os.path.exists(path):
            unwrap(model).load_mm_state_dict(load_file(path), strict=False)
        lora_dir = os.path.join(resume_from_checkpoint, LORA_DIRNAME)
        inner = getattr(model, "module", model)
        if os.path.isdir(lora_dir) and hasattr(inner, "peft_config"):
            from peft import set_peft_model_state_dict
            from peft.utils import load_peft_weights

            set_peft_model_state_dict(inner, load_peft_weights(lora_dir))
        logger.info("Resumed MotionEpic trainables from %s", resume_from_checkpoint)

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_motionepic(self.model, output_dir, state_dict=state_dict)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saved MotionEpic trainables to %s", output_dir)
