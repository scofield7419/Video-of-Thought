"""MotionEpic training entry point (all stages).

    torchrun --nproc_per_node 8 -m vot.train.train --config configs/train/stage2_grounding.yaml \\
        --deepspeed configs/deepspeed/zero2.json --output_dir checkpoints/stage2

Values in ``--config`` (YAML) become argument defaults; explicit command-line
flags override them.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional, Sequence

import torch

from .arguments import DataArguments, ModelArguments

logger = logging.getLogger("vot.train")


def parse_args(argv: Optional[Sequence[str]] = None):
    import yaml
    from transformers import HfArgumentParser, TrainingArguments

    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, rest = pre.parse_known_args(argv)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if known.config:
        with open(known.config) as f:
            cfg = yaml.safe_load(f) or {}
        valid = {a.dest for a in parser._actions}
        unknown = sorted(set(cfg) - valid)
        if unknown:
            raise ValueError(f"Unknown keys in {known.config}: {unknown}")
        parser.set_defaults(**cfg)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=rest)
    training_args.remove_unused_columns = False
    check_supported(training_args)
    return model_args, data_args, training_args


def zero_stage(training_args) -> int:
    """ZeRO stage of ``--deepspeed`` (path or dict), 0 without DeepSpeed."""
    ds = getattr(training_args, "deepspeed", None)
    if not ds:
        return 0
    if isinstance(ds, str):
        import json

        with open(ds) as f:
            ds = json.load(f)
    stage = (ds.get("zero_optimization") or {}).get("stage", 0)
    return int(stage) if str(stage).isdigit() else 0


def check_supported(training_args) -> None:
    """ZeRO-3 is not supported: ranks run data-dependent modules (video / STSG
    encoders, RoI embeddings, phrase-cache misses), which deadlocks parameter
    gathering, and BLIP-2 init / LoRA merging assume unpartitioned weights. With a
    frozen 7B LLM, ZeRO-2 fits on 40GB+ GPUs."""
    if zero_stage(training_args) >= 3:
        raise ValueError(
            "DeepSpeed ZeRO-3 is not supported for MotionEpic training (data-dependent multimodal branches); "
            "use configs/deepspeed/zero2.json."
        )


def _dtype(training_args) -> torch.dtype:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return torch.float32


def build_model(model_args: ModelArguments, data_args: DataArguments, training_args):
    check_supported(training_args)
    from ..models.motionepic.builder import (
        TRAINABLES_NAME,
        apply_lora,
        build_motionepic,
        load_lora,
        unwrap,
    )
    from ..models.motionepic.configuration import LORA_DIRNAME, MotionEpicConfig

    if model_args.motionepic_config:
        p = model_args.motionepic_config
        cfg = MotionEpicConfig.from_yaml(p) if p.endswith((".yaml", ".yml")) else MotionEpicConfig.load(p)
    elif model_args.init_from:
        cfg = MotionEpicConfig.load(model_args.init_from)
    else:
        cfg = MotionEpicConfig()
    cfg.llm_name_or_path = model_args.llm_name_or_path
    cfg.vision_name_or_path = model_args.vision_name_or_path
    cfg.sample_fps = data_args.sample_fps
    cfg.max_frames = max(cfg.max_frames, data_args.max_frames)
    cfg.blip2_qformer_init = None if model_args.init_from else model_args.blip2_qformer_init

    model, processor = build_motionepic(
        cfg,
        torch_dtype=_dtype(training_args),
        attn_implementation=model_args.attn_implementation,
        model_max_length=model_args.model_max_length,
    )
    has_prev_lora = False
    if model_args.init_from:
        from safetensors.torch import load_file

        w = os.path.join(model_args.init_from, TRAINABLES_NAME)
        missing = model.load_mm_state_dict(load_file(w), strict=False)
        logger.info("Initialised multimodal modules from %s (%d tensors missing)", w, len(missing))
        has_prev_lora = os.path.isdir(os.path.join(model_args.init_from, LORA_DIRNAME))

    model.requires_grad_(False)
    model.set_mm_trainable(model_args.train_projector, model_args.train_stsg_encoder)
    if has_prev_lora:
        lora_dir = os.path.join(model_args.init_from, LORA_DIRNAME)
        if model_args.lora_enable:
            flags = {n: p.requires_grad for n, p in model.named_parameters()}
            model = load_lora(model, lora_dir, is_trainable=True)
            for n, p in unwrap(model).named_parameters():
                if "lora_" not in n and flags.get(n, False):
                    p.requires_grad_(True)
        else:
            model = load_lora(model, lora_dir).merge_and_unload()
            model.requires_grad_(False)
            model.set_mm_trainable(model_args.train_projector, model_args.train_stsg_encoder)
    elif model_args.lora_enable:
        model = apply_lora(model, r=model_args.lora_r, alpha=model_args.lora_alpha, dropout=model_args.lora_dropout)

    if not training_args.deepspeed:
        # HF Trainer (no DeepSpeed) keeps no fp32 master weights: small AdamW updates
        # vanish in bf16, so trainable parameters are held in fp32 (outputs are cast
        # back to the LLM dtype when spliced; autocast handles the matmuls)
        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.float()
    if training_args.gradient_checkpointing:
        unwrap(model).enable_input_require_grads()
        if training_args.gradient_checkpointing_kwargs is None:
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    unwrap(model).config.use_cache = False

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info("Trainable parameters: %.2fM / %.2fM", n_train / 1e6, n_all / 1e6)
    return model, processor


def train(argv: Optional[List[str]] = None) -> None:
    from ..data.sft import SFTCollator, SFTDataset, load_records
    from .trainer import MotionEpicTrainer

    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
    model_args, data_args, training_args = parse_args(argv)
    if not data_args.data_path:
        raise ValueError("--data_path is required")
    model, processor = build_model(model_args, data_args, training_args)

    records = load_records(data_args.data_path)
    if data_args.max_samples:
        records = records[: data_args.max_samples]
    dataset = SFTDataset(
        records,
        processor,
        video_root=data_args.video_root,
        frames_root=data_args.frames_root,
        stsg_root=data_args.stsg_root,
        sample_fps=data_args.sample_fps,
        max_frames=data_args.max_frames,
        system=data_args.system_prompt,
    )
    collator = SFTCollator(processor, max_length=model_args.model_max_length)
    trainer = MotionEpicTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        mm_lr=model_args.mm_lr,
    )
    ckpts = [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")] if os.path.isdir(training_args.output_dir) else []
    trainer.train(resume_from_checkpoint=bool(ckpts) or None)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
