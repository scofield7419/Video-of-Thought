"""MotionEpic: LLaMA/Vicuna causal LM with video and STSG token splicing.

The multimodal modules are attached *after* the LLM is loaded
(:meth:`MotionEpicForCausalLM.attach_multimodal`), so that ``from_pretrained``
of a plain Vicuna checkpoint never has to initialise them.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
from transformers import LlamaForCausalLM

from .configuration import IGNORE_INDEX, STSG_TOKEN_INDEX, VIDEO_TOKEN_INDEX, MotionEpicConfig
from .featurizer import STSGBatch
from .projector import VideoQFormerProjector
from .stsg_encoder import RecurrentGraphTransformer
from .vision import CLIPVisionTower, PhraseEncoder

TRAINABLE_PREFIXES = ("video_projector.", "stsg_encoder.")

logger = logging.getLogger(__name__)


class MotionEpicForCausalLM(LlamaForCausalLM):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.mm_config: Optional[MotionEpicConfig] = None
        self.model_max_length: Optional[int] = None

    # ------------------------------------------------------------ assembly
    def attach_multimodal(
        self,
        cfg: MotionEpicConfig,
        load_vision_weights: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> "MotionEpicForCausalLM":
        dev = self.get_input_embeddings().weight.device
        dtype = dtype or self.get_input_embeddings().weight.dtype
        cfg.llm_hidden_size = self.config.hidden_size
        self.vision_tower = CLIPVisionTower(cfg, load_weights=load_vision_weights).to(dev, dtype)
        self.phrase_encoder = PhraseEncoder(cfg, load_weights=load_vision_weights).to(dev)
        if self.phrase_encoder.dim != cfg.stsg_feature_dim or self.vision_tower.embed_dim != cfg.stsg_feature_dim:
            raise ValueError(
                f"stsg_feature_dim={cfg.stsg_feature_dim} must equal the CLIP joint-space dim "
                f"(vision {self.vision_tower.embed_dim}, text {self.phrase_encoder.dim})"
            )
        self.video_projector = VideoQFormerProjector(cfg, self.vision_tower.hidden_size, self.config.hidden_size).to(dev, dtype)
        self.stsg_encoder = RecurrentGraphTransformer(cfg, self.config.hidden_size).to(dev, dtype)
        self.mm_config = cfg
        self.config.motionepic = cfg.to_dict()
        return self

    @property
    def has_multimodal(self) -> bool:
        return self.mm_config is not None

    def mm_trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.state_dict().items() if k.startswith(TRAINABLE_PREFIXES)}

    def load_mm_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True) -> List[str]:
        own = self.mm_trainable_state_dict()
        sd = {k: v for k, v in sd.items() if k.startswith(TRAINABLE_PREFIXES)}
        missing = [k for k in own if k not in sd]
        unexpected = [k for k in sd if k not in own]
        if strict and (missing or unexpected):
            raise KeyError(f"missing={missing[:5]}... unexpected={unexpected[:5]}...")
        self.load_state_dict(sd, strict=False)
        return missing

    def set_mm_trainable(self, projector: bool, stsg_encoder: bool) -> None:
        self.video_projector.requires_grad_(projector)
        self.stsg_encoder.requires_grad_(stsg_encoder)

    # ------------------------------------------------------------- encoders
    def encode_video(self, videos: torch.Tensor, video_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[V, T, 3, H, W] -> [V, K, D]``; padded frames are skipped by the ViT."""
        v, t = videos.shape[:2]
        if video_mask is None:
            video_mask = torch.ones(v, t, dtype=torch.bool, device=videos.device)
        video_mask = video_mask.to(videos.device).bool()
        flat = videos.flatten(0, 1)
        valid = video_mask.flatten()
        feats_valid = self.vision_tower.frame_features(flat[valid])
        feats = feats_valid.new_zeros(v * t, *feats_valid.shape[1:])
        feats[valid] = feats_valid
        feats = feats.view(v, t, *feats.shape[1:])
        return self.video_projector(feats, video_mask)

    def encode_stsg(self, batch: STSGBatch) -> torch.Tensor:
        """``STSGBatch -> [G, K, D]``."""
        dev = self.get_input_embeddings().weight.device
        batch = batch.to(dev, self.vision_tower.dtype)
        f = self.mm_config.stsg_feature_dim
        if batch.roi_pixels is not None and batch.roi_pixels.shape[0] > 0:
            roi = self.vision_tower.image_embeds(batch.roi_pixels).float()
        else:
            roi = torch.zeros(1, f, device=dev)
        has_roi = batch.node_roi >= 0
        roi_feat = roi[batch.node_roi.clamp(min=0)] * has_roi[..., None]
        phr = self.phrase_encoder(batch.phrases or ["object"], device=dev, dtype=torch.float32)
        label_feat = phr[batch.node_label.clamp(min=0)] * (batch.node_label >= 0)[..., None]
        edge_feat = phr[batch.edge_phrase] if batch.edge_phrase.numel() else torch.zeros(0, f, device=dev)
        tokens, _ = self.stsg_encoder(
            roi_feat,
            has_roi,
            label_feat,
            batch.node_box,
            batch.node_mask,
            batch.frame_time,
            batch.frame_mask,
            batch.edge_index,
            edge_feat,
            batch.coref_prev,
        )
        return tokens

    # -------------------------------------------------------------- splicing
    def prepare_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        stsg: Optional[STSGBatch] = None,
        padding_side: str = "right",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Replace sentinel ids by video / STSG tokens.

        ``videos`` and ``stsg`` hold one entry per ``<video>`` / ``<stsg>``
        placeholder in the batch, in reading order. Returns
        ``(inputs_embeds, attention_mask, position_ids, labels)``.
        """
        embed = self.get_input_embeddings()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.bool()
        n_vid = int((input_ids == VIDEO_TOKEN_INDEX).sum())
        n_sg = int((input_ids == STSG_TOKEN_INDEX).sum())
        vid_tok = self.encode_video(videos, video_mask) if (videos is not None and n_vid) else None
        sg_tok = self.encode_stsg(stsg) if (stsg is not None and n_sg) else None
        if n_vid != (0 if vid_tok is None else vid_tok.shape[0]):
            raise ValueError(f"{n_vid} <video> placeholders but {0 if videos is None else videos.shape[0]} videos")
        if n_sg != (0 if sg_tok is None else sg_tok.shape[0]):
            raise ValueError(f"{n_sg} <stsg> placeholders but {0 if stsg is None else len(stsg)} STSGs")

        embeds_out, labels_out = [], []
        vi = si = 0
        for b in range(input_ids.shape[0]):
            keep = attention_mask[b]
            ids = input_ids[b][keep]
            lab = labels[b][keep] if labels is not None else torch.full_like(ids, IGNORE_INDEX)
            text = embed(ids.clamp(min=0))
            pieces, lpieces, start = [], [], 0
            for p in torch.nonzero(ids < 0).flatten().tolist():
                pieces.append(text[start:p])
                lpieces.append(lab[start:p])
                if ids[p] == VIDEO_TOKEN_INDEX:
                    mm = vid_tok[vi]
                    vi += 1
                elif ids[p] == STSG_TOKEN_INDEX:
                    mm = sg_tok[si]
                    si += 1
                else:
                    raise ValueError(f"unknown sentinel id {int(ids[p])}")
                pieces.append(mm.to(text.dtype))
                lpieces.append(torch.full((mm.shape[0],), IGNORE_INDEX, dtype=lab.dtype, device=lab.device))
                start = p + 1
            pieces.append(text[start:])
            lpieces.append(lab[start:])
            e, l = torch.cat(pieces), torch.cat(lpieces)
            limit = self.model_max_length
            if limit and e.shape[0] > limit:
                if labels is not None:
                    # training: SFTCollator already fits samples; this is only a safety net
                    e, l = e[:limit], l[:limit]
                else:
                    # inference: keep the tail so the prompt still ends with "ASSISTANT:"
                    logger.warning("Prompt of %d tokens exceeds model_max_length=%d; keeping the last %d", e.shape[0], limit, limit)
                    e, l = e[-limit:], l[-limit:]
            embeds_out.append(e)
            labels_out.append(l)

        if self.training:
            embeds_out[0] = embeds_out[0] + self._unused_trainables_term(vid_tok is None, sg_tok is None)

        n = max(e.shape[0] for e in embeds_out)
        bsz, dim = len(embeds_out), embeds_out[0].shape[-1]
        dev, dtype = embeds_out[0].device, embeds_out[0].dtype
        inputs_embeds = torch.zeros(bsz, n, dim, device=dev, dtype=dtype)
        new_labels = torch.full((bsz, n), IGNORE_INDEX, dtype=torch.long, device=dev)
        new_mask = torch.zeros(bsz, n, dtype=torch.long, device=dev)
        position_ids = torch.zeros(bsz, n, dtype=torch.long, device=dev)
        for i, (e, l) in enumerate(zip(embeds_out, labels_out)):
            m = e.shape[0]
            sl = slice(0, m) if padding_side == "right" else slice(n - m, n)
            inputs_embeds[i, sl] = e
            new_labels[i, sl] = l
            new_mask[i, sl] = 1
            position_ids[i, sl] = torch.arange(m, device=dev)
        return inputs_embeds, new_mask, position_ids, (new_labels if labels is not None else None)

    def _unused_trainables_term(self, video_unused: bool, stsg_unused: bool) -> torch.Tensor:
        """Zero-valued term touching trainable modules that a batch did not use,
        so DDP / ZeRO see a gradient for every trainable parameter."""
        term = None
        for unused, mod in ((video_unused, self.video_projector), (stsg_unused, self.stsg_encoder)):
            if not unused:
                continue
            for p in mod.parameters():
                if p.requires_grad:
                    t = p.sum() * 0.0
                    term = t if term is None else term + t
        dtype = self.get_input_embeddings().weight.dtype
        return term.to(dtype) if term is not None else torch.zeros((), dtype=dtype, device=self.get_input_embeddings().weight.device)

    def _needs_mm(self, input_ids: Optional[torch.Tensor]) -> bool:
        if input_ids is None or not self.has_multimodal:
            return False
        if bool((input_ids < 0).any()):
            return True
        return self.training and any(p.requires_grad for m in (self.video_projector, self.stsg_encoder) for p in m.parameters())

    # ------------------------------------------------------------ hf entry
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep=0,
        videos: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        stsg: Optional[STSGBatch] = None,
        **kwargs,
    ):
        if inputs_embeds is None and self._needs_mm(input_ids):
            inputs_embeds, attention_mask, position_ids, labels = self.prepare_multimodal_inputs(
                input_ids, attention_mask, labels, videos, video_mask, stsg, padding_side="right"
            )
            input_ids = None
        if labels is not None and not bool((labels != IGNORE_INDEX).any()):
            # an all-masked batch (e.g. every reply truncated) gives a zero loss instead of NaN
            out = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                cache_position=cache_position, logits_to_keep=logits_to_keep,
                **{k: v for k, v in kwargs.items() if k != "num_items_in_batch"},
            )
            out.loss = out.logits.sum() * 0.0
            return out
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        stsg: Optional[STSGBatch] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Generation always runs from ``inputs_embeds``, so the output holds
        only the newly generated tokens."""
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            if self.has_multimodal:
                inputs_embeds, attention_mask, _, _ = self.prepare_multimodal_inputs(
                    input_ids, attention_mask, None, videos, video_mask, stsg, padding_side="left"
                )
            else:
                inputs_embeds = self.get_input_embeddings()(input_ids)
        return super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
