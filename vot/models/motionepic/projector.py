"""Video Q-Former projector.

A frame-level BLIP-2 Q-Former compresses each frame's ViT-L/14 patch tokens into
``num_frame_query_tokens`` queries; temporal position embeddings are added and a
light video-level Q-Former pools all frames into ``num_query_tokens`` video
tokens, which a linear layer maps to the LLM width. The token budget of a video
is thus independent of its length (8 fps sampling can yield many frames).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from .configuration import MotionEpicConfig

logger = logging.getLogger(__name__)


def _qformer(hidden: int, layers: int, heads: int, inter: int, enc_dim: int, cross_freq: int) -> nn.Module:
    from transformers import Blip2QFormerConfig, Blip2QFormerModel

    cfg = Blip2QFormerConfig(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        intermediate_size=inter,
        encoder_hidden_size=enc_dim,
        cross_attention_frequency=cross_freq,
    )
    return Blip2QFormerModel(cfg)


class VideoQFormerProjector(nn.Module):
    def __init__(self, cfg: MotionEpicConfig, vision_hidden_size: int, llm_hidden_size: int) -> None:
        super().__init__()
        h = cfg.qformer_hidden_size
        self.frame_qformer = _qformer(h, cfg.frame_qformer_layers, cfg.qformer_num_heads, cfg.qformer_intermediate_size, vision_hidden_size, 2)
        self.frame_queries = nn.Parameter(torch.zeros(1, cfg.num_frame_query_tokens, h))
        self.temporal_embed = nn.Embedding(cfg.max_frames, h)
        self.video_qformer = _qformer(h, cfg.video_qformer_layers, cfg.qformer_num_heads, cfg.qformer_intermediate_size, h, 1)
        self.video_queries = nn.Parameter(torch.zeros(1, cfg.num_query_tokens, h))
        self.proj = nn.Linear(h, llm_hidden_size)
        self.max_frames = cfg.max_frames
        nn.init.normal_(self.frame_queries, std=0.02)
        nn.init.normal_(self.video_queries, std=0.02)
        nn.init.normal_(self.temporal_embed.weight, std=0.02)

    @property
    def num_tokens(self) -> int:
        return self.video_queries.shape[1]

    def forward(self, frame_feats: torch.Tensor, frame_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``frame_feats [B, T, P, Dv]`` (+ ``frame_mask [B, T]``) -> ``[B, K, D_llm]``."""
        b, t, p, dv = frame_feats.shape
        if t > self.max_frames:
            raise ValueError(f"{t} frames exceed max_frames={self.max_frames}")
        if frame_mask is None:
            frame_mask = torch.ones(b, t, dtype=torch.bool, device=frame_feats.device)
        dtype = self.frame_queries.dtype
        x = frame_feats.reshape(b * t, p, dv).to(dtype)
        q = self.frame_queries.expand(b * t, -1, -1)
        enc_mask = torch.ones(b * t, p, dtype=torch.long, device=x.device)
        fq = self.frame_qformer(query_embeds=q, encoder_hidden_states=x, encoder_attention_mask=enc_mask).last_hidden_state
        nq = fq.shape[1]
        pos = self.temporal_embed(torch.arange(t, device=x.device))
        fq = fq.view(b, t, nq, -1) + pos[None, :, None, :]
        mem = fq.reshape(b, t * nq, -1)
        mem_mask = frame_mask[:, :, None].expand(b, t, nq).reshape(b, t * nq).long()
        vq = self.video_queries.expand(b, -1, -1)
        out = self.video_qformer(query_embeds=vq, encoder_hidden_states=mem, encoder_attention_mask=mem_mask).last_hidden_state
        return self.proj(out)

    def load_blip2_frame_qformer(self, name_or_path: str) -> None:
        """Initialise the frame Q-Former (and its queries) from a BLIP-2 checkpoint.

        BLIP-2 cross-attends to EVA ViT-g (1408-d) features, so the cross-attention
        key/value projections do not match ViT-L/14 (1024-d) and stay random; all
        shape-compatible weights are copied.
        """
        from transformers import Blip2Model

        src = Blip2Model.from_pretrained(name_or_path, torch_dtype=torch.float32)
        sd = {k[len("qformer."):]: v for k, v in src.state_dict().items() if k.startswith("qformer.")}
        own = self.frame_qformer.state_dict()
        ok = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
        self.frame_qformer.load_state_dict(ok, strict=False)
        q = src.query_tokens.data
        if q.shape[-1] == self.frame_queries.shape[-1]:
            n = min(q.shape[1], self.frame_queries.shape[1])
            self.frame_queries.data[:, :n] = q[:, :n].to(self.frame_queries.dtype)
        logger.info("BLIP-2 Q-Former init: copied %d/%d tensors", len(ok), len(own))
        del src
