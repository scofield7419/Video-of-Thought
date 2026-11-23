"""Recurrent Graph Transformer for spatial-temporal scene graphs.

Paper Sec. 3.1: "we retrofit the Graph Transformer (Dwivedi & Bresson, 2020)
with recurrent propagation to encode the multi-frame STSG information".

* Within a frame, :class:`GraphTransformerLayer` is the edge-featured graph
  transformer of Dwivedi & Bresson: attention is restricted to graph neighbours
  (+ self loop), the per-dimension score ``(Q_i * K_j / sqrt(d)) * E e_ij`` both
  weights the messages and updates the edge stream. Predicate edges are added in
  both directions with a learned direction embedding.
* Across frames, node states are propagated recurrently along temporal
  coreference edges: the input state of node ``i`` at frame ``t`` is
  ``GRU(x_{t,i}, h_{t-1, coref(i)})`` (zero state when the node starts a track).
* A Perceiver resampler pools all node states into ``stsg_num_tokens`` tokens in
  the LLM embedding space.

Everything operates on dense padded tensors (no DGL): ``[B, T, N, ...]`` with
``B`` graphs, ``T`` frames and ``N`` nodes per frame.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import MotionEpicConfig


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    args = x.float()[..., None] * freqs
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class GraphTransformerLayer(nn.Module):
    """Edge-featured graph transformer layer (Dwivedi & Bresson, 2020), dense form."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0, ffn_mult: int = 2, update_edges: bool = True) -> None:
        """``update_edges=False`` (last layer) skips the edge stream, whose output
        would be discarded and leave parameters without gradients."""
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.h, self.d = heads, dim // heads
        self.update_edges = update_edges
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.o_h = nn.Linear(dim, dim)
        self.norm_h1, self.norm_h2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn_h = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        if update_edges:
            self.o_e = nn.Linear(dim, dim)
            self.norm_e1, self.norm_e2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.ffn_e = nn.Sequential(nn.Linear(dim, dim * ffn_mult), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_mult, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, e: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """``h [B, N, D]``, ``e [B, N, N, D]`` (``e[:, i, j]`` = edge j -> i), ``adj [B, N, N]`` bool."""
        b, n, dim = h.shape
        q = self.q(h).view(b, n, 1, self.h, self.d)
        k = self.k(h).view(b, 1, n, self.h, self.d)
        v = self.v(h).view(b, n, self.h, self.d)
        pe = self.e(e).view(b, n, n, self.h, self.d)
        score = q * k / math.sqrt(self.d) * pe  # [B, N(i), N(j), H, d]
        logits = score.sum(-1).clamp(-5.0, 5.0).masked_fill(~adj[..., None], float("-inf"))
        attn = torch.softmax(logits, dim=2).nan_to_num(0.0)  # [B, i, j, H]
        msg = torch.einsum("bijh,bjhd->bihd", self.drop(attn), v).reshape(b, n, dim)

        h1 = self.norm_h1(h + self.drop(self.o_h(msg)))
        h2 = self.norm_h2(h1 + self.drop(self.ffn_h(h1)))
        if not self.update_edges:
            return h2, e
        e1 = self.norm_e1(e + self.drop(self.o_e(score.reshape(b, n, n, dim))))
        e2 = self.norm_e2(e1 + self.drop(self.ffn_e(e1)))
        return h2, e2 * adj[..., None]


class PerceiverResampler(nn.Module):
    def __init__(self, dim: int, num_tokens: int, layers: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.null = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln_q": nn.LayerNorm(dim),
                        "ln_kv": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "ln_ff": nn.LayerNorm(dim),
                        "ff": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                    }
                )
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, mem: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
        """``mem [B, M, D]``, ``mem_mask [B, M]`` (True = valid) -> ``[B, K, D]``.

        A learned null memory slot keeps attention defined for empty graphs.
        """
        b = mem.shape[0]
        mem = torch.cat([self.null.expand(b, -1, -1).to(mem.dtype), mem], dim=1)
        mem_mask = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=mem.device), mem_mask], dim=1)
        lat = self.latents[None].expand(b, -1, -1).to(mem.dtype)
        k = lat.shape[1]
        for blk in self.blocks:
            q = blk["ln_q"](lat)
            kv = torch.cat([blk["ln_kv"](mem), q], dim=1)
            pad = torch.cat([~mem_mask, torch.zeros(b, k, dtype=torch.bool, device=mem.device)], dim=1)
            lat = lat + blk["attn"](q, kv, kv, key_padding_mask=pad, need_weights=False)[0]
            lat = lat + blk["ff"](blk["ln_ff"](lat))
        return self.norm(lat)


class RecurrentGraphTransformer(nn.Module):
    def __init__(self, cfg: MotionEpicConfig, llm_hidden_size: int) -> None:
        super().__init__()
        f, d = cfg.stsg_feature_dim, cfg.stsg_hidden_size
        self.in_roi = nn.Linear(f, d)
        self.in_label = nn.Linear(f, d)
        self.no_roi = nn.Parameter(torch.zeros(d))
        self.in_box = nn.Sequential(nn.Linear(8, d), nn.GELU(), nn.Linear(d, d))
        self.in_time = nn.Linear(d, d)
        self.frame_pos = nn.Embedding(cfg.max_stsg_frames, d)
        self.in_norm = nn.LayerNorm(d)
        self.in_edge = nn.Linear(f, d)
        self.edge_dir = nn.Parameter(torch.randn(2, d) * 0.02)
        self.self_edge = nn.Parameter(torch.randn(d) * 0.02)
        self.gru = nn.GRUCell(d, d)
        n_layers = cfg.stsg_num_layers
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(d, cfg.stsg_num_heads, cfg.stsg_dropout, cfg.stsg_ffn_mult, update_edges=i < n_layers - 1)
                for i in range(n_layers)
            ]
        )
        self.resampler = PerceiverResampler(d, cfg.stsg_num_tokens, cfg.stsg_resampler_layers, cfg.stsg_num_heads, cfg.stsg_dropout)
        self.out = nn.Sequential(nn.Linear(d, llm_hidden_size), nn.GELU(), nn.Linear(llm_hidden_size, llm_hidden_size))
        self.hidden = d
        self.max_frames = cfg.max_stsg_frames

    @property
    def num_tokens(self) -> int:
        return self.resampler.latents.shape[0]

    def node_inputs(self, roi_feat, has_roi, label_feat, node_box, frame_time) -> torch.Tensor:
        b, t, n, _ = roi_feat.shape
        dtype = self.in_roi.weight.dtype
        vis = torch.where(has_roi[..., None], self.in_roi(roi_feat.to(dtype)), self.no_roi.to(dtype))
        box = node_box.to(dtype)
        has_box = (box[..., 2:].sum(-1, keepdim=True) > 0).to(dtype)
        cx, cy = box[..., :1] + box[..., 2:3] / 2, box[..., 1:2] + box[..., 3:4] / 2
        box_feat = torch.cat([box, cx, cy, box[..., 2:3] * box[..., 3:4], has_box], dim=-1)
        time = self.in_time(sinusoidal(frame_time, self.hidden).to(dtype))
        pos = self.frame_pos(torch.arange(t, device=roi_feat.device).clamp(max=self.max_frames - 1))
        x = vis + self.in_label(label_feat.to(dtype)) + self.in_box(box_feat) + (time + pos)[:, :, None, :]
        return self.in_norm(x)

    def frame_edges(self, b: int, n: int, t: int, edge_index: torch.Tensor, edge_feat: torch.Tensor, node_mask_t: torch.Tensor):
        """Dense edge tensor + adjacency of frame ``t`` (both directions, self loops)."""
        dtype = self.in_edge.weight.dtype
        dev = node_mask_t.device
        e = torch.zeros(b, n, n, self.hidden, device=dev, dtype=dtype)
        cnt = torch.zeros(b, n, n, 1, device=dev, dtype=dtype)
        if edge_index.numel():
            sel = edge_index[:, 1] == t
            if sel.any():
                ei, ef = edge_index[sel], self.in_edge(edge_feat[sel].to(dtype))
                bb, s, o = ei[:, 0], ei[:, 2], ei[:, 3]
                one = torch.ones(len(bb), 1, device=dev, dtype=dtype)
                e.index_put_((bb, s, o), ef + self.edge_dir[0], accumulate=True)  # s receives from o
                cnt.index_put_((bb, s, o), one, accumulate=True)
                e.index_put_((bb, o, s), ef + self.edge_dir[1], accumulate=True)
                cnt.index_put_((bb, o, s), one, accumulate=True)
        adj = cnt[..., 0] > 0
        e = e / cnt.clamp(min=1)
        eye = torch.eye(n, dtype=torch.bool, device=dev)[None]
        e = torch.where(eye[..., None], self.self_edge.to(dtype), e)
        valid = node_mask_t[:, :, None] & node_mask_t[:, None, :]
        adj = (adj & valid) | eye  # padded nodes only see themselves
        return e, adj

    def forward(
        self,
        roi_feat: torch.Tensor,
        has_roi: torch.Tensor,
        label_feat: torch.Tensor,
        node_box: torch.Tensor,
        node_mask: torch.Tensor,
        frame_time: torch.Tensor,
        frame_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        coref_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(tokens [B, K, D_llm], node_states [B, T, N, D])``.

        ``edge_index [E, 4]`` rows are ``(graph, frame, subject, object)`` with
        ``edge_feat [E, F]`` predicate embeddings; ``coref_prev [B, T, N]`` is the
        index of the coreferent node in frame ``t-1`` or -1.
        """
        b, t, n, _ = roi_feat.shape
        x = self.node_inputs(roi_feat, has_roi, label_feat, node_box, frame_time)
        states = []
        prev: Optional[torch.Tensor] = None
        for ti in range(t):
            xt = x[:, ti]
            if prev is None:
                s = torch.zeros_like(xt)
            else:
                idx = coref_prev[:, ti]
                s = torch.gather(prev, 1, idx.clamp(min=0)[..., None].expand(-1, -1, self.hidden))
                s = s * (idx >= 0)[..., None].to(s.dtype)
            h = self.gru(xt.reshape(b * n, -1), s.reshape(b * n, -1)).view(b, n, -1)
            mask_t = node_mask[:, ti] & frame_mask[:, ti, None]
            e, adj = self.frame_edges(b, n, ti, edge_index, edge_feat, mask_t)
            for layer in self.layers:
                h, e = layer(h, e, adj)
            h = h * mask_t[..., None].to(h.dtype)
            states.append(h)
            prev = h
        node_states = torch.stack(states, dim=1)
        pos = self.frame_pos(torch.arange(t, device=x.device).clamp(max=self.max_frames - 1))
        mem = (node_states + pos[None, :, None, :]).reshape(b, t * n, -1)
        mem_mask = (node_mask & frame_mask[..., None]).reshape(b, t * n)
        tokens = self.out(self.resampler(mem, mem_mask))
        if self.training and edge_index.numel() == 0:
            # edge-less batch: keep in_edge / edge_dir in the graph (DDP needs a gradient for every parameter)
            unused = sum(p.sum() for p in (*self.in_edge.parameters(), self.edge_dir)) * 0.0
            tokens = tokens + unused.to(tokens.dtype)
        return tokens, node_states
