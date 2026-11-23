"""Frozen CLIP towers.

* :class:`CLIPVisionTower` (ViT-L/14) yields per-frame patch features for the
  Q-Former and joint-space image embeddings for STSG RoIs (node features f_i).
* :class:`PhraseEncoder` embeds object categories c_i and predicates into the
  same joint space (CLIP text tower), with a per-string cache since STSG
  vocabularies are small and heavily repeated.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration import MotionEpicConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, cfg: MotionEpicConfig, load_weights: bool = True) -> None:
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

        self.select_layer = cfg.vision_select_layer
        self.select_feature = cfg.vision_select_feature
        if cfg.vision_config is not None:
            self.model = CLIPVisionModelWithProjection(CLIPVisionConfig(**cfg.vision_config))
        elif load_weights:
            self.model = CLIPVisionModelWithProjection.from_pretrained(cfg.vision_name_or_path)
        else:
            self.model = CLIPVisionModelWithProjection(CLIPVisionConfig.from_pretrained(cfg.vision_name_or_path))
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode: bool = True) -> "CLIPVisionTower":
        super().train(False)  # always frozen / eval (no dropout)
        return self

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    @property
    def embed_dim(self) -> int:
        return self.model.config.projection_dim

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def frame_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W] -> [N, P, hidden]`` patch tokens of the selected layer."""
        out = self.model.vision_model(
            pixel_values=pixel_values.to(device=self.device, dtype=self.dtype), output_hidden_states=True
        )
        feats = out.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            feats = feats[:, 1:]
        return feats

    @torch.no_grad()
    def image_embeds(self, pixel_values: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        """``[N, 3, H, W] -> [N, D]`` L2-normalised joint-space embeddings."""
        outs = []
        for i in range(0, pixel_values.shape[0], chunk):
            px = pixel_values[i : i + chunk].to(device=self.device, dtype=self.dtype)
            outs.append(self.model(pixel_values=px).image_embeds)
        if not outs:
            return torch.zeros(0, self.embed_dim, device=self.device, dtype=self.dtype)
        return F.normalize(torch.cat(outs), dim=-1)


class PhraseEncoder(nn.Module):
    """Embeds short phrases (category labels, predicates) with a cache."""

    def __init__(self, cfg: MotionEpicConfig, load_weights: bool = True) -> None:
        super().__init__()
        self.kind = cfg.phrase_encoder
        self.dim = cfg.stsg_feature_dim
        self._cache: Dict[str, torch.Tensor] = {}
        self.model = None
        self.tokenizer = None
        if self.kind == "clip":
            from transformers import AutoTokenizer, CLIPTextConfig, CLIPTextModelWithProjection

            if cfg.text_config is not None:
                self.model = CLIPTextModelWithProjection(CLIPTextConfig(**cfg.text_config))
            elif load_weights:
                self.model = CLIPTextModelWithProjection.from_pretrained(cfg.vision_name_or_path)
            else:
                self.model = CLIPTextModelWithProjection(CLIPTextConfig.from_pretrained(cfg.vision_name_or_path))
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.vision_name_or_path)
            self.model.requires_grad_(False)
            self.model.eval()
            self.dim = self.model.config.projection_dim
        elif self.kind != "hash":
            raise ValueError(f"Unknown phrase encoder {self.kind!r}")

    def train(self, mode: bool = True) -> "PhraseEncoder":
        super().train(False)
        return self

    def _hash_vec(self, text: str) -> torch.Tensor:
        seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
        g = torch.Generator().manual_seed(seed)
        return F.normalize(torch.randn(self.dim, generator=g), dim=-1)

    @torch.no_grad()
    def forward(self, phrases: List[str], device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        missing = [p for p in dict.fromkeys(phrases) if p not in self._cache]
        if missing:
            if self.kind == "hash":
                for p in missing:
                    self._cache[p] = self._hash_vec(p)
            else:
                dev = next(self.model.parameters()).device
                toks = self.tokenizer([p.replace("_", " ") for p in missing], padding=True, truncation=True, return_tensors="pt").to(dev)
                emb = F.normalize(self.model(**toks).text_embeds.float(), dim=-1).cpu()
                for p, e in zip(missing, emb):
                    self._cache[p] = e
        if not phrases:
            return torch.zeros(0, self.dim, device=device, dtype=dtype or torch.float32)
        out = torch.stack([self._cache[p] for p in phrases])
        return out.to(device=device, dtype=dtype or out.dtype)
