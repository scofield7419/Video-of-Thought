"""Local MotionEpic backend (the paper's own model).

Messages are rendered through the Vicuna v1 template:

* ``VideoPart`` -> ``<video>`` (Q-Former video tokens);
* ``STSGPart`` -> ``<STSG>\\n<stsg>\\n{expression}\\n</STSG>`` by default, i.e. the
  graph is fed both to the STSG encoder (``<stsg>`` tokens, RoI features cropped
  from the most recent video in the conversation) and as its textual expression,
  which is how a generated partial STSG is "recycled" as input of the next VoT
  step (paper Sec. 3.2). ``stsg_mode`` selects ``"graph"``, ``"text"`` or
  ``"graph+text"``.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Sequence, Tuple

from .base import BackendCapabilities, GenerationConfig, LLMResponse, Message, STSGPart, TextPart, VideoLLM, VideoPart

logger = logging.getLogger(__name__)


class MotionEpicBackend(VideoLLM):
    name = "motionepic"
    capabilities = BackendCapabilities(native_stsg=True, knows_stsg_format=True, max_images=None)

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        *,
        model=None,
        processor=None,
        llm_name_or_path: Optional[str] = None,
        vision_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        stsg_mode: str = "graph+text",
        system: Optional[str] = None,
    ) -> None:
        super().__init__()
        import torch

        if stsg_mode not in {"graph", "text", "graph+text"}:
            raise ValueError(f"stsg_mode must be graph/text/graph+text, got {stsg_mode!r}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if model is None:
            if checkpoint is None:
                raise ValueError("Pass a MotionEpic checkpoint directory or (model, processor)")
            from ..models.motionepic.builder import load_motionepic

            model, processor = load_motionepic(
                checkpoint,
                llm_name_or_path=llm_name_or_path,
                vision_name_or_path=vision_name_or_path,
                torch_dtype=torch_dtype,
                device=device,
            )
        elif processor is None:
            raise ValueError("processor is required when model is given")
        self.model = model.eval()
        self.processor = processor
        self.device = next(model.parameters()).device
        self.stsg_mode = stsg_mode
        self.system = system
        self._warned = False

    @classmethod
    def from_pretrained(cls, checkpoint: str, **kwargs) -> "MotionEpicBackend":
        """``build_backend("motionepic:<checkpoint>")`` entry point."""
        return cls(checkpoint=checkpoint, **kwargs)

    def describe(self):
        cfg = self.processor.cfg
        return {"backend": self.name, "llm": cfg.llm_name_or_path, "vision": cfg.vision_name_or_path, "stsg_mode": self.stsg_mode}

    # -------------------------------------------------------------- render
    def _fit_clip(self, clip):
        mf = self.processor.cfg.max_frames
        if len(clip) > mf:
            if not self._warned:
                logger.warning(
                    "Clip with %d frames exceeds max_frames=%d; it is subsampled and STSG frame numbers refer "
                    "to the subsampled clip. Sample videos with max_frames<=%d for consistent grounding.",
                    len(clip), mf, mf,
                )
                self._warned = True
            clip = clip.subsample(mf)
        return clip

    def render(self, messages: Sequence[Message]):
        """Messages -> (Conversation, clips, [(stsg, clip)]) in placeholder order."""
        from ..models.motionepic.conversation import vicuna_v1

        systems = [m.text() for m in messages if m.role == "system"]
        conv = vicuna_v1("\n".join(systems) if systems else self.system)
        clips: List = []
        stsgs: List[Tuple] = []
        last_clip = None
        merged: List[Tuple[str, str]] = []
        for m in messages:
            if m.role == "system":
                continue
            chunks: List[str] = []
            for p in m.content:
                if isinstance(p, TextPart):
                    chunks.append(p.text)
                elif isinstance(p, VideoPart):
                    last_clip = self._fit_clip(p.clip)
                    clips.append(last_clip)
                    chunks.append("<video>")
                elif isinstance(p, STSGPart):
                    chunks.append(self._render_stsg(p, last_clip, stsgs))
            text = "\n".join(c.strip("\n") for c in chunks if c is not None and c != "")
            role = conv.roles[0] if m.role == "user" else conv.roles[1]
            if merged and merged[-1][0] == role:
                merged[-1] = (role, merged[-1][1] + "\n" + text)
            else:
                merged.append((role, text))
        for role, text in merged:
            conv.append(role, text)
        if not merged or merged[-1][0] != conv.roles[1]:
            conv.append(conv.roles[1], None)
        return conv, clips, stsgs

    def _render_stsg(self, part: STSGPart, clip, stsgs: List[Tuple]) -> str:
        expr = part.stsg.to_expression(max_frames=part.max_frames) if not part.stsg.is_empty else "(empty)"
        lines = [f"<{part.label}>"]
        if self.stsg_mode in {"graph", "graph+text"}:
            stsgs.append((part.stsg, clip))
            lines.append("<stsg>")
        if self.stsg_mode in {"text", "graph+text"}:
            lines.append(expr)
        lines.append(f"</{part.label}>")
        return "\n".join(lines)

    # ------------------------------------------------------------ generate
    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        import torch

        conv, clips, stsgs = self.render(messages)
        inputs = self.processor.build_inputs(conv, clips, stsgs)
        n_prompt = int(inputs["input_ids"].shape[1])
        n_prompt += len(clips) * (self.model.video_projector.num_tokens - 1)
        n_prompt += len(stsgs) * (self.model.stsg_encoder.num_tokens - 1)
        for k, v in list(inputs.items()):
            if v is None:
                continue
            inputs[k] = v.to(self.device)
        if gen.seed is not None:
            torch.manual_seed(gen.seed)
        tok = self.processor.tokenizer
        kwargs = dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.temperature > 0,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if gen.temperature > 0:
            kwargs.update(temperature=gen.temperature, top_p=gen.top_p)
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        text = tok.decode(out[0], skip_special_tokens=True).strip()
        for s in gen.stop or []:
            if s in text:
                text = text[: text.index(s)]
        return LLMResponse(
            text=text.strip(),
            prompt_tokens=n_prompt,
            completion_tokens=int(out.shape[1]),
            latency=time.time() - t0,
            model=self.processor.cfg.llm_name_or_path,
        )
