"""Plug any video LLM into VoT by implementing one method.

A backend receives a list of :class:`vot.backends.Message` whose content mixes
text, a sampled video (``VideoPart`` with PIL frames + timestamps) and STSGs
(``STSGPart``). Render them however your model expects and return the text.
"""

from typing import Sequence

from vot import VideoOfThought
from vot.backends import GenerationConfig, LLMResponse, Message, STSGPart, TextPart, VideoLLM, VideoPart
from vot.backends.base import BackendCapabilities, render_stsg_text


class MyVideoLLM(VideoLLM):
    name = "my-model"
    capabilities = BackendCapabilities(native_stsg=False, knows_stsg_format=False)

    def __init__(self, model):
        super().__init__()
        self.model = model  # e.g. a HF pipeline taking (frames, prompt)

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        frames, prompt = [], []
        for m in messages:
            for p in m.content:
                if isinstance(p, VideoPart):
                    frames = p.clip.frames
                    prompt.append("<video>")
                elif isinstance(p, STSGPart):
                    prompt.append(render_stsg_text(p))
                elif isinstance(p, TextPart):
                    prompt.append(p.text)
            prompt.append("\n")
        text = self.model(frames=frames, prompt="".join(prompt), max_new_tokens=gen.max_new_tokens)
        return LLMResponse(text=text)


if __name__ == "__main__":
    echo = lambda frames, prompt, max_new_tokens: "The involved targets are [the person]."  # noqa: E731
    vot = VideoOfThought(MyVideoLLM(echo))
    print("backend ready:", vot.model.describe())
