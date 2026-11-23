"""API backends are tested offline with fake SDK clients."""

from types import SimpleNamespace

import pytest

from vot.backends.base import GenerationConfig, Message, STSGPart, VideoPart


class _Recorder:
    def __init__(self, response):
        self.kwargs = None
        self.response = response

    def create(self, **kw):
        self.kwargs = kw
        return self.response


def test_openai_rendering_and_params(clip3, truck_stsg):
    pytest.importorskip("openai")
    from vot.backends.openai_compat import OpenAICompatibleBackend

    b = OpenAICompatibleBackend(model="gpt-4o", api_key="test")
    resp = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2), model="gpt-4o", id="x",
    )
    rec = _Recorder(resp)
    b.client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    msgs = [Message.system("sys"), Message.user("<Video>\n", VideoPart(clip3), "\n</Video>\n", STSGPart(truck_stsg), "Q?")]
    r = b.generate(msgs, GenerationConfig(max_new_tokens=64, temperature=0.0))
    assert r.text == "ok" and r.prompt_tokens == 11
    sent = rec.kwargs
    assert sent["max_tokens"] == 64 and sent["temperature"] == 0.0
    content = sent["messages"][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 3
    assert content[1]["text"].startswith("Frame 1 (t=0.50s)")
    assert any(c["type"] == "text" and "<STSG>" in c["text"] and "Temporal Coreference Edge" in c["text"] for c in content)

    reasoning = OpenAICompatibleBackend(model="o3", api_key="test")
    reasoning.client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    reasoning.generate(msgs)
    assert "temperature" not in rec.kwargs and rec.kwargs["max_completion_tokens"] >= 16000


def test_anthropic_rendering_and_params(clip3, truck_stsg):
    pytest.importorskip("anthropic")
    from vot.backends.anthropic_backend import FALLBACK_BETA, AnthropicBackend

    b = AnthropicBackend(api_key="test")
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=5, output_tokens=3, cache_read_input_tokens=100, cache_creation_input_tokens=0),
        stop_reason="end_turn", model="claude-opus-5-5",
    )
    rec = _Recorder(resp)
    b.client = SimpleNamespace(beta=SimpleNamespace(messages=rec), messages=rec)
    msgs = [
        Message.system("sys"),
        Message.user("<Video>\n", VideoPart(clip3), "\n</Video>\nQ1"),
        Message.assistant("A1"),
        Message.user(STSGPart(truck_stsg), "Q2"),
    ]
    r = b.generate(msgs, GenerationConfig(max_new_tokens=512, temperature=0.0, top_p=1.0))
    assert r.text == "answer" and r.prompt_tokens == 105
    kw = rec.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "sys"
    assert "temperature" not in kw and "top_p" not in kw
    assert kw["max_tokens"] >= 16000 and kw["output_config"] == {"effort": "medium"}
    assert kw["betas"] == [FALLBACK_BETA] and kw["fallbacks"] == "default"
    roles = [m["role"] for m in kw["messages"]]
    assert roles == ["user", "assistant", "user"]
    imgs = [blk for blk in kw["messages"][0]["content"] if blk["type"] == "image"]
    assert len(imgs) == 3 and imgs[-1].get("cache_control") == {"type": "ephemeral"} and "cache_control" not in imgs[0]
    assert imgs[0]["source"]["media_type"] == "image/jpeg"


def test_anthropic_never_sends_empty_content(clip3):
    pytest.importorskip("anthropic")
    from vot.backends.anthropic_backend import AnthropicBackend

    b = AnthropicBackend(api_key="test")
    payload = b.to_api([Message.system("s"), Message.user("q"), Message.assistant(""), Message.user("q2")])
    assert all(m["content"] for m in payload["messages"])
