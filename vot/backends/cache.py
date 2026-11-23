"""Persistent response cache for any backend (reproducible, resumable evaluation)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from typing import Any, Dict, Sequence

from .base import GenerationConfig, LLMResponse, Message, STSGPart, TextPart, VideoLLM, VideoPart


def message_key(messages: Sequence[Message], gen: GenerationConfig, backend: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(backend, sort_keys=True, default=str).encode())
    h.update(json.dumps(gen.__dict__, sort_keys=True, default=str).encode())
    for m in messages:
        h.update(m.role.encode())
        for p in m.content:
            if isinstance(p, TextPart):
                h.update(b"T" + p.text.encode())
            elif isinstance(p, VideoPart):
                h.update(b"V" + p.clip.fingerprint().encode())
            elif isinstance(p, STSGPart):
                h.update(b"G" + p.stsg.to_json(sort_keys=True).encode() + str(p.max_frames).encode())
    return h.hexdigest()


class CachedLLM(VideoLLM):
    """Wrap a backend with an SQLite cache keyed by (backend, generation config, messages)."""

    def __init__(self, inner: VideoLLM, path: str = ".cache/vot_llm.sqlite") -> None:
        super().__init__()
        self.inner = inner
        self.name = f"cached:{inner.name}"
        self.capabilities = inner.capabilities
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()

    def _generate(self, messages: Sequence[Message], gen: GenerationConfig) -> LLMResponse:
        key = message_key(messages, gen, self.inner.describe())
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row is not None:
            d = json.loads(row[0])
            return LLMResponse(text=d["text"], prompt_tokens=d.get("p", 0), completion_tokens=d.get("c", 0), model=d.get("m"), cached=True)
        r = self.inner.generate(messages, gen)
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?)",
                (key, json.dumps({"text": r.text, "p": r.prompt_tokens, "c": r.completion_tokens, "m": r.model})),
            )
            self._db.commit()
        return r

    def describe(self) -> Dict[str, Any]:
        return {**self.inner.describe(), "cached": True}
