"""Shared helpers for benchmark converters."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from ..videoqa import VideoQASample, is_unified_file, load_samples

RawPath = Union[str, Sequence[str]]


def as_list(raw: RawPath) -> List[str]:
    return [raw] if isinstance(raw, str) else list(raw)


def maybe_unified(raw: RawPath, benchmark: str) -> Optional[List[VideoQASample]]:
    """If ``raw`` already is a unified JSONL file, load it (tagging the benchmark)."""
    paths = as_list(raw)
    if len(paths) == 1 and is_unified_file(paths[0]):
        samples = load_samples(paths[0])
        for s in samples:
            s.benchmark = s.benchmark or benchmark
        return samples
    return None


def find_split_file(raw: RawPath, split: str, patterns: Sequence[str]) -> str:
    """Return ``raw`` if it is a file, else the first ``pattern.format(split=split)`` inside it."""
    paths = as_list(raw)
    for p in paths:
        if os.path.isfile(p):
            return p
    for p in paths:
        if os.path.isdir(p):
            for pat in patterns:
                cand = os.path.join(p, pat.format(split=split))
                if os.path.isfile(cand):
                    return cand
    raise FileNotFoundError(f"No annotation file for split '{split}' in {paths} (tried {list(patterns)})")


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_records(path: str) -> List[Dict[str, Any]]:
    """Load a ``.csv``, ``.json`` (list or dict of records) or ``.jsonl`` file as records."""
    if path.endswith(".csv"):
        return read_csv(path)
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    data = read_json(path)
    if isinstance(data, dict):
        if all(isinstance(v, dict) for v in data.values()):
            return [{"_key": k, **v} for k, v in data.items()]
        raise ValueError(f"Unsupported JSON layout in {path}")
    return list(data)


def read_id_list(path: str) -> List[str]:
    """Video-id list from ``.txt`` (one per line), ``.json`` or ``.pkl``."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            ids = pickle.load(f)
    elif path.endswith(".json"):
        ids = read_json(path)
    else:
        with open(path, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
    return [str(i) for i in ids]


def stable_seed(*parts: Any) -> int:
    h = hashlib.md5("::".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def stable_shuffle(items: Iterable[Any], *seed_parts: Any) -> List[Any]:
    """Deterministic permutation (independent of PYTHONHASHSEED)."""
    import random

    out = list(items)
    random.Random(stable_seed(*seed_parts)).shuffle(out)
    return out


def to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
