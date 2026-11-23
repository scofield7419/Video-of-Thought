"""Registry of the VideoQA benchmarks evaluated in the paper (Sec. 5.1).

Fine-tuning setting: VLEP, STAR, IntentQA, Social-IQ, Causal-VidQA, NExT-QA.
Zero-shot setting: MSR-VTT-QA, ActivityNet-QA (plus NExT-QA and STAR).

Usage::

    from vot.data.benchmarks import get_benchmark
    spec = get_benchmark("nextqa")
    samples = spec.convert("/data/nextqa", split="val")
    report = spec.evaluate(samples, predictions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..videoqa import VideoQASample
from . import activitynet_qa, causalvidqa, intentqa, msrvtt_qa, nextqa, socialiq, star, vlep


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    convert: Callable[..., List[VideoQASample]]
    task: str  # "mc" (multiple choice) or "oe" (open-ended)
    setting: Tuple[str, ...]  # paper settings: "finetune" and/or "zeroshot"
    default_split: str
    categories: Tuple[str, ...] = ()
    description: str = ""
    category_groups: Dict[str, str] = field(default_factory=dict)

    def evaluate(
        self,
        samples: Sequence[VideoQASample],
        predictions: Any,
        judge: Optional[Callable[..., Any]] = None,
    ) -> Dict[str, Any]:
        from ...eval.metrics import evaluate

        return evaluate(samples, predictions, benchmark=self.name, judge=judge)


BENCHMARKS: Dict[str, BenchmarkSpec] = {
    spec.name: spec
    for spec in [
        BenchmarkSpec(
            "nextqa", nextqa.convert, "mc", ("finetune", "zeroshot"), "val",
            ("Causal", "Temporal", "Descriptive"), "NExT-QA multiple-choice (5 options)", dict(nextqa.TYPE_GROUPS),
        ),
        BenchmarkSpec(
            "star", star.convert, "mc", ("finetune", "zeroshot"), "val", star.CATEGORIES,
            "STAR situated reasoning (4 options)",
        ),
        BenchmarkSpec("vlep", vlep.convert, "mc", ("finetune",), "dev", (), "VLEP future event prediction (2 options)"),
        BenchmarkSpec(
            "intentqa", intentqa.convert, "mc", ("finetune",), "test", ("Why", "How", "Before&After"),
            "IntentQA intent reasoning (5 options)", dict(intentqa.TYPE_GROUPS),
        ),
        BenchmarkSpec(
            "socialiq", socialiq.convert, "mc", ("finetune",), "val", ("2-way", "4-way"),
            "Social-IQ social intelligence QA (2-way / 4-way)",
        ),
        BenchmarkSpec(
            "causalvidqa", causalvidqa.convert, "mc", ("finetune",), "test", ("D", "E", "P", "C"),
            "Causal-VidQA description / explanation / prediction / counterfactual (4 options)",
        ),
        BenchmarkSpec(
            "msrvtt_qa", msrvtt_qa.convert, "oe", ("zeroshot",), "test", msrvtt_qa.QUESTION_TYPES,
            "MSRVTT-QA open-ended",
        ),
        BenchmarkSpec(
            "activitynet_qa", activitynet_qa.convert, "oe", ("zeroshot",), "test",
            tuple(activitynet_qa.ANSWER_TYPES.values()), "ActivityNet-QA open-ended",
        ),
    ]
}

ALIASES: Mapping[str, str] = {
    "next-qa": "nextqa", "next_qa": "nextqa", "intent-qa": "intentqa", "social-iq": "socialiq",
    "social_iq": "socialiq", "causal-vidqa": "causalvidqa", "causal_vidqa": "causalvidqa",
    "msrvtt": "msrvtt_qa", "msr-vtt": "msrvtt_qa", "msrvtt-qa": "msrvtt_qa",
    "activitynet": "activitynet_qa", "anet_qa": "activitynet_qa", "activitynet-qa": "activitynet_qa",
}


def get_benchmark(name: str) -> BenchmarkSpec:
    key = name.lower().strip()
    key = ALIASES.get(key, key)
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[key]


__all__ = ["ALIASES", "BENCHMARKS", "BenchmarkSpec", "get_benchmark"]
