"""VideoQA metrics and benchmark reports.

Prediction records are dicts ``{"qid": ..., "pred": ...}`` where ``pred`` is an
option index (``int``; letters such as ``"B"``/``"B. text"`` or the option
text are also accepted) for multiple choice, a list of indices for
multi-answer questions and a string for open-ended questions. ``predictions``
may be a list of such records or a ``{qid: pred}`` mapping. Missing
predictions count as wrong.

All accuracies are percentages. Benchmark reports reproduce the columns of the
paper's tables:

* NExT-QA: Acc@All, Acc@C, Acc@T, Acc@D (Table 3)
* STAR: Int., Seq., Pre., Fea. (Table 1) and their mean
* Causal-VidQA: Acc@D, Acc@E, Acc@P (A, R, AR), Acc@C (A, R, AR) (Table 2),
  plus Acc@All = mean(D, E, P-AR, C-AR)
* Social-IQ: 2-way, 4-way (Table 1)
* MSRVTT-QA / ActivityNet-QA: open-ended accuracy (normalized exact match, or
  an LLM judge following the Video-ChatGPT protocol) and token F1.
"""

from __future__ import annotations

import ast
import re
import string
from collections import Counter, OrderedDict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..data.videoqa import OPTION_LETTERS, VideoQASample

Pred = Union[int, str, List[Any], None]
JudgeResult = Union[bool, Tuple[bool, float]]
#: ``judge(question, gold_answer, prediction) -> bool | (bool, score)``
JudgeFn = Callable[[str, str, str], JudgeResult]

NEXTQA_GROUPS: Dict[str, str] = {
    "CW": "C", "CH": "C", "TN": "T", "TC": "T", "TP": "T", "DL": "D", "DC": "D", "DO": "D",
}
INTENTQA_GROUPS: Dict[str, str] = {"CW": "Why", "CH": "How", "TP": "Before&After", "TN": "Before&After", "TP&TN": "Before&After"}
STAR_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("Interaction", "Int."), ("Sequence", "Seq."), ("Prediction", "Pre."), ("Feasibility", "Fea."),
)


# --------------------------------------------------------------- normalization
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def normalize_answer(s: Any) -> str:
    """Lowercase, strip punctuation/articles, map number words to digits."""
    s = str(s).lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(_NUMBER_WORDS.get(t, t) for t in s.split())


def exact_match(pred: Any, gold: Union[str, Sequence[str]]) -> bool:
    golds = [gold] if isinstance(gold, str) else list(gold)
    p = normalize_answer(pred)
    return any(p == normalize_answer(g) for g in golds)


def token_f1(pred: Any, gold: Union[str, Sequence[str]]) -> float:
    golds = [gold] if isinstance(gold, str) else list(gold)
    p_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            best = max(best, float(p_toks == g_toks))
            continue
        common = Counter(p_toks) & Counter(g_toks)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p_toks), same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


# ------------------------------------------------------------- choice parsing
_PAREN_LETTER = re.compile(r"^\(\s*([A-Za-z])\s*\)")
_LETTER = re.compile(r"^([A-Z])(?:\s*[.:)\]]|$)")
_PREFIXED = re.compile(r"^(?:option|answer|choice)\s*[:#]?\s*\(?([A-Za-z])\)?(?=\s*(?:[.:)\]]|$))", re.IGNORECASE)
_ANSWER_IS = re.compile(r"(?i:answer)\s*(?:is|:)?\s*\[?\(?([A-Z])(?=\s*(?:[.:)\]]|$))")


def coerce_choice(pred: Any, candidates: Optional[Sequence[str]] = None, n: Optional[int] = None) -> Optional[int]:
    """Map a raw prediction to an option index (``None`` if it cannot be resolved)."""
    if n is None and candidates is not None:
        n = len(candidates)

    def ok(i: int) -> Optional[int]:
        return i if (n is None or 0 <= i < n) else None

    if pred is None or isinstance(pred, bool):
        return None
    if isinstance(pred, int):
        return ok(pred)
    if isinstance(pred, float) and pred.is_integer():
        return ok(int(pred))
    s = str(pred).strip()
    if not s:
        return None
    if s.lstrip("-").isdigit():
        return ok(int(s))
    if candidates:
        norm = normalize_answer(s)
        exact = [i for i, c in enumerate(candidates) if normalize_answer(c) == norm]
        if len(exact) == 1:
            return exact[0]
    for pat in (_PAREN_LETTER, _LETTER, _PREFIXED):
        m = pat.match(s)
        if m:
            return ok(OPTION_LETTERS.index(m.group(1).upper()))
    m = _ANSWER_IS.search(s)
    if m:
        return ok(OPTION_LETTERS.index(m.group(1)))
    if candidates:
        norm = normalize_answer(s)
        hits = [i for i, c in enumerate(candidates) if normalize_answer(c) and normalize_answer(c) in norm]
        if len(hits) == 1:
            return hits[0]
    return None


def coerce_choices(pred: Any, candidates: Optional[Sequence[str]] = None) -> Optional[List[int]]:
    if pred is None:
        return None
    items = pred if isinstance(pred, (list, tuple, set)) else re.split(r"[,;\s]+", str(pred).strip())
    out = []
    for p in items:
        i = coerce_choice(p, candidates)
        if i is None:
            return None
        out.append(i)
    return sorted(set(out))


# --------------------------------------------------------------- correctness
def _as_pred_map(predictions: Union[Mapping[str, Pred], Iterable[Mapping[str, Any]]]) -> Dict[str, Pred]:
    if isinstance(predictions, Mapping):
        return {str(k): v for k, v in predictions.items()}
    out: Dict[str, Pred] = {}
    for rec in predictions:
        out[str(rec["qid"])] = rec.get("pred")
    return out


def _judge(judge: JudgeFn, sample: VideoQASample, pred: Any) -> Tuple[bool, Optional[float]]:
    res = judge(sample.question, str(sample.answer), "" if pred is None else str(pred))
    if isinstance(res, tuple):
        return bool(res[0]), float(res[1])
    return bool(res), None


def is_correct(sample: VideoQASample, pred: Pred, judge: Optional[JudgeFn] = None) -> bool:
    if pred is None or sample.answer is None:
        return False
    if sample.is_multi_choice:
        if isinstance(sample.answer, list):
            got = coerce_choices(pred, sample.candidates)
            return got is not None and got == sorted(set(int(a) for a in sample.answer))
        return coerce_choice(pred, sample.candidates) == int(sample.answer)
    if judge is not None:
        return _judge(judge, sample, pred)[0]
    return exact_match(pred, str(sample.answer))


def _acc(flags: Sequence[bool]) -> float:
    return 100.0 * sum(flags) / len(flags) if flags else float("nan")


def accuracy(
    samples: Sequence[VideoQASample],
    predictions: Union[Mapping[str, Pred], Iterable[Mapping[str, Any]]],
    judge: Optional[JudgeFn] = None,
) -> float:
    preds = _as_pred_map(predictions)
    labeled = [s for s in samples if s.answer is not None]
    return _acc([is_correct(s, preds.get(s.qid), judge) for s in labeled])


def _per_category(samples: Sequence[VideoQASample], flags: Dict[str, bool], key: Callable[[VideoQASample], Optional[str]]):
    groups: Dict[str, List[bool]] = OrderedDict()
    for s in samples:
        k = key(s)
        if k is None:
            continue
        groups.setdefault(k, []).append(flags[s.qid])
    return OrderedDict((k, {"n": len(v), "acc": _acc(v)}) for k, v in groups.items())


# ------------------------------------------------------------ benchmark reports
def _base(samples: Sequence[VideoQASample], preds: Dict[str, Pred], judge: Optional[JudgeFn]):
    labeled = [s for s in samples if s.answer is not None]
    flags = {s.qid: is_correct(s, preds.get(s.qid), judge) for s in labeled}
    report: Dict[str, Any] = {
        "n": len(labeled),
        "answered": sum(1 for s in labeled if preds.get(s.qid) is not None),
        "metrics": OrderedDict(),
        "per_category": _per_category(labeled, flags, lambda s: s.category),
    }
    return labeled, flags, report


def report_generic(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    report["metrics"]["Acc"] = _acc(list(flags.values()))
    return report


def report_nextqa(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    m = report["metrics"]
    m["Acc@All"] = _acc(list(flags.values()))
    by_group = _per_category(labeled, flags, lambda s: NEXTQA_GROUPS.get(s.category or ""))
    for g in ("C", "T", "D"):
        m[f"Acc@{g}"] = by_group.get(g, {"acc": float("nan")})["acc"]
    return report


def report_intentqa(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    m = report["metrics"]
    by_group = _per_category(labeled, flags, lambda s: INTENTQA_GROUPS.get(s.category or ""))
    for g in ("Why", "How", "Before&After"):
        m[g] = by_group.get(g, {"acc": float("nan")})["acc"]
    m["Total"] = _acc(list(flags.values()))
    return report


def report_star(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    m = report["metrics"]
    cats = report["per_category"]
    vals = []
    for full, short in STAR_COLUMNS:
        v = cats.get(full, {"acc": float("nan")})["acc"]
        m[short] = v
        if v == v:  # not NaN
            vals.append(v)
    m["Mean"] = sum(vals) / len(vals) if vals else float("nan")
    m["Overall"] = _acc(list(flags.values()))
    return report


def report_causalvidqa(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    m = report["metrics"]

    def part_acc(qtype: str, part: str) -> float:
        return _acc([flags[s.qid] for s in labeled if s.meta.get("qtype") == qtype and s.meta.get("part") == part])

    def ar_acc(qtype: str) -> float:
        groups: Dict[str, Dict[str, bool]] = {}
        for s in labeled:
            if s.meta.get("qtype") == qtype:
                groups.setdefault(s.meta.get("group", s.qid), {})[s.meta.get("part", "answer")] = flags[s.qid]
        return _acc([g.get("answer", False) and g.get("reason", False) for g in groups.values()])

    m["Acc@D"] = part_acc("descriptive", "answer")
    m["Acc@E"] = part_acc("explanatory", "answer")
    for qtype, code in (("predictive", "P"), ("counterfactual", "C")):
        m[f"Acc@{code}-A"] = part_acc(qtype, "answer")
        m[f"Acc@{code}-R"] = part_acc(qtype, "reason")
        m[f"Acc@{code}-AR"] = ar_acc(qtype)
    parts = [m["Acc@D"], m["Acc@E"], m["Acc@P-AR"], m["Acc@C-AR"]]
    valid = [p for p in parts if p == p]
    m["Acc@All"] = sum(valid) / len(valid) if valid else float("nan")
    return report


def report_socialiq(samples, preds, judge=None) -> Dict[str, Any]:
    labeled, flags, report = _base(samples, preds, judge)
    for way in ("2-way", "4-way"):
        if way in report["per_category"]:
            report["metrics"][way] = report["per_category"][way]["acc"]
    return report


def report_open_ended(samples, preds, judge=None) -> Dict[str, Any]:
    labeled = [s for s in samples if s.answer is not None]
    em = {s.qid: preds.get(s.qid) is not None and exact_match(preds.get(s.qid), str(s.answer)) for s in labeled}
    f1 = [token_f1(preds.get(s.qid) or "", str(s.answer)) for s in labeled]
    report: Dict[str, Any] = {
        "n": len(labeled),
        "answered": sum(1 for s in labeled if preds.get(s.qid) is not None),
        "metrics": OrderedDict(),
    }
    m = report["metrics"]
    if judge is not None:
        verdicts, scores = {}, []
        for s in labeled:
            p = preds.get(s.qid)
            ok, sc = _judge(judge, s, p) if p is not None else (False, 0.0)
            verdicts[s.qid] = ok
            if sc is not None:
                scores.append(sc)
        m["Acc"] = _acc(list(verdicts.values()))
        if scores:
            m["Score"] = sum(scores) / len(scores)
        report["per_category"] = _per_category(labeled, verdicts, lambda s: s.category)
    else:
        m["Acc"] = _acc(list(em.values()))
        report["per_category"] = _per_category(labeled, em, lambda s: s.category)
    m["EM"] = _acc(list(em.values()))
    m["F1"] = 100.0 * sum(f1) / len(f1) if f1 else float("nan")
    return report


REPORTS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "nextqa": report_nextqa,
    "intentqa": report_intentqa,
    "star": report_star,
    "vlep": report_generic,
    "causalvidqa": report_causalvidqa,
    "socialiq": report_socialiq,
    "msrvtt_qa": report_open_ended,
    "activitynet_qa": report_open_ended,
}


def evaluate(
    samples: Sequence[VideoQASample],
    predictions: Union[Mapping[str, Pred], Iterable[Mapping[str, Any]]],
    benchmark: Optional[str] = None,
    judge: Optional[JudgeFn] = None,
) -> Dict[str, Any]:
    """Score predictions; ``benchmark`` selects the paper-table report layout."""
    preds = _as_pred_map(predictions)
    name = benchmark or (samples[0].benchmark if samples else None)
    if name in REPORTS:
        fn = REPORTS[name]
    elif samples and not samples[0].is_multi_choice:
        fn = report_open_ended
    else:
        fn = report_generic
    report = fn(samples, preds, judge)
    report["benchmark"] = name
    report["missing"] = report["n"] - report["answered"]
    return report


# ---------------------------------------------------------------- LLM judge
JUDGE_SYSTEM_PROMPT = (
    "You evaluate answers to video questions. Given a question, the reference answer and a predicted answer, "
    "decide whether the prediction matches the reference in meaning. Focus on meaningful agreement rather than "
    "exact wording; synonyms and paraphrases count as correct. Reply only with a Python dictionary string of the "
    "form {'pred': 'yes', 'score': 4}, where 'pred' is 'yes' or 'no' and 'score' is an integer from 0 to 5 "
    "rating how well the prediction matches (5 = perfect match)."
)
JUDGE_USER_PROMPT = (
    "Question: {question}\nReference answer: {answer}\nPredicted answer: {pred}\n"
    "Return the dictionary only, with no other text."
)


def parse_judge_output(text: str) -> Tuple[bool, float]:
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            d = ast.literal_eval(m.group(0))
            if isinstance(d, dict) and "pred" in d:
                return str(d["pred"]).strip().lower().startswith("yes"), float(d.get("score", 0))
        except (ValueError, SyntaxError):
            pass
    yes = re.search(r"pred['\"]?\s*[:=]\s*['\"]?(yes|no)", text, re.IGNORECASE)
    score = re.search(r"score['\"]?\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    ok = bool(yes and yes.group(1).lower() == "yes")
    return ok, float(score.group(1)) if score else 0.0


def make_llm_judge(complete: Callable[[str, str], str]) -> JudgeFn:
    """Wrap a text-completion callable ``complete(system, user) -> str`` into a
    judge following the Video-ChatGPT open-ended QA protocol (yes/no + 0-5 score)."""

    def judge(question: str, answer: str, pred: str) -> Tuple[bool, float]:
        out = complete(JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT.format(question=question, answer=answer, pred=pred))
        return parse_judge_output(out)

    return judge


# ------------------------------------------------------------------ formatting
def _fmt(v: Any, digits: int) -> str:
    if isinstance(v, float):
        return "-" if v != v else f"{v:.{digits}f}"
    return str(v)


def format_report(report: Dict[str, Any], digits: int = 1, per_category: bool = True) -> str:
    """Markdown tables: headline metrics, then the per-category breakdown."""
    lines = []
    title = report.get("benchmark") or "results"
    lines.append(f"**{title}** (n={report.get('n', 0)}, missing={report.get('missing', 0)})")
    lines.append("")
    metrics = report.get("metrics", {})
    if metrics:
        lines.append("| " + " | ".join(metrics) + " |")
        lines.append("|" + "|".join([":---:"] * len(metrics)) + "|")
        lines.append("| " + " | ".join(_fmt(v, digits) for v in metrics.values()) + " |")
    cats = report.get("per_category") or {}
    if per_category and cats:
        lines += ["", "| Category | n | Acc |", "|---|---:|---:|"]
        for k, v in cats.items():
            lines.append(f"| {k} | {v['n']} | {_fmt(v['acc'], digits)} |")
    return "\n".join(lines)


__all__ = [
    "JudgeFn",
    "REPORTS",
    "accuracy",
    "coerce_choice",
    "coerce_choices",
    "evaluate",
    "exact_match",
    "format_report",
    "is_correct",
    "make_llm_judge",
    "normalize_answer",
    "parse_judge_output",
    "token_f1",
]
