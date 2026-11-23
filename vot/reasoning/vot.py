"""The Video-of-Thought reasoning engine (paper Sec. 4, Fig. 4, Appendix A.1).

    Step-1  task definition + target identification          -> [Target]
    Step-2  object tracking: partial STSG of the targets      -> [Target Tracklet]
    Step-3  action analysing over tracklet + neighbour scenes -> [Observation and Implication]
    Step-4  QA via ranking: score every candidate 1-10 with a rationale, then rank
            (open-ended questions first generate candidates: Step-4-Pre)
    Step-5  verification from the perception (pixel grounding) and the cognition
            (commonsense) perspectives; on failure the rationale is recorded and
            Step-4 is re-executed to reselect the answer.

The engine is backend-agnostic: it runs with the MotionEpic video MLLM (native
video + STSG encoders) or with any API model through :class:`~vot.backends.base.VideoLLM`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence, Tuple, Union

from ..backends.base import GenerationConfig, LLMResponse, Message, STSGPart, TextPart, VideoLLM, VideoPart
from ..grounding.base import STSGProvider
from ..stsg.expression import STSG_FORMAT_SPEC, parse_stsg
from ..stsg.matching import ground_targets
from ..stsg.ops import align_to_reference, neighbor_scene
from ..stsg.schema import STSG
from ..video.io import VideoClip, load_video
from .config import VoTConfig
from .parsing import parse_open_candidates, parse_ranking, parse_score, parse_targets, parse_verdict
from .prompts import OPTION_LETTERS, PromptTemplates, format_candidate, format_question, format_targets, truncate, wrap
from .result import ReasoningResult, StepRecord

logger = logging.getLogger(__name__)

VideoInput = Union[str, VideoClip, Sequence]


@dataclass
class _Run:
    """Mutable per-instance state."""

    system: Message
    clip: VideoClip
    question: str
    candidates: List[str]
    multi_choice: bool
    trace: List[StepRecord] = field(default_factory=list)
    round: int = 0


class VideoOfThought:
    """Run VoT on (video, question[, candidates]) instances.

    Example::

        from vot import VideoOfThought, build_backend
        vot = VideoOfThought(build_backend("openai:gpt-4o"))
        res = vot("demo.mp4", "What is the relationship between the white truck and this neighborhood?",
                  ["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"])
        print(res.pretty())
    """

    def __init__(
        self,
        model: VideoLLM,
        config: Optional[VoTConfig] = None,
        stsg_provider: Optional[STSGProvider] = None,
        prompts: Optional[PromptTemplates] = None,
    ) -> None:
        self.model = model
        self.cfg = config or VoTConfig()
        self.cfg.validate()
        self.stsg_provider = stsg_provider
        self.P = prompts or PromptTemplates()

    # ------------------------------------------------------------------ utils
    def load_clip(self, video: VideoInput, start: Optional[float] = None, end: Optional[float] = None) -> VideoClip:
        cfg = self.cfg
        if isinstance(video, VideoClip):
            clip = video
            if start is not None or end is not None:
                lo, hi = start or 0.0, end if end is not None else float("inf")
                keep = [i for i, t in enumerate(clip.timestamps) if lo <= t < hi] or list(range(len(clip)))
                clip = VideoClip([clip.frames[i] for i in keep], [clip.timestamps[i] for i in keep], clip.source,
                                 clip.native_fps, clip.duration, dict(clip.meta))
            if cfg.max_frames:
                clip = clip.subsample(cfg.max_frames)
        else:
            clip = load_video(video, fps=cfg.fps, max_frames=cfg.max_frames, start=start, end=end, frames_fps=cfg.frames_fps)
        # frame k of every prompt must be the k-th image the model receives
        cap = self.model.capabilities.max_images
        if cap and len(clip) > cap:
            clip = clip.subsample(cap)
        if cfg.frame_max_side and not self.model.capabilities.native_stsg:
            clip = clip.resized(cfg.frame_max_side)
        return clip

    def _gen(self, max_new_tokens: Optional[int] = None) -> GenerationConfig:
        g = self.cfg.gen
        return replace(g, max_new_tokens=max_new_tokens) if max_new_tokens else g

    def _call(self, run: _Run, step: str, messages: List[Message], parse=None, gen: Optional[GenerationConfig] = None) -> Tuple[LLMResponse, object]:
        r = self.model.generate(messages, gen or self._gen())
        parsed = parse(r.text) if parse else None
        run.trace.append(StepRecord.from_call(step, run.round, messages, r, parsed if _jsonable(parsed) else str(parsed)))
        return r, parsed

    def _video_parts(self, clip: VideoClip) -> list:
        return [TextPart("<Video>\n"), VideoPart(clip), TextPart("\n</Video>\n")]

    def _stsg_spec(self, clip: VideoClip) -> str:
        if self.model.capabilities.knows_stsg_format:
            return ""
        return f"\nThe video is given as {len(clip)} sampled frames, numbered 1 to {len(clip)}.\n{STSG_FORMAT_SPEC}"

    # ------------------------------------------------------------------ steps
    def step1_targets(self, run: _Run) -> Tuple[List[str], List[Message]]:
        q = format_question(run.question, run.candidates if run.multi_choice else None)
        text = self.P.step1.format(question=q) + self.P.hint(self.P.step1_hint)
        msgs = [run.system, Message.user(*self._video_parts(run.clip), wrap("Question", text))]
        r, targets = self._call(run, "step1_targets", msgs, lambda t: parse_targets(t, self.cfg.max_targets))
        return list(targets or []), msgs + [Message.assistant(r.text)]

    def step2_tracking(
        self, run: _Run, targets: List[str], chain: List[Message], full: Optional[STSG]
    ) -> Tuple[STSG, List[Message]]:
        mode = self.cfg.step2_mode
        if mode == "auto":
            mode = "select" if full is not None and not full.is_empty else "generate"
        if mode in {"select", "match"} and (full is None or full.is_empty):
            logger.warning("step2_mode=%s needs an STSG; falling back to 'generate'", mode)
            mode = "generate"

        if mode == "match":
            partial = ground_targets(targets, full, hops=0)  # type: ignore[arg-type]
            run.trace.append(StepRecord("step2_tracking[match]", run.round, f"match {targets}", partial.to_expression(), partial.to_dict()))
            if not partial.is_empty:
                return partial, chain
            mode = "select"

        tgt = format_targets(targets)
        text = self.P.step2.format(targets=tgt)
        parts: list = []
        if mode == "select":
            parts.append(STSGPart(full, label="STSG", max_frames=self.cfg.stsg_max_frames))  # type: ignore[arg-type]
            parts.append(TextPart("\n"))
            text += " " + self.P.step2_with_reference
        text += self._stsg_spec(run.clip) + self.P.hint(self.P.step2_hint, targets=tgt)
        user = Message.user(*parts, wrap("Question", text))
        if self.cfg.context == "chain":
            msgs = chain + [user]
        else:
            msgs = [run.system, Message.user(*self._video_parts(run.clip), *user.content)]
        parse = lambda t: parse_stsg(t, image_size=run.clip.size)  # noqa: E731
        r, partial = self._call(run, f"step2_tracking[{mode}]", msgs, parse, self._gen(self.cfg.step2_max_new_tokens))
        partial = partial if isinstance(partial, STSG) else STSG()
        valid = [i for i in partial.frame_indices() if 1 <= i <= len(run.clip)]
        if len(valid) != len(partial.frames):  # drop frames the model invented
            partial = partial.select_frames(valid)
        for f in partial.frames:
            f.timestamp = run.clip.timestamps[f.index - 1]
        run.trace[-1].parsed = partial.to_dict()
        if full is not None and not full.is_empty:
            if partial.is_empty:
                partial = ground_targets(targets, full, hops=0)
            elif mode == "select" and self.cfg.snap_boxes_to_reference:
                partial = align_to_reference(partial, full)
        partial.meta.setdefault("source", f"step2:{mode}")
        return partial, msgs + [Message.assistant(r.text)]

    def step3_action(self, run: _Run, targets: List[str], graph: STSG, chain: List[Message]) -> str:
        text = self.P.step3.format(targets=format_targets(targets)) + self.P.hint(self.P.step3_hint)
        parts = [TextPart(wrap("Question", text) + "\n"), STSGPart(graph, label="STSG", max_frames=self.cfg.stsg_max_frames)]
        if self.cfg.context == "chain":
            msgs = chain + [Message.user(*parts)]
        else:
            pre = self._video_parts(run.clip) if self.cfg.step3_with_video else []
            msgs = [run.system, Message.user(*pre, *parts)]
        r, _ = self._call(run, "step3_action", msgs)
        return r.text.strip()

    def step4_pre(self, run: _Run, observation: str) -> Tuple[List[str], List[Optional[float]], List[str]]:
        k = self.cfg.num_open_candidates
        text = self.P.step4_pre.format(question=run.question, observation=observation, k=k) + self.P.hint(self.P.step4_pre_hint)
        msgs = [run.system, Message.user(wrap("Question", text))]
        r, cands = self._call(run, "step4_pre", msgs, lambda t: parse_open_candidates(t, k))
        run.trace[-1].parsed = [c.__dict__ for c in cands]
        if not cands:
            line = next((l.strip() for l in r.text.splitlines() if l.strip()), r.text.strip())
            return [line[:300]], [None], [r.text]
        return [c.text for c in cands], [c.score for c in cands], [c.rationale for c in cands]

    def step4_rank(
        self,
        run: _Run,
        candidates: List[str],
        observation: str,
        question_display: str,
        rejected: Sequence[int],
        feedback: Sequence[Tuple[str, str]],
        prior_scores: Optional[List[Optional[float]]] = None,
        prior_rationales: Optional[List[str]] = None,
        rescore: bool = True,
    ) -> Tuple[Union[int, List[int]], List[Optional[float]], List[str]]:
        n = len(candidates)
        if rescore or prior_scores is None:
            batch = []
            for i in range(n):
                cand = format_candidate(candidates, i)
                text = self.P.step4_score.format(question=question_display, candidate=cand, observation=observation)
                for ans, reason in feedback:
                    text += "\n" + self.P.step4_feedback.format(answer=ans, reason=reason)
                text += self.P.hint(self.P.step4_score_hint, candidate=cand)
                batch.append([run.system, Message.user(wrap(f"Question for Answer {OPTION_LETTERS[i]}", text))])
            rs = self.model.generate_many(batch, self._gen(), max_workers=self.cfg.parallel_calls)
            scores, rationales = [], []
            for i, (msgs, r) in enumerate(zip(batch, rs)):
                s = parse_score(r.text)
                scores.append(s)
                rationales.append(r.text.strip())
                run.trace.append(StepRecord.from_call(f"step4_score[{OPTION_LETTERS[i]}]", run.round, msgs, r, s))
        else:
            scores, rationales = list(prior_scores), list(prior_rationales or [""] * n)

        allowed = [i for i in range(n) if i not in rejected] or list(range(n))
        if self.cfg.multi_answer:
            chosen = [i for i in allowed if (scores[i] or 0) >= self.cfg.multi_answer_threshold]
            if not chosen:
                chosen = [max(allowed, key=lambda i: (scores[i] or 0, -i))]
            return chosen, scores, rationales

        known = [(scores[i], i) for i in allowed if scores[i] is not None]
        top = max((s for s, _ in known), default=None)
        tied = [i for s, i in known if s == top]
        need_model = self.cfg.ranking == "model" or (self.cfg.ranking == "auto" and (len(tied) != 1 or len(known) < len(allowed)))
        model_pick: Optional[int] = None
        if need_model and n > 1:
            items = ", ".join(
                self.P.step4_rank_item.format(candidate=format_candidate(candidates, i), score=_fmt_score(scores[i]))
                for i in range(n)
            )
            text = self.P.step4_rank.format(score_list=items)
            if rejected:
                text += " " + "; ".join(f"[{format_candidate(candidates, j)}] has been rejected by verification" for j in rejected) + "."
            text += self.P.hint(self.P.step4_rank_hint)
            msgs = [run.system, Message.user(wrap("Question for Ranking", text))]
            r, (order, final) = self._call(run, "step4_rank", msgs, lambda t: parse_ranking(t, candidates))
            ranked = ([final] if final is not None else []) + [i for i in order if i != final]
            model_pick = next((i for i in ranked if i in allowed), None)

        # "We then rank the scores of all options and select the most optimal answer":
        # a unique score maximum is authoritative; the ranking prompt breaks ties
        # and covers unparsable scores.
        if len(tied) == 1 and len(known) == len(allowed):
            choice = tied[0]
        elif model_pick is not None and (not tied or model_pick in tied or len(known) < len(allowed)):
            choice = model_pick
        elif tied:
            choice = tied[0]
        else:
            choice = model_pick if model_pick is not None else allowed[0]
        return choice, scores, rationales

    def step5_verify(self, run: _Run, answer: str, observation: str, graph: STSG) -> Tuple[Optional[bool], str]:
        q = format_question(run.question, run.candidates if run.multi_choice else None)
        items = []
        if self.cfg.verify_perception:
            items.append(self.P.step5_perception.format(answer=answer))
        if self.cfg.verify_cognition:
            items.append(self.P.step5_cognition.format(answer=answer, observation=observation))
        body = self.P.step5_head.format(question=q) + "\n" + "\n".join(f"  {k}) {it}" for k, it in enumerate(items, 1))
        body += "\n" + self.P.step5_tail + self.P.hint(self.P.step5_hint)
        parts: list = list(self._video_parts(run.clip))
        if self.cfg.verify_perception and not graph.is_empty:
            parts += [STSGPart(graph, label="STSG", max_frames=self.cfg.stsg_max_frames), TextPart("\n")]
        msgs = [run.system, Message.user(*parts, wrap("Question", body))]
        r, verdict = self._call(run, "step5_verify", msgs, parse_verdict)
        return verdict, r.text.strip()

    # -------------------------------------------------------------- pipeline
    def __call__(self, video: VideoInput, question: str, candidates: Optional[Sequence[str]] = None, **kw) -> ReasoningResult:
        return self.run(video, question, candidates, **kw)

    def run(
        self,
        video: VideoInput,
        question: str,
        candidates: Optional[Sequence[str]] = None,
        *,
        stsg: Optional[STSG] = None,
        video_id: Optional[str] = None,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> ReasoningResult:
        """Answer ``question`` about ``video``.

        Args:
            video: path, frame directory, list of PIL images or a :class:`VideoClip`.
            candidates: options of a multiple-choice question; ``None`` for open-ended QA.
            stsg: an STSG of the sampled clip (overrides ``stsg_provider``).
            video_id: key for :class:`~vot.grounding.base.StaticSTSGProvider`.
            start, end: question segment in seconds.
        """
        cfg = self.cfg
        clip = self.load_clip(video, start, end)
        cands = [str(c) for c in candidates] if candidates else []
        mc = bool(cands)
        if len(cands) > len(OPTION_LETTERS):
            raise ValueError("too many candidates")
        run = _Run(Message.system(self.P.task_definition(mc, cfg.multi_answer)), clip, question.strip(), cands, mc)

        # Step-1
        targets, chain = self.step1_targets(run)

        # Step-2
        full = stsg
        if full is None and self.stsg_provider is not None and cfg.step2_mode != "generate":
            full = self.stsg_provider(clip, targets=targets, question=question, video_id=video_id)
        partial, chain = self.step2_tracking(run, targets, chain, full)

        # Step-3
        graph = neighbor_scene(partial, full, cfg.neighbor_hops)
        observation = self.step3_action(run, targets, graph, chain)
        obs = truncate(observation, cfg.observation_max_chars) if cfg.observation_max_chars else " ".join(observation.split())

        # Step-4 (+Pre for open-ended)
        prior_s: Optional[List[Optional[float]]] = None
        prior_r: Optional[List[str]] = None
        if mc:
            options, q_disp = cands, format_question(run.question, cands)
        else:
            options, prior_s, prior_r = self.step4_pre(run, obs)
            q_disp = run.question
        rescore = mc or cfg.rescore_open_candidates or prior_s is None or any(s is None for s in prior_s)

        rejected: List[int] = []
        feedback: List[Tuple[str, str]] = []
        verified: Optional[bool] = None
        vtext = ""
        choice, scores, rationales = self.step4_rank(run, options, obs, q_disp, rejected, feedback, prior_s, prior_r, rescore)
        first = (choice, scores, rationales)

        # Step-5; on failure record the rationale and re-execute Step-4 to reselect.
        do_verify = cfg.verify and (cfg.verify_perception or cfg.verify_cognition)
        while do_verify:
            answer = _answer_str(options, choice, mc)
            verdict, vtext = self.step5_verify(run, answer, obs, graph)
            if verdict is not False:
                verified = verdict
                break
            feedback.append((answer, truncate(vtext, cfg.feedback_max_chars)))
            if not cfg.multi_answer:
                rejected.append(choice)  # type: ignore[arg-type]
            if run.round >= cfg.max_verify_rounds or len(rejected) >= len(options):
                # No candidate survived verification: keep the top-ranked answer of the
                # first round (the best evidence-based choice) and flag it as unverified.
                choice, scores, rationales = first
                verified = False
                break
            run.round += 1
            prev = choice
            choice, scores, rationales = self.step4_rank(
                run, options, obs, q_disp, rejected, feedback, scores, rationales, cfg.rescore_on_reject
            )
            if cfg.multi_answer and choice == prev:  # re-execution did not change the selection
                verified = False
                break

        answer_text = _answer_str(options, choice, mc, with_letter=False)
        usage = {
            "calls": sum(1 for s in run.trace if "[match]" not in s.step),
            "prompt_tokens": sum(s.prompt_tokens for s in run.trace),
            "completion_tokens": sum(s.completion_tokens for s in run.trace),
        }
        return ReasoningResult(
            method="vot",
            question=run.question,
            candidates=options,
            answer_index=choice,
            answer_text=answer_text,
            multi_choice=mc,
            targets=targets,
            partial_stsg=partial,
            observation=observation,
            scores=scores,
            rationales=rationales,
            verified=verified,
            verification=vtext,
            rejected=rejected,
            rounds=run.round,
            generated_candidates=not mc,
            trace=run.trace,
            usage=usage,
            meta={"backend": self.model.describe(), "num_frames": len(clip), "timestamps": [round(t, 3) for t in clip.timestamps]},
        )


def _fmt_score(s: Optional[float]) -> str:
    if s is None:
        return "unknown"
    return str(int(s)) if float(s).is_integer() else f"{s:.1f}"


def _answer_str(options: List[str], choice: Union[int, List[int]], mc: bool, with_letter: bool = True) -> str:
    idx = choice if isinstance(choice, list) else [choice]
    if with_letter and mc:
        return "; ".join(format_candidate(options, i) for i in idx)
    return "; ".join(options[i] for i in idx)


def _jsonable(x: object) -> bool:
    return x is None or isinstance(x, (str, int, float, bool, list, dict, tuple))
