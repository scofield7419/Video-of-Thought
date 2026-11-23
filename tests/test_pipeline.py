import json
import re

import pytest

from conftest import TRUCK_OPTS, TRUCK_Q
from vot import DirectAnswer, VanillaCoT, VideoOfThought, VoTConfig
from vot.backends.base import BackendCapabilities, STSGPart, VideoPart
from vot.backends.cache import CachedLLM
from vot.backends.mock import RuleBasedMockLLM, ScriptedLLM


@pytest.mark.parametrize("context", ["chain", "isolated"])
@pytest.mark.parametrize("mode", ["auto", "generate", "select", "match"])
def test_vot_multichoice(truck_stsg, clip3, context, mode):
    m = RuleBasedMockLLM(stsg=truck_stsg, answer_key="Clean")
    vot = VideoOfThought(m, VoTConfig(context=context, step2_mode=mode))
    r = vot(clip3, TRUCK_Q, TRUCK_OPTS, stsg=None if mode == "generate" else truck_stsg)
    assert r.answer_index == 2 and r.answer_text == "Clean Services"
    assert r.verified is True and r.rounds == 0
    assert r.targets and not r.partial_stsg.is_empty
    steps = [s.step for s in r.trace]
    assert steps[0] == "step1_targets" and steps[1].startswith("step2_tracking") and "step3_action" in steps
    assert sum(s.startswith("step4_score") for s in steps) == 4 and steps[-1] == "step5_verify"
    json.dumps(r.to_dict())


def test_verification_failure_reexecutes_step4(truck_stsg, clip3):
    m = RuleBasedMockLLM(stsg=truck_stsg, answer_key="Clean", reject_first=True)
    r = VideoOfThought(m)(clip3, TRUCK_Q, TRUCK_OPTS, stsg=truck_stsg)
    assert r.rejected == [2] and r.rounds == 1 and r.answer_index != 2 and r.verified is True
    second = [s for s in r.trace if s.step.startswith("step4_score") and s.round == 1]
    assert len(second) == 4 and "rejected by verification" in second[0].prompt


def test_verification_budget_exhausted(truck_stsg, clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "verify the previous answer" in u:
            return "It contradicts the observations.\nVerdict: INCORRECT"
        return RuleBasedMockLLM(stsg=truck_stsg, answer_key="Clean").generate(msgs).text

    r = VideoOfThought(ScriptedLLM(fn), VoTConfig(max_verify_rounds=1))(clip3, TRUCK_Q, TRUCK_OPTS, stsg=truck_stsg)
    assert len(r.rejected) == 2 and r.rounds == 1
    # nothing survived verification: keep the round-0 top answer, flagged as unverified
    assert r.verified is False and r.answer_index == 2 and r.scores[2] == 9


def test_unique_score_beats_model_ranking(clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "score the rationality" in u:
            s = 8 if "[B. Buildings]" in u else 4
            return f"the rationality of the answer [x] is {s}."
        if "rank the candidate" in u:
            return "So, the final answer is [A. Transportation]."
        if "verify" in u:
            return "Verdict: CORRECT"
        return "The involved targets are [truck]."

    r = VideoOfThought(ScriptedLLM(fn))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.answer_index == 1


def test_ties_broken_by_model_ranking(clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "score the rationality" in u:
            return "the rationality of the answer [x] is 7." if ("[C." in u or "[D." in u) else "the rationality of the answer [x] is 2."
        if "rank the candidate" in u:
            return "ranking results in descending order is [D] [C] [A] [B]. So, the final answer is [D. Entertainment Facilities]."
        if "verify" in u:
            return "Verdict: CORRECT"
        return "The involved targets are [truck]."

    sl = ScriptedLLM(fn)
    r = VideoOfThought(sl, VoTConfig(ranking="auto"))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.answer_index == 3
    r2 = VideoOfThought(sl, VoTConfig(ranking="score"))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r2.answer_index == 2 and not any(s.step == "step4_rank" for s in r2.trace)


def test_open_ended(truck_stsg, clip3):
    r = VideoOfThought(RuleBasedMockLLM(stsg=truck_stsg))(clip3, "What are the people doing?", stsg=truck_stsg)
    assert not r.multi_choice and r.generated_candidates and len(r.candidates) >= 2
    assert r.prediction == r.answer_text and r.answer_text in r.candidates
    assert any(s.step == "step4_pre" for s in r.trace)


def test_multi_answer(truck_stsg, clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "score the rationality" in u:
            return "the rationality of the answer [x] is 9." if ("[A." in u or "[C." in u) else "the rationality of the answer [x] is 2."
        if "verify" in u:
            return "Verdict: CORRECT"
        return "The involved targets are [truck]."

    sl = ScriptedLLM(fn)
    r = VideoOfThought(sl, VoTConfig(multi_answer=True))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.answer_index == [0, 2]
    assert "all possible correct option(s)" in sl.calls[0][0].text()


def test_prompts_follow_paper_and_modalities(truck_stsg, clip3):
    sl = ScriptedLLM(lambda msgs: RuleBasedMockLLM(stsg=truck_stsg, answer_key="Clean").generate(msgs).text)
    VideoOfThought(sl, VoTConfig(context="isolated"))(clip3, TRUCK_Q, TRUCK_OPTS, stsg=truck_stsg)
    by_step = {}
    for msgs in sl.calls:
        u = msgs[-1]
        t = u.text()
        key = next(k for k in ["possible targets", "Provide the tracklet", "analyze the motion", "score the rationality",
                                "rank the candidate", "verify the previous answer"] if k in t)
        by_step.setdefault(key, u)
    sys = sl.calls[0][0].text()
    assert sys.startswith("Now you are an expert in analyzing video data") and "the most suitable option" in sys
    assert "Given the question: [What is the relationship between the white truck and this neighborhood? A. Transportation  B. Buildings" in by_step["possible targets"].text()
    assert by_step["possible targets"].has_video()
    assert by_step["Provide the tracklet"].has_video()
    assert not by_step["analyze the motion"].has_video() and any(isinstance(p, STSGPart) for p in by_step["analyze the motion"].content)
    assert "Combining all possible related commonsense" in by_step["analyze the motion"].text()
    assert not by_step["score the rationality"].has_video()
    v = by_step["verify the previous answer"]
    assert v.has_video() and any(isinstance(p, STSGPart) for p in v.content)
    assert "1) checking the pixel grounding information" in v.text() and "2) determining from a cognition perspective" in v.text()
    # STSG grammar is given to models not trained on it
    assert "Temporal Coreference Edge" in by_step["Provide the tracklet"].text()


@pytest.mark.parametrize("perc,cog", [(True, False), (False, True)])
def test_verification_ablation(truck_stsg, clip3, perc, cog):
    sl = ScriptedLLM(lambda msgs: RuleBasedMockLLM(stsg=truck_stsg).generate(msgs).text)
    VideoOfThought(sl, VoTConfig(verify_perception=perc, verify_cognition=cog))(clip3, TRUCK_Q, TRUCK_OPTS, stsg=truck_stsg)
    v = [m[-1] for m in sl.calls if "verify the previous answer" in m[-1].text()][0]
    assert ("pixel grounding" in v.text()) == perc and ("cognition perspective" in v.text()) == cog
    assert any(isinstance(p, STSGPart) for p in v.content) == perc


def test_no_verify(truck_stsg, clip3):
    r = VideoOfThought(RuleBasedMockLLM(stsg=truck_stsg), VoTConfig(verify=False))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.verified is None and not any(s.step == "step5_verify" for s in r.trace)


def test_native_stsg_backend_gets_no_format_spec(truck_stsg, clip3):
    caps = BackendCapabilities(native_stsg=True, knows_stsg_format=True)
    sl = ScriptedLLM(lambda msgs: RuleBasedMockLLM(stsg=truck_stsg).generate(msgs).text, capabilities=caps)
    VideoOfThought(sl)(clip3, TRUCK_Q, TRUCK_OPTS)
    t = [m[-1].text() for m in sl.calls if "Provide the tracklet" in m[-1].text()][0]
    assert "Temporal Coreference Edge" not in t


def test_baselines(truck_stsg, clip3):
    m = RuleBasedMockLLM(stsg=truck_stsg)
    assert DirectAnswer(m)(clip3, TRUCK_Q, TRUCK_OPTS).answer_index == 0
    r = VanillaCoT(m)(clip3, TRUCK_Q, TRUCK_OPTS, stsg=truck_stsg)
    assert r.answer_index == 0 and r.meta["with_stsg"] and "step by step" in r.trace[0].prompt


def test_cache(tmp_path, truck_stsg, clip3):
    inner = RuleBasedMockLLM(stsg=truck_stsg, answer_key="Clean")
    c = CachedLLM(inner, str(tmp_path / "c.sqlite"))
    r1 = VideoOfThought(c)(clip3, TRUCK_Q, TRUCK_OPTS)
    calls = inner.usage.calls
    r2 = VideoOfThought(CachedLLM(inner, str(tmp_path / "c.sqlite")))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert inner.usage.calls == calls and r1.answer_index == r2.answer_index


def test_trace_render_has_placeholders(truck_stsg, clip3):
    r = VideoOfThought(RuleBasedMockLLM(stsg=truck_stsg))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert re.search(r"<video: 3 frames", r.trace[0].prompt)


def test_step2_drops_invented_frames(clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "Provide the tracklet" in u:
            return "\n".join(f'Frame {k}: {{Objects: [{{"truck": [{k}, 40, 30, 25]}}], Triplets: []}}' for k in (1, 2, 9))
        if "verify" in u:
            return "Verdict: CORRECT"
        if "score the rationality" in u:
            return "the rationality of the answer [x] is 5."
        return "The involved targets are [the truck]."

    r = VideoOfThought(ScriptedLLM(fn), VoTConfig(step2_mode="generate"))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.partial_stsg.frame_indices() == [1, 2]
    assert r.partial_stsg.frame(2).timestamp == clip3.timestamps[1]


def test_bold_verdict_is_respected(clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "verify the previous answer" in u:
            return "There is no contradiction... wait, the grounding shows otherwise.\n**Verdict**: INCORRECT"
        if "score the rationality" in u:
            return "the rationality of the answer [x] is 9." if "[A." in u else "the rationality of the answer [x] is 3."
        return "The involved targets are [truck]."

    r = VideoOfThought(ScriptedLLM(fn), VoTConfig(max_verify_rounds=1))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.rejected and r.rejected[0] == 0 and r.verified is False


def test_multi_answer_stops_when_selection_unchanged(clip3):
    def fn(msgs):
        u = msgs[-1].text()
        if "verify the previous answer" in u:
            return "Verdict: INCORRECT"
        if "score the rationality" in u:
            return "the rationality of the answer [x] is 9." if "[A." in u else "the rationality of the answer [x] is 2."
        return "The involved targets are [truck]."

    sl = ScriptedLLM(fn)
    r = VideoOfThought(sl, VoTConfig(multi_answer=True, max_verify_rounds=3, rescore_on_reject=False))(clip3, TRUCK_Q, TRUCK_OPTS)
    assert r.answer_index == [0] and r.verified is False
    assert sum("verify the previous answer" in m[-1].text() for m in sl.calls) == 1


def test_clip_respects_backend_image_cap_and_segment(clip3):
    from vot.video import VideoClip
    from PIL import Image

    long = VideoClip([Image.new("RGB", (8, 8))] * 10, [float(i) for i in range(10)])
    sl = ScriptedLLM(lambda m: "x", capabilities=BackendCapabilities(max_images=4))
    vot = VideoOfThought(sl, VoTConfig(max_frames=8))
    assert len(vot.load_clip(long)) == 4
    seg = vot.load_clip(long, start=2.0, end=5.0)
    assert all(2.0 <= t < 5.0 for t in seg.timestamps) and len(seg) == 3
