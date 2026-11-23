import math

from vot.data.videoqa import VideoQASample
from vot.eval.metrics import (
    accuracy,
    coerce_choice,
    coerce_choices,
    evaluate,
    exact_match,
    format_report,
    make_llm_judge,
    normalize_answer,
    parse_judge_output,
    token_f1,
)

CANDS = ["a dog", "the cat", "Two birds", "A"]


def mc(qid, ans, cat=None, bench=None, n=4, meta=None):
    return VideoQASample(qid, "v", "q?", [f"opt{i}" for i in range(n)], ans, cat, benchmark=bench, meta=meta or {})


def test_coerce_choice():
    assert coerce_choice(2, CANDS) == 2
    assert coerce_choice("2", CANDS) == 2
    assert coerce_choice(9, CANDS) is None
    assert coerce_choice("B", CANDS) == 1
    assert coerce_choice("C. Two birds", CANDS) == 2
    assert coerce_choice("(b) the cat", CANDS) == 1
    assert coerce_choice("Option D", CANDS) == 3
    assert coerce_choice("the cat", CANDS) == 1
    assert coerce_choice("A", CANDS) == 3  # exact option text beats the letter
    assert coerce_choice("a dog", CANDS) == 0  # not parsed as letter "A"
    assert coerce_choice("So, the final answer is [C].", CANDS) == 2
    assert coerce_choice("I think it is two birds flying", CANDS) == 2
    assert coerce_choice("nonsense", CANDS) is None
    assert coerce_choice(None, CANDS) is None and coerce_choice(True, CANDS) is None
    xyz = ["x", "y", "z"]
    assert coerce_choices(["A", "C"], xyz) == [0, 2]
    assert coerce_choices("C, A", xyz) == [0, 2]
    assert coerce_choices(["A", "?"], xyz) is None


def test_normalization():
    assert normalize_answer("The Two dogs!") == "2 dogs"
    assert exact_match("two dogs", "2 dogs.")
    assert not exact_match("dogs", "2 dogs")
    assert math.isclose(token_f1("a red car", "red truck"), 0.5)
    assert token_f1("", "") == 1.0


def test_accuracy_and_categories():
    samples = [mc("1", 0, "x"), mc("2", 1, "x"), mc("3", 2, "y"), mc("4", None, "y")]
    preds = [{"qid": "1", "pred": 0}, {"qid": "2", "pred": "A"}, {"qid": "3", "pred": "C"}]
    assert accuracy(samples, preds) == 100 * 2 / 3
    rep = evaluate(samples, {"1": 0, "2": 1})
    assert rep["n"] == 3 and rep["missing"] == 1
    assert rep["metrics"]["Acc"] == 100 * 2 / 3
    assert rep["per_category"]["x"]["acc"] == 100.0 and rep["per_category"]["y"]["acc"] == 0.0


def test_multi_answer():
    s = VideoQASample("m", "v", "q", ["a", "b", "c"], [0, 2])
    assert accuracy([s], {"m": [2, 0]}) == 100.0
    assert accuracy([s], {"m": "A, C"}) == 100.0
    assert accuracy([s], {"m": [0]}) == 0.0


def test_nextqa_report():
    samples = [mc("1", 0, "CW", "nextqa"), mc("2", 0, "TN", "nextqa"), mc("3", 0, "DO", "nextqa"), mc("4", 0, "CH", "nextqa")]
    rep = evaluate(samples, {"1": 0, "2": 1, "3": 0, "4": 1})
    m = rep["metrics"]
    assert list(m) == ["Acc@All", "Acc@C", "Acc@T", "Acc@D"]
    assert (m["Acc@All"], m["Acc@C"], m["Acc@T"], m["Acc@D"]) == (50.0, 50.0, 0.0, 100.0)


def test_star_report():
    samples = [mc("Interaction_1", 0, "Interaction", "star"), mc("Sequence_1", 0, "Sequence", "star"),
               mc("Prediction_1", 0, "Prediction", "star"), mc("Feasibility_1", 0, "Feasibility", "star")]
    m = evaluate(samples, {"Interaction_1": 0, "Sequence_1": 0, "Prediction_1": 1, "Feasibility_1": 0})["metrics"]
    assert (m["Int."], m["Seq."], m["Pre."], m["Fea."], m["Mean"]) == (100.0, 100.0, 0.0, 100.0, 75.0)


def test_causalvidqa_report():
    def s(qid, qtype, part, group):
        return mc(qid, 0, qtype[0].upper(), "causalvidqa", meta={"qtype": qtype, "part": part, "group": group})

    samples = [
        s("d1", "descriptive", "answer", "d1"), s("e1", "explanatory", "answer", "e1"),
        s("p1a", "predictive", "answer", "p1"), s("p1r", "predictive", "reason", "p1"),
        s("p2a", "predictive", "answer", "p2"), s("p2r", "predictive", "reason", "p2"),
        s("c1a", "counterfactual", "answer", "c1"), s("c1r", "counterfactual", "reason", "c1"),
    ]
    preds = {"d1": 0, "e1": 1, "p1a": 0, "p1r": 0, "p2a": 0, "p2r": 1, "c1a": 1, "c1r": 0}
    m = evaluate(samples, preds)["metrics"]
    assert m["Acc@D"] == 100.0 and m["Acc@E"] == 0.0
    assert (m["Acc@P-A"], m["Acc@P-R"], m["Acc@P-AR"]) == (100.0, 50.0, 50.0)
    assert (m["Acc@C-A"], m["Acc@C-R"], m["Acc@C-AR"]) == (0.0, 100.0, 0.0)
    assert m["Acc@All"] == (100.0 + 0.0 + 50.0 + 0.0) / 4


def test_socialiq_report():
    samples = [mc("a", 0, "2-way", "socialiq", n=2), mc("b", 1, "2-way", "socialiq", n=2), mc("c", 3, "4-way", "socialiq")]
    m = evaluate(samples, {"a": 0, "b": 0, "c": 3})["metrics"]
    assert m == {"2-way": 50.0, "4-way": 100.0}


def test_open_ended_report_and_judge():
    samples = [
        VideoQASample("1", "v", "what is he doing", answer="cooking", category="what", benchmark="msrvtt_qa"),
        VideoQASample("2", "v", "how many dogs", answer="2", category="how", benchmark="msrvtt_qa"),
    ]
    preds = {"1": "Cooking.", "2": "2 dogs"}
    rep = evaluate(samples, preds)
    assert rep["metrics"]["Acc"] == 50.0 and rep["metrics"]["EM"] == 50.0
    assert 50.0 < rep["metrics"]["F1"] < 100.0

    def fake_llm(system, user):
        assert "Reference answer" in user
        return "{'pred': 'yes', 'score': 4}" if "Cooking" in user else "{'pred': 'no', 'score': 1}"

    judge = make_llm_judge(fake_llm)
    rep = evaluate(samples, preds, judge=judge)
    assert rep["metrics"]["Acc"] == 50.0 and rep["metrics"]["Score"] == 2.5
    assert parse_judge_output("Sure! {'pred': 'Yes', 'score': 5}") == (True, 5.0)
    assert parse_judge_output('"pred": "no", "score": 2') == (False, 2.0)


def test_format_report():
    samples = [mc("1", 0, "CW", "nextqa"), mc("2", 1, "TN", "nextqa")]
    text = format_report(evaluate(samples, {"1": 0}))
    assert "| Acc@All | Acc@C | Acc@T | Acc@D |" in text
    assert "| 50.0 | 100.0 | 0.0 | - |" in text
    assert "| CW | 1 | 100.0 |" in text
