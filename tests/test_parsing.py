from vot.reasoning.parsing import (
    parse_choice,
    parse_open_candidates,
    parse_ranking,
    parse_score,
    parse_targets,
    parse_verdict,
)

OPTS = ["Transportation", "Buildings", "Clean Services", "Entertainment Facilities"]


def test_targets():
    assert parse_targets("The involved targets are [the white truck], [the neighborhood].") == ["the white truck", "the neighborhood"]
    assert parse_targets("Targets: the dog, the trainer and the ball") == ["the dog", "the trainer", "the ball"]
    assert parse_targets("") == []
    assert len(parse_targets("The involved targets are " + ", ".join(f"[t{i}]" for i in range(10)), max_targets=3)) == 3


def test_score():
    assert parse_score("Based on the observations and commonsense, the rationality of the answer [A. Transportation] is 5. While ...") == 5
    assert parse_score("the rationality of the answer [B. 3 people] is 8.") == 8  # digits inside the candidate are ignored
    assert parse_score("Score: 7/10. The rationale ...") == 7
    assert parse_score("I would rate it 6 out of 10") == 6
    assert parse_score("no number here") is None
    assert parse_score("rationality is 42") is None


def test_choice_and_ranking():
    out = ("Based on the rationality, the ranking results in descending order is [C. Clean Service] [A. Transportation] "
           "[D. Entertainment Facilities] [B. Buildings]. So, the final answer is [C. Clean Service].")
    order, final = parse_ranking(out, OPTS)
    assert order == [2, 0, 3, 1] and final == 2
    assert parse_choice("Final answer: B", OPTS) == 1
    assert parse_choice("(D)", OPTS) == 3
    assert parse_choice("I think the answer is clean services.", OPTS) == 2
    assert parse_choice("blah", OPTS) is None


def test_open_candidates():
    txt = ("Answer A: garbage collection | Score: 9 | Rationale: men pour trash into the truck\n"
           "Answer B: moving house | Score: 3 | Rationale: no furniture\n"
           "Answer C: garbage collection | Score: 9 | Rationale: duplicate\n")
    c = parse_open_candidates(txt, 4)
    assert [x.text for x in c] == ["garbage collection", "moving house"] and c[0].score == 9 and "trash" in c[0].rationale


def test_verdict():
    assert parse_verdict("... reasoning ...\nVerdict: CORRECT") is True
    assert parse_verdict("Verdict: INCORRECT") is False
    assert parse_verdict("Overall, the answer [C. Clean Service] is correct.") is True
    assert parse_verdict("The answer contradicts the observed tracklet, so it is incorrect.") is False
    assert parse_verdict("hmm") is None


def test_review_regressions_choice():
    c = ["walk away", "walk", "a dog runs away", "run"]
    assert parse_choice("So, the final answer is [A dog runs away].", c) == 2
    assert parse_choice("A dog runs away.", c) == 2
    assert parse_choice("Final answer: a dog runs away", c) == 2
    assert parse_choice("I don't know", ["x"] * 10) is None
    assert parse_choice("**Final answer:** C", c) == 2
    from vot.reasoning.parsing import match_option

    assert match_option("walk", c) == 1


def test_review_regressions_verdict_and_scores():
    assert parse_verdict("**Verdict**: INCORRECT") is False
    assert parse_verdict("There is no contradiction with the observations. **Verdict**: CORRECT") is True
    assert parse_verdict("There is no contradiction with the observations, so the answer is correct.") is True
    assert parse_score("the rationality of the answer [A. x] is: 6") == 6
    assert parse_score("Using a score from 1-10, this answer deserves 8.") == 8
    c = parse_open_candidates("Answer A: The team scored 3 goals | Score: 8")
    assert c[0].text == "The team scored 3 goals" and c[0].score == 8
    c = parse_open_candidates("1. Answer: He is late | Score: 7 | Rationale: running\n2. Answer: He is chased | Score: 4")
    assert [(x.text, x.score) for x in c] == [("He is late", 7), ("He is chased", 4)]
    c = parse_open_candidates("Answer A: garbage collection\nScore: 9\nRationale: men pour trash\nAnswer B: moving house\nScore: 3")
    assert [(x.text, x.score) for x in c] == [("garbage collection", 9), ("moving house", 3)] and c[0].rationale == "men pour trash"
