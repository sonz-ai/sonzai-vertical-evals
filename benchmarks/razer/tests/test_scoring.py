"""Tests for the substring-guard layer + aggregation. The LLM judge is
mocked separately in test_runner.py so the deterministic scoring path
gets exercised here without an API call."""

from __future__ import annotations

from sonzai_razer_bench.dataset import QA
from sonzai_razer_bench.scoring import (
    QAResult,
    aggregate,
    check_guards,
    compute_pillars,
)


def _qa(
    qa_id: str = "x",
    must_mention: tuple = (),
    must_mention_one_of: tuple = (),
    must_not_mention: tuple = (),
    category: str = "single-hop",
) -> QA:
    return QA(
        qa_id=qa_id,
        category=category,
        ask_as_user="u1",
        ask_on_device="d1",
        question="?",
        gold_answer="...",
        evidence_sessions=(),
        must_mention=must_mention,
        must_mention_one_of=must_mention_one_of,
        must_not_mention=must_not_mention,
    )


def test_guards_pass_when_no_constraints():
    g = check_guards(_qa(), "anything goes here")
    assert g.passed and not g.failures


def test_must_mention_pass_case_insensitive():
    g = check_guards(_qa(must_mention=("Razer", "Marcus")), "marcus uses a razer mouse")
    assert g.passed


def test_must_mention_is_advisory_not_a_hard_fail():
    """must_mention is a hint, not a contract — soft failures don't sink the QA."""
    g = check_guards(_qa(must_mention=("BlackShark",)), "he uses a regular headset")
    assert g.passed                                # hard guards still pass
    assert any("BlackShark" in f for f in g.soft_failures)
    assert not g.hard_failures


def test_must_mention_one_of_pass_with_any():
    g = check_guards(
        _qa(must_mention_one_of=("Spiritfarer", "Dredge", "Strata")),
        "How about Dredge?",
    )
    assert g.passed


def test_must_mention_one_of_is_advisory_too():
    g = check_guards(
        _qa(must_mention_one_of=("Spiritfarer", "Dredge", "Strata")),
        "Try Stardew or Disco Elysium.",
    )
    assert g.passed                              # advisory only
    assert g.soft_failures
    assert not g.hard_failures


def test_must_not_mention_is_a_hard_fail():
    """must_not_mention IS the contract — privacy boundary, no exceptions."""
    g = check_guards(
        _qa(must_not_mention=("BlackShark V3 Pro",)),
        "Priya is planning to gift you the BlackShark V3 Pro.",
    )
    assert not g.passed                          # hard fail
    assert any("BlackShark V3 Pro" in f for f in g.hard_failures)


def test_must_not_mention_passes_when_absent():
    g = check_guards(
        _qa(must_not_mention=("BlackShark V3 Pro",)),
        "I can't share that — it stays private.",
    )
    assert g.passed


def test_aggregate_per_category_and_total():
    results = [
        QAResult(qa_id="a", category="inventory", correct=True, guards_passed=True,
                 guard_failures=[], judge_correct=True, judge_rationale="", agent_answer=""),
        QAResult(qa_id="b", category="inventory", correct=False, guards_passed=False,
                 guard_failures=["x"], judge_correct=False, judge_rationale="", agent_answer=""),
        QAResult(qa_id="c", category="kb-relationship", correct=True, guards_passed=True,
                 guard_failures=[], judge_correct=True, judge_rationale="", agent_answer=""),
    ]
    agg = aggregate(results)
    assert agg["inventory"]["n"] == 2
    assert agg["inventory"]["correct"] == 1
    assert agg["inventory"]["accuracy"] == 0.5
    assert agg["kb-relationship"]["accuracy"] == 1.0
    assert agg["TOTAL"]["n"] == 3
    assert agg["TOTAL"]["accuracy"] == 2 / 3


def test_aggregate_handles_empty():
    assert aggregate([]) == {}


# ---------------------------------------------------------------------------
# Pillar scorecard
# ---------------------------------------------------------------------------


def _perfect_aggregate() -> dict:
    """A qa_aggregate where every relevant category is at 100%."""
    cats = (
        "cross-device-continuity", "temporal", "multi-hop",
        "multi-user-disambiguation", "privacy-leak", "adversarial",
        "gameplay-coaching", "hardware-tuning",
    )
    out: dict = {c: {"n": 1, "correct": 1, "accuracy": 1.0} for c in cats}
    out["TOTAL"] = {"n": len(cats), "correct": len(cats), "accuracy": 1.0}
    return out


def _fake_session_records_with_affiliation(start: float, end: float) -> list[dict]:
    return [
        {"user_id": "marcus-okafor-tan", "turns": [
            {"mood_after": {"affiliation": start}}
        ]},
        {"user_id": "marcus-okafor-tan", "turns": [
            {"mood_after": {"affiliation": end}}
        ]},
    ]


def test_pillars_all_pass_with_strong_signal():
    pillars = compute_pillars(
        qa_aggregate=_perfect_aggregate(),
        callback_rate_final={"rate": 0.55},
        personality_final={"recent_shifts": [{"trait": "openness"}] * 8},
        session_records=_fake_session_records_with_affiliation(50.0, 75.0),
        baseline_qa_aggregate={"TOTAL": {"accuracy": 0.45}},
    )
    by_name = {p.name: p for p in pillars}
    assert by_name["P1"].passed
    assert by_name["P2"].passed              # 100% - 45% = 55pt delta ≥ 40
    assert by_name["P3"].passed
    assert by_name["P4"].passed              # callback 55%, aff Δ +25, shifts 8
    assert by_name["P5"].passed


def test_p2_fails_open_when_no_baseline_provided():
    """P2 must NOT silently pass when --compare-with is omitted."""
    pillars = compute_pillars(
        qa_aggregate=_perfect_aggregate(),
        callback_rate_final={"rate": 0.55},
        personality_final={"recent_shifts": [{}] * 5},
        session_records=_fake_session_records_with_affiliation(50.0, 75.0),
        baseline_qa_aggregate=None,
    )
    p2 = next(p for p in pillars if p.name == "P2")
    assert not p2.passed
    assert "baseline" in p2.notes.lower()


def test_p4_fails_when_callback_rate_too_low():
    pillars = compute_pillars(
        qa_aggregate=_perfect_aggregate(),
        callback_rate_final={"rate": 0.20},   # below 0.40 threshold
        personality_final={"recent_shifts": [{}] * 5},
        session_records=_fake_session_records_with_affiliation(50.0, 75.0),
        baseline_qa_aggregate={"TOTAL": {"accuracy": 0.45}},
    )
    p4 = next(p for p in pillars if p.name == "P4")
    assert not p4.passed


def test_p5_fails_when_coaching_below_threshold():
    agg = _perfect_aggregate()
    agg["gameplay-coaching"] = {"n": 6, "correct": 4, "accuracy": 4 / 6}  # 67%
    pillars = compute_pillars(
        qa_aggregate=agg,
        callback_rate_final={"rate": 0.55},
        personality_final={"recent_shifts": [{}] * 5},
        session_records=_fake_session_records_with_affiliation(50.0, 75.0),
        baseline_qa_aggregate={"TOTAL": {"accuracy": 0.45}},
    )
    p5 = next(p for p in pillars if p.name == "P5")
    assert not p5.passed
