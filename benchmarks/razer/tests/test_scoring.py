"""Tests for the substring-guard layer + aggregation. The LLM judge is
mocked separately in test_runner.py so the deterministic scoring path
gets exercised here without an API call."""

from __future__ import annotations

from sonzai_razer_bench.dataset import QA
from sonzai_razer_bench.scoring import (
    QAResult,
    aggregate,
    check_guards,
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
