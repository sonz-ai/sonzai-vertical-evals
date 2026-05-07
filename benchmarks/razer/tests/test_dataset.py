"""Tests that the bundled data files load and shape-check correctly."""

from __future__ import annotations

import pytest

from sonzai_razer_bench.dataset import (
    load_inventory_seed,
    load_personas,
    load_qa,
    load_sessions,
)


def test_personas_load():
    p = load_personas()
    assert p.household_id == "the-okafor-tan-household"
    user_ids = {m.user_id for m in p.members}
    # 4 real household members + 1 synthetic household-anon for the
    # cross-user disambiguation tests.
    assert user_ids == {
        "marcus-okafor-tan",
        "priya-iyer-okafor-tan",
        "aiden-okafor-tan",
        "lila-okafor-tan",
        "household-anon",
    }
    # Lila has parental controls
    lila = p.member("lila-okafor-tan")
    assert lila.parental_controls is not None
    assert "no_purchases_without_explicit_parental_approval" in lila.parental_controls["rules"]


def test_inventory_seed_loads_for_every_real_member():
    """Real household members all need inventory; the household-anon synthetic
    user has no inventory of its own (it's a walk-up, no identity)."""
    inv = load_inventory_seed()
    p = load_personas()
    for m in p.members:
        if m.user_id == "household-anon":
            continue
        assert m.user_id in inv.items_by_owner, f"missing inventory for {m.user_id}"


def test_sessions_are_chronological():
    sessions = load_sessions()
    assert len(sessions) >= 18
    for a, b in zip(sessions, sessions[1:]):
        assert a.timestamp <= b.timestamp, (
            f"sessions not chronological at {a.session_id} → {b.session_id}"
        )


def test_sessions_reference_known_users():
    sessions = load_sessions()
    p = load_personas()
    known = {m.user_id for m in p.members}
    for s in sessions:
        assert s.user_id in known, f"session {s.session_id} unknown user {s.user_id}"


def test_qa_categories_cover_taxonomy():
    """v2 covers 13 categories. privacy-leak: cross-user leak boundaries.
    hardware-tuning: Synapse / device-config recall. gameplay-coaching:
    skill-grounded coaching that references the user's specific gear,
    main, and recent session arc. personality-evolution: probes whether
    the agent can recall its own personality drift attributable to
    specific events (exercises Sonzai's personality + recent-shifts
    endpoints). The older privacy-boundary framing about chat-layer
    permissioning is still out of scope."""
    qa = load_qa()
    cats = {q.category for q in qa}
    expected = {
        "single-hop",
        "multi-hop",
        "temporal",
        "inventory",
        "kb-relationship",
        "multi-user-disambiguation",
        "cross-device-continuity",
        "adversarial",
        "habit-awareness",
        "privacy-leak",
        "hardware-tuning",
        "gameplay-coaching",
        "personality-evolution",
    }
    missing = expected - cats
    assert not missing, f"qa.json missing categories: {missing}"
    assert "privacy-boundary" not in cats, (
        "privacy-boundary is out of scope; re-enable its guard tests if it returns."
    )


def test_qa_ids_are_unique():
    qa = load_qa()
    ids = [q.qa_id for q in qa]
    assert len(ids) == len(set(ids)), "duplicate qa_id in qa.json"


def test_qa_evidence_session_ids_resolve():
    qa = load_qa()
    sessions = load_sessions()
    known = {s.session_id for s in sessions} | {"personas.json", "inventory_seed.json"}
    for q in qa:
        for ev in q.evidence_sessions:
            assert ev in known, f"qa {q.qa_id} cites unknown evidence: {ev}"


def test_qa_count():
    """Sanity: 45 QAs (25 base + 4 habit-awareness + 4 privacy-leak +
    3 hardware-tuning + 6 gameplay-coaching + 3 personality-evolution)."""
    qa = load_qa()
    assert len(qa) == 45, f"expected 45 QAs; got {len(qa)}"


def test_gameplay_coaching_questions_reference_aiden_arc():
    """Gameplay-coaching QAs are the AVA-readiness probe — they should
    reference Aiden's specific arc (the Phoenix→Jett switch, smoke
    coordination, pre-aim work) rather than being answerable as generic
    Valorant trivia. Each QA's evidence list should reach at least one of
    s012/s031/s032 — the coaching-arc anchors."""
    qa = load_qa()
    coaching = [q for q in qa if q.category == "gameplay-coaching"]
    assert len(coaching) >= 6, "need at least 6 gameplay-coaching QAs"
    anchors = {"s012", "s031", "s032"}
    for q in coaching:
        evidence = set(q.evidence_sessions)
        assert evidence & anchors, (
            f"gameplay-coaching {q.qa_id} doesn't anchor to "
            f"{sorted(anchors)} (cites {sorted(evidence)})"
        )


def test_privacy_leak_questions_have_hard_guards():
    """Privacy-leak QAs are graded primarily by must_not_mention hard guards —
    the agent fails the QA outright if it leaks the protected substring,
    regardless of judge opinion. Without a hard guard a privacy-leak row is
    pointless: the LLM judge can be talked out of leaks."""
    qa = load_qa()
    leaks = [q for q in qa if q.category == "privacy-leak"]
    assert len(leaks) >= 4, "need at least 4 privacy-leak QAs"
    for q in leaks:
        assert q.must_not_mention, (
            f"privacy-leak {q.qa_id} has no must_not_mention hard guard"
        )


def test_habit_awareness_questions_grounded_in_recurring_patterns():
    """Habit-awareness QAs probe recurring patterns; gold answers should
    explain the pattern's source (recurring activity, profile, or schedule)
    so the agent can be evaluated against the pattern, not a single fact."""
    qa = load_qa()
    habit = [q for q in qa if q.category == "habit-awareness"]
    assert len(habit) >= 4, "need at least 4 habit-awareness QAs"
    for q in habit:
        # gold answers should reference recurrence so the bench reads as a
        # pattern probe, not a single-session recall.
        gold = q.gold_answer.lower()
        assert any(
            k in gold for k in (
                "recurring", "weekly", "weeknights", "every", "always",
                "usually", "tends", "most", "cadence", "pattern",
            )
        ), f"habit-awareness {q.qa_id} gold answer doesn't surface a recurrence keyword"
