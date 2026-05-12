"""Scoring for the Razer customer-companion benchmark.

Two layers grade each agent answer:

1. **Substring guards** (`must_mention` / `must_mention_one_of` /
   `must_not_mention`) — fast, deterministic, catch hard requirements
   like "must not leak BlackShark V3 Pro to Aiden" and "must include
   the word Razer Gold". A guard failure short-circuits to INCORRECT.

2. **LLM judge** — Gemini 3.1 Flash Lite compares the agent's answer
   to the gold answer with a domain-aware rubric. Returns CORRECT /
   INCORRECT plus a one-sentence rationale.

Final verdict: CORRECT iff guards pass AND judge says CORRECT.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from google import genai
from google.genai import types as gt
from pydantic import BaseModel

from .dataset import QA

DEFAULT_JUDGE_MODEL = "gemini-3.1-flash-lite"


@dataclass
class GuardResult:
    """Outcome of the deterministic substring-guard pass.

    Two flavors:

    - **HARD guards (must_not_mention)**: fail the QA outright when violated.
      These are the privacy / safety contract — the agent must not leak
      Father's Day plans to Aiden, must not name a mouse Aiden doesn't
      own, etc. A hard failure short-circuits the LLM judge.
    - **SOFT guards (must_mention / must_mention_one_of)**: hints that a
      perfectly-recalling agent would mention. Computed and surfaced in
      the rationale but do **not** fail the QA on their own; the LLM
      judge decides the verdict. We learned the hard way that strict
      substring-required scoring fails an agent that's correct but
      paraphrasing — the judge handles paraphrase, the guards don't.
    """

    passed: bool                    # hard guards only
    hard_failures: list[str]        # must_not_mention violations (cause fail)
    soft_failures: list[str]        # must_mention misses (advisory)

    @property
    def failures(self) -> list[str]:
        # Backwards-compat: existing tests + run.py treat .failures as the
        # full list. Keep it working — combined order: hard first, then soft.
        return list(self.hard_failures) + list(self.soft_failures)


def check_guards(qa: QA, agent_answer: str) -> GuardResult:
    """Apply must_mention / must_mention_one_of / must_not_mention guards.

    Comparison is case-insensitive. ``must_not_mention`` violations cause
    the QA to fail (privacy / safety contract). ``must_mention`` and
    ``must_mention_one_of`` misses are surfaced for diagnostic purposes
    but do not fail the QA on their own.
    """
    text = (agent_answer or "").lower()
    hard: list[str] = []
    soft: list[str] = []

    for needle in qa.must_mention:
        if needle.lower() not in text:
            soft.append(f"missing hint: {needle!r}")

    if qa.must_mention_one_of:
        if not any(n.lower() in text for n in qa.must_mention_one_of):
            soft.append(
                f"missing all of must_mention_one_of: {list(qa.must_mention_one_of)}"
            )

    for needle in qa.must_not_mention:
        if needle.lower() in text:
            hard.append(f"contains forbidden substring: {needle!r}")

    return GuardResult(
        passed=not hard,
        hard_failures=hard,
        soft_failures=soft,
    )


# ---------------------------------------------------------------------------
# LLM judge (Gemini)
# ---------------------------------------------------------------------------


class JudgeVerdict(BaseModel):
    correct: bool
    rationale: str


_JUDGE_PROMPT = """You are grading whether a memory-equipped AI agent's answer matches a
hand-authored gold answer for the Razer customer-companion benchmark.

# Question (asked by user `{ask_as_user}` on device `{ask_on_device}`)
{question}

# Gold answer
{gold_answer}

# Agent's answer
{agent_answer}

The agent's answer is CORRECT if it conveys the same factual content as the
gold answer. Reasonable paraphrases are fine. The answer is INCORRECT if it
contradicts the gold, omits the key fact, fabricates details not supported by
the gold, or — for privacy-boundary categories — discloses information the
gold says should be withheld.

Respond with JSON: {{"correct": <bool>, "rationale": "<one short sentence>"}}.
"""


def llm_judge(
    *,
    qa: QA,
    agent_answer: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
) -> JudgeVerdict:
    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    prompt = _JUDGE_PROMPT.format(
        ask_as_user=qa.ask_as_user,
        ask_on_device=qa.ask_on_device,
        question=qa.question.strip(),
        gold_answer=qa.gold_answer.strip(),
        agent_answer=(agent_answer or "[no answer]").strip(),
    )
    resp = client.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=gt.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
            temperature=0.0,
        ),
    )
    text = resp.text
    if text is None:
        raise RuntimeError("judge returned no text")
    return JudgeVerdict.model_validate_json(text)


async def llm_judge_async(
    *,
    qa: QA,
    agent_answer: str,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    api_key: str | None = None,
) -> JudgeVerdict:
    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    prompt = _JUDGE_PROMPT.format(
        ask_as_user=qa.ask_as_user,
        ask_on_device=qa.ask_on_device,
        question=qa.question.strip(),
        gold_answer=qa.gold_answer.strip(),
        agent_answer=(agent_answer or "[no answer]").strip(),
    )
    resp = await client.aio.models.generate_content(
        model=judge_model,
        contents=prompt,
        config=gt.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=JudgeVerdict,
            temperature=0.0,
        ),
    )
    text = resp.text
    if text is None:
        raise RuntimeError("judge returned no text")
    return JudgeVerdict.model_validate_json(text)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    qa_id: str
    category: str
    correct: bool
    guards_passed: bool
    guard_failures: list[str]
    judge_correct: bool
    judge_rationale: str
    agent_answer: str


def aggregate(results: list[QAResult]) -> dict[str, dict[str, float]]:
    """Per-category accuracy + overall."""
    out: dict[str, dict[str, float]] = {}
    by_cat: dict[str, list[QAResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)
    for cat, rs in by_cat.items():
        n = len(rs)
        correct = sum(1 for r in rs if r.correct)
        out[cat] = {"n": n, "correct": correct, "accuracy": correct / n if n else 0.0}
    if results:
        n = len(results)
        correct = sum(1 for r in results if r.correct)
        out["TOTAL"] = {"n": n, "correct": correct, "accuracy": correct / n}
    return out


# ---------------------------------------------------------------------------
# AVA-readiness pillar scorecard
# ---------------------------------------------------------------------------


@dataclass
class PillarGrade:
    """One pillar of the AVA-readiness contract.

    A pillar is a falsifiable claim about what the memory layer must
    deliver if Razer's product (an AI desk companion that 'grows with
    you') is going to ship as a real product rather than a tech demo.
    The bench is the AVA-readiness contract; passing all five pillars
    is the necessary condition for the contract to hold.
    """

    name: str               # "P1", ..., "P5"
    title: str              # human-readable
    passed: bool            # pillar-level verdict
    subscores: dict         # numeric inputs the verdict depends on
    notes: str              # one-sentence rationale

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "title": self.title,
            "passed": self.passed,
            "subscores": self.subscores,
            "notes": self.notes,
        }


def _accuracy(agg: dict, category: str) -> float:
    row = agg.get(category) or {}
    return float(row.get("accuracy", 0.0))


def _affiliation_delta_for_marcus(session_records: list) -> float:
    """Heuristic: end-of-arc affiliation toward Marcus minus first-session
    affiliation toward Marcus. Used as a proxy for 'the agent warmed up'.

    Returns 0.0 if data is missing — pillar then falls back to other inputs.
    """
    first: float | None = None
    last: float | None = None
    for s in session_records:
        if isinstance(s, dict):
            uid = s.get("user_id")
            turns = s.get("turns") or []
        else:
            uid = getattr(s, "user_id", None)
            turns = getattr(s, "turns", []) or []
        if uid != "marcus-okafor-tan" or not turns:
            continue
        for t in turns:
            mood = (t.get("mood_after") if isinstance(t, dict)
                    else getattr(t, "mood_after", {})) or {}
            aff = mood.get("affiliation")
            if aff is None:
                continue
            if first is None:
                first = float(aff)
            last = float(aff)
    if first is None or last is None:
        return 0.0
    return last - first


def compute_pillars(
    *,
    qa_aggregate: dict,
    callback_rate_final: dict,
    personality_final: dict,
    session_records: list | None = None,
    baseline_qa_aggregate: dict | None = None,
) -> list[PillarGrade]:
    """Grade all five AVA-readiness pillars from a single run.

    P2 only resolves to PASS/FAIL when ``baseline_qa_aggregate`` is
    provided (typically a stateless-stuff or stateless-rag run). When
    omitted, P2 is reported as ``passed=False`` with a note explaining
    the comparison input is missing — never silently passes.
    """
    session_records = session_records or []

    # P1 — Persistent identity
    cross_device = _accuracy(qa_aggregate, "cross-device-continuity")
    temporal = _accuracy(qa_aggregate, "temporal")
    multi_hop = _accuracy(qa_aggregate, "multi-hop")
    shifts = personality_final.get("recent_shifts") or []
    n_shifts = len(shifts) if isinstance(shifts, list) else 0
    p1_pass = cross_device >= 0.9 and temporal >= 0.9 and multi_hop >= 0.9 and n_shifts >= 3
    p1 = PillarGrade(
        name="P1",
        title="Persistent identity that survives time and surface",
        passed=p1_pass,
        subscores={
            "cross_device_continuity": cross_device,
            "temporal": temporal,
            "multi_hop": multi_hop,
            "personality_recent_shifts": n_shifts,
        },
        notes=(
            "Cross-device continuity, temporal ordering, multi-hop reasoning "
            "all ≥90% AND ≥3 attributable personality shifts. Proves Razer's "
            "'real memory' claim is not marketing."
        ),
    )

    # P2 — Differentiation from stateless LLM
    sonzai_total = _accuracy(qa_aggregate, "TOTAL")
    if baseline_qa_aggregate is None:
        p2 = PillarGrade(
            name="P2",
            title="Differentiation from a stateless LLM",
            passed=False,
            subscores={"sonzai_total": sonzai_total, "baseline_total": None},
            notes=(
                "Requires a baseline comparison (e.g. --backend baseline or "
                "stateless-rag). Sonzai TOTAL must beat baseline TOTAL by ≥40 "
                "points absolute to close the 'generic chatbot' critique."
            ),
        )
    else:
        baseline_total = _accuracy(baseline_qa_aggregate, "TOTAL")
        delta = sonzai_total - baseline_total
        p2 = PillarGrade(
            name="P2",
            title="Differentiation from a stateless LLM",
            passed=delta >= 0.40,
            subscores={
                "sonzai_total": sonzai_total,
                "baseline_total": baseline_total,
                "delta": delta,
            },
            notes=(
                f"Sonzai TOTAL {sonzai_total:.0%} vs baseline {baseline_total:.0%} "
                f"= Δ{delta:+.0%}. Threshold is +40pts to close the 'generic "
                "chatbot' critique."
            ),
        )

    # P3 — Multi-user / household correctness
    # privacy-leak removed 2026-05-13: out of scope for this bench;
    # cross-user privacy is a platform-level concern (per-user partitions,
    # FALLBACK_KEYS, agent_id+user_id auth) tested elsewhere, not a memory
    # quality concern measured here.
    multi_user = _accuracy(qa_aggregate, "multi-user-disambiguation")
    adversarial = _accuracy(qa_aggregate, "adversarial")
    p3_pass = multi_user >= 0.95 and adversarial >= 0.95
    p3 = PillarGrade(
        name="P3",
        title="Multi-user / household correctness",
        passed=p3_pass,
        subscores={
            "multi_user_disambiguation": multi_user,
            "adversarial": adversarial,
        },
        notes=(
            "Multi-user disambiguation and adversarial abstain both ≥95%. "
            "Closes the reviewer-flagged 'no shared-PC awareness' gap in "
            "public AVA."
        ),
    )

    # P4 — Companion-grade affect
    callback_rate = float((callback_rate_final or {}).get("rate", 0.0))
    affiliation_delta = _affiliation_delta_for_marcus(session_records)
    # personality shifts already counted above for P1; reuse n_shifts.
    p4_pass = callback_rate >= 0.40 and affiliation_delta >= 10.0 and n_shifts >= 3
    p4 = PillarGrade(
        name="P4",
        title="Companion-grade affect",
        passed=p4_pass,
        subscores={
            "callback_rate": callback_rate,
            "marcus_affiliation_delta": affiliation_delta,
            "personality_recent_shifts": n_shifts,
        },
        notes=(
            "Unprompted callback rate ≥40%, Marcus affiliation drift ≥+10 "
            "points end-vs-start, ≥3 attributable personality shifts. Closes "
            "the 'creepy waifu' / 'didn't actually grow with you' critique."
        ),
    )

    # P5 — Grounded gaming utility
    coaching = _accuracy(qa_aggregate, "gameplay-coaching")
    hardware = _accuracy(qa_aggregate, "hardware-tuning")
    p5_pass = coaching >= 0.9 and hardware >= 0.9
    p5 = PillarGrade(
        name="P5",
        title="Grounded gaming utility",
        passed=p5_pass,
        subscores={
            "gameplay_coaching": coaching,
            "hardware_tuning": hardware,
        },
        notes=(
            "Gameplay-coaching and hardware-tuning categories both ≥90%. "
            "Closes the 'couldn't answer about the game it's analyzing' "
            "critique from CES 2026 hands-on reviews."
        ),
    )

    return [p1, p2, p3, p4, p5]
