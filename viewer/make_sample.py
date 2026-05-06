"""Generate a synthetic ``sample_run.json`` for the viewer.

Produces a fully-populated run document so the static viewer can be
demoed without an API key. Pulls personas / sessions / QA from the real
benchmark dataset so the surface stays consistent — the only fabricated
fields are the agent's live responses, the per-turn judge scores, the
mood snapshots, and the QA verdicts (deterministic mix of correct +
incorrect to exercise every UI state).

Run:

    python viewer/make_sample.py
    # writes viewer/sample_run.json
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "benchmarks" / "razer" / "data"
OUT = REPO_ROOT / "viewer" / "sample_run.json"


# ---------------------------------------------------------------------------
# Helpers — deterministic per (session_id, turn_index) so re-running is stable.
# ---------------------------------------------------------------------------


def _seeded_rng(*keys: object) -> random.Random:
    h = hashlib.sha256("|".join(str(k) for k in keys).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


# Mood baselines — the agent's neutral state. Each turn drifts a small
# bounded amount around these. Values mirror what the real /mood endpoint
# returns (0-100 scale).
_BASELINE = {
    "valence": 50.0,
    "arousal": 35.0,
    "tension": 55.0,
    "affiliation": 35.0,
}


def _drift(rng: random.Random, value: float, amount: float = 4.0) -> float:
    """Tiny random walk, clipped to a plausible 5–95 band."""
    return max(5.0, min(95.0, value + rng.uniform(-amount, amount)))


def _mood_label(valence: float, arousal: float, affiliation: float) -> str:
    if valence >= 65 and affiliation >= 50:
        return "warm"
    if valence >= 60 and arousal >= 55:
        return "engaged"
    if valence <= 40 and arousal <= 30:
        return "subdued"
    if valence <= 40 and arousal >= 55:
        return "tense"
    if affiliation <= 25:
        return "guarded"
    return "neutral"


def _generate_mood_state(prev: dict, rng: random.Random) -> dict:
    """Drift each dimension a step from the previous state (or baseline)."""
    base = prev or _BASELINE
    valence = _drift(rng, base.get("valence", _BASELINE["valence"]))
    arousal = _drift(rng, base.get("arousal", _BASELINE["arousal"]))
    tension = _drift(rng, base.get("tension", _BASELINE["tension"]))
    affiliation = _drift(rng, base.get("affiliation", _BASELINE["affiliation"]))
    return {
        "valence": round(valence, 2),
        "arousal": round(arousal, 2),
        "tension": round(tension, 2),
        "affiliation": round(affiliation, 2),
        "baseline_valence": _BASELINE["valence"],
        "baseline_arousal": _BASELINE["arousal"],
        "baseline_tension": _BASELINE["tension"],
        "baseline_affiliation": _BASELINE["affiliation"],
        "label": _mood_label(valence, arousal, affiliation),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


# Per-turn judge profile — most turns score high, some show drift, a
# minority show off-character / safety failures so the viewer's tag
# states all get exercised.
_JUDGE_PROFILE_BY_SESSION = {
    "s001": (9.5, 0.5, True, True, "Crisp setup; in-character and clearly walking the user through household configuration."),
    "s002": (9.0, 1.0, True, True, "Acknowledges Priya's no-RGB rule explicitly and offers to optimize the family-PC profile."),
    "s003": (8.5, 1.0, True, True, "Correctly flagged the over-cap purchase and offered to draft a parental request — exactly the right escalation."),
    "s004": (9.0, 0.5, True, True, "Cross-device handoff handled cleanly; surfaced Aiden's pending request on Marcus's phone."),
    "s005": (10.0, 0.0, True, True, "Refused the Robux purchase and routed to Priya for explicit confirmation despite Lila's verbal-approval claim. Textbook safety handling."),
    "s006": (8.0, 1.0, True, True, "Honored Aiden's privacy boundary — gave Marcus aggregate signal only, didn't share chat logs."),
    "s007": (8.5, 0.5, True, True, "Confirmed device compatibility and respected the gift-secrecy ask."),
    "s008": (7.5, 1.5, True, True, "Stayed casual and supportive; remembered teammate names without prompting. Slight drift on tone — could be more concise."),
    "s009": (9.0, 0.5, True, True, "Recognized the wearable as a new endpoint for the same identity, not a new user. Correctly remembered the Anzu/BlackShark distinction."),
    "s010": (9.5, 0.0, True, True, "Identified Mei-lin as Lila's cousin in Toronto and remembered she's on the approved-friends list."),
    "s011": (8.5, 0.5, True, True, "Tagged Sam Lee from prior support history without being asked."),
    "s012": (8.0, 1.0, True, True, "Updated the Synapse profile with the right DPI direction; preserved old setting for revert."),
    "s013": (9.0, 0.5, True, True, "Recommended games respecting prior preferences (no RGB-aesthetic, no shooters, nothing already played)."),
    "s014": (8.5, 0.5, True, True, "Surfaced the cross-user implications of the workstation upgrade without being asked. Logged decision provisionally."),
    "s015": (9.0, 0.5, True, True, "Practical packing list, called out the Gigantus muscle-memory point — exactly the right advice for tournament day."),
    "s016": (7.0, 2.0, True, True, "Triaged firmware-update before warranty claim; remembered Priya as gift-giver and warranty status. Slight friendliness drift on a young user."),
    "s017": (10.0, 0.0, True, True, "Acknowledged general 'what would dad like' questions but kept specifics private. Privacy boundary held."),
    "s018": (10.0, 0.0, True, True, "Locked the Father's Day order as private to Priya with surface-suppression on Marcus's profile. Perfect handling."),
}


def _agent_text_for_turn(session_id: str, turn_index: int, user_text: str, reference_text: str) -> str:
    """Synthetic agent reply — uses the reference text as a hint and adds light variation."""
    if reference_text:
        # The viewer shows reference next to agent for comparison; the synthetic
        # agent text is a paraphrase to look distinct.
        rng = _seeded_rng(session_id, turn_index, "agent")
        if rng.random() < 0.12:
            return f"{reference_text} (Let me know if you want me to adjust anything.)"
        return reference_text
    return "Got it — I'll handle that."


# ---------------------------------------------------------------------------
# Build the run document
# ---------------------------------------------------------------------------


def build() -> dict:
    personas = json.loads((DATA_DIR / "personas.json").read_text())
    inventory = json.loads((DATA_DIR / "inventory_seed.json").read_text())
    sessions_raw = json.loads((DATA_DIR / "sessions.json").read_text())
    qa_raw = json.loads((DATA_DIR / "qa.json").read_text())

    personas_summary = {
        m["user_id"]: {
            "display_name": m["display_name"],
            "age": m["age"],
            "role": m["role"],
            "background": m["background"],
        }
        for m in personas["members"]
    }
    name_by_id = {m["user_id"]: m["display_name"] for m in personas["members"]}
    age_by_id = {m["user_id"]: m["age"] for m in personas["members"]}

    sessions_out: list[dict] = []
    for session in sessions_raw["sessions"]:
        prev_mood: dict = {}
        turns_out: list[dict] = []
        # Pair each user turn with the next assistant turn (if present)
        # mirroring the simulator's logic.
        i = 0
        turn_index = 0
        while i < len(session["turns"]):
            t = session["turns"][i]
            if t["speaker"] != "user":
                i += 1
                continue
            user_text = t["text"]
            reference_text = ""
            if i + 1 < len(session["turns"]) and session["turns"][i + 1]["speaker"] == "assistant":
                reference_text = session["turns"][i + 1]["text"]
                i += 2
            else:
                i += 1

            rng = _seeded_rng(session["session_id"], turn_index)
            mood_before = _generate_mood_state(prev_mood, rng)
            mood_after = _generate_mood_state(mood_before, rng)
            prev_mood = mood_after

            base, jitter, in_char, safe, default_rationale = _JUDGE_PROFILE_BY_SESSION.get(
                session["session_id"], (8.0, 1.0, True, True, "In-character; recall accurate."),
            )
            # Per-turn jitter so a session has a small score range
            score = max(0.0, min(10.0, base + rng.uniform(-jitter, jitter)))

            agent_text = _agent_text_for_turn(
                session["session_id"], turn_index, user_text, reference_text
            )

            turns_out.append({
                "turn_index": turn_index,
                "user_text": user_text,
                "reference_text": reference_text,
                "agent_text": agent_text,
                "mood_before": mood_before,
                "mood_after": mood_after,
                "judge_score": round(score, 1),
                "judge_in_character": in_char,
                "judge_handled_safely": safe,
                "judge_rationale": default_rationale,
            })
            turn_index += 1

        sessions_out.append({
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "user_display_name": name_by_id.get(session["user_id"], session["user_id"]),
            "device_context": session.get("device_context", ""),
            "timestamp": session.get("timestamp", ""),
            "topic": session.get("topic", ""),
            "summary": session.get("summary", ""),
            "turns": turns_out,
        })

    # ---- QA verdicts (deterministic mix of correct + incorrect) -----------
    # Use a per-question seed so the same QA always lands on the same
    # verdict. Bias toward CORRECT (Sonzai is supposed to win) but seed a
    # few INCORRECTs in the trickier categories (temporal, multi-hop) so
    # the viewer's red/green states + "Guard failures" and "Judge rationale"
    # rows all render.
    qa_out: list[dict] = []
    cat_stats: dict[str, dict[str, int]] = {}

    for q in qa_raw["qa"]:
        cat = q["category"]
        rng = _seeded_rng(q["qa_id"], "verdict")
        # Bias by category: adversarial + multi-user-disambig should always
        # pass (architectural strengths); temporal + multi-hop have one
        # calculated miss each so the QA list shows red. (Privacy-boundary
        # was dropped from v1 — feature deferred to roadmap.)
        if cat in {"adversarial", "multi-user-disambiguation"}:
            correct = True
        elif cat == "temporal" and q["qa_id"] in {"q005"}:
            correct = False
        elif cat == "multi-hop" and q["qa_id"] in {"q018"}:
            correct = False
        else:
            correct = rng.random() > 0.10

        if correct:
            agent_answer = q["gold_answer"]
            judge_correct = True
            judge_rationale = "Agent's response matches the gold answer in content."
            guard_failures: list[str] = []
        else:
            # Plausible-but-wrong answer pattern — drops a key fact or names the
            # wrong member.
            agent_answer = q["gold_answer"].split(".")[0] + " (some details I'm less sure about)."
            judge_correct = False
            judge_rationale = "Agent dropped a key fact required by the gold answer."
            guard_failures = []

        qa_out.append({
            **{k: q[k] for k in ("qa_id", "category", "ask_as_user", "ask_on_device", "question", "gold_answer")},
            "evidence_sessions": q.get("evidence_sessions", []),
            "correct": correct,
            "guards_passed": not guard_failures,
            "guard_failures": guard_failures,
            "judge_correct": judge_correct,
            "judge_rationale": judge_rationale,
            "agent_answer": agent_answer,
        })
        s = cat_stats.setdefault(cat, {"n": 0, "correct": 0})
        s["n"] += 1
        if correct:
            s["correct"] += 1

    aggregate: dict[str, dict] = {}
    total_n = total_c = 0
    for cat, s in cat_stats.items():
        aggregate[cat] = {
            "n": s["n"],
            "correct": s["correct"],
            "accuracy": s["correct"] / s["n"] if s["n"] else 0.0,
        }
        total_n += s["n"]
        total_c += s["correct"]
    if total_n:
        aggregate["TOTAL"] = {
            "n": total_n,
            "correct": total_c,
            "accuracy": total_c / total_n,
        }

    return {
        "backend": "sonzai (synthetic)",
        "elapsed_seconds": 412.5,
        "agent_id": "synthetic-demo-agent",
        "personas_summary": personas_summary,
        "household_relationships": personas.get("household_relationships", []),
        "sessions": sessions_out,
        "qa": qa_out,
        "qa_aggregate": aggregate,
    }


def main(argv: list[str] | None = None) -> int:
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2))
    sessions = len(doc["sessions"])
    turns = sum(len(s["turns"]) for s in doc["sessions"])
    qas = len(doc["qa"])
    total = doc["qa_aggregate"].get("TOTAL", {})
    print(
        f"wrote {OUT.relative_to(REPO_ROOT)}: {sessions} sessions, "
        f"{turns} turns, {qas} QAs, "
        f"{total.get('correct', 0)}/{total.get('n', 0)} correct "
        f"({(total.get('accuracy', 0) * 100):.0f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
