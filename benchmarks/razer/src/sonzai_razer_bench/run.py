"""Razer customer-companion benchmark runner.

Two-phase flow that produces the data points the viewer renders for an
exec demo:

PHASE 1 — Live conversation simulation (Sonzai backend only).
  For each session in ``data/sessions.json``, the simulated user's lines
  are replayed turn-by-turn to the Sonzai agent. The agent's live
  response (via ``client.agents.chat``) is captured alongside a mood
  snapshot before/after the turn and a per-turn LLM judge score. The
  pre-written assistant turns from the dataset become *reference* text
  for side-by-side comparison.

PHASE 2 — QA evaluation.
  After the conversation arc, each QA in ``data/qa.json`` is asked of
  the agent (Sonzai or baseline) scoped to the right user_id. Substring
  guards + LLM judge produce a per-question correctness verdict. The
  per-category accuracy table and a single rich JSONL output are written
  to ``results/``. Viewer reads the JSONL and replays the demo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .dataset import (
    QA,
    load_inventory_seed,
    load_personas,
    load_qa,
    load_sessions,
)
from .scoring import (
    DEFAULT_JUDGE_MODEL,
    PillarGrade,
    QAResult,
    aggregate,
    check_guards,
    compute_pillars,
    llm_judge_async,
)

logger = logging.getLogger("sonzai_razer_bench")

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _filter_qa(
    qa: list[QA], categories: list[str] | None, limit: int | None
) -> list[QA]:
    out = qa
    if categories:
        wanted = {c.strip() for c in categories if c.strip()}
        out = [q for q in out if q.category in wanted]
    if limit and limit > 0:
        out = out[:limit]
    return out


async def _grade_one(
    *, qa: QA, agent_answer: str, judge_model: str
) -> QAResult:
    guards = check_guards(qa, agent_answer)
    judge_correct = False
    rationale = ""
    if guards.passed:
        try:
            verdict = await llm_judge_async(
                qa=qa, agent_answer=agent_answer, judge_model=judge_model
            )
            judge_correct = verdict.correct
            rationale = verdict.rationale
        except Exception as e:
            rationale = f"judge_failed: {e}"
    else:
        rationale = "; ".join(guards.failures)
    correct = guards.passed and judge_correct
    return QAResult(
        qa_id=qa.qa_id,
        category=qa.category,
        correct=correct,
        guards_passed=guards.passed,
        guard_failures=list(guards.failures),
        judge_correct=judge_correct,
        judge_rationale=rationale,
        agent_answer=agent_answer,
    )


def _print_table(agg: dict[str, dict[str, float]]) -> None:
    print("\n=== Razer customer-companion ===")
    print(f"{'category'.ljust(28)}{'n':>6}{'correct':>10}{'accuracy':>11}")
    print("-" * 55)
    for cat, row in sorted(agg.items()):
        if cat == "TOTAL":
            continue
        print(
            f"{cat.ljust(28)}{int(row['n']):>6}{int(row['correct']):>10}"
            f"{row['accuracy']:>10.0%}"
        )
    if "TOTAL" in agg:
        print("-" * 55)
        row = agg["TOTAL"]
        print(
            f"{'TOTAL'.ljust(28)}{int(row['n']):>6}{int(row['correct']):>10}"
            f"{row['accuracy']:>10.0%}"
        )


def _print_pillars(pillars: list[PillarGrade]) -> None:
    print("\n=== AVA-readiness pillar scorecard ===")
    for p in pillars:
        mark = "✓ PASS" if p.passed else "✗ FAIL"
        print(f"  {p.name} {p.title.ljust(47)} {mark}")
        print(f"      {p.notes}")
    n_pass = sum(1 for p in pillars if p.passed)
    print(f"\nPillars passed: {n_pass}/{len(pillars)}")


def _qa_to_meta(qa: QA) -> dict:
    return {
        "qa_id": qa.qa_id,
        "category": qa.category,
        "ask_as_user": qa.ask_as_user,
        "ask_on_device": qa.ask_on_device,
        "question": qa.question,
        "gold_answer": qa.gold_answer,
        "evidence_sessions": list(qa.evidence_sessions),
    }


# ---------------------------------------------------------------------------
# Sonzai backend: live conversation + QA
# ---------------------------------------------------------------------------


def _callback_rate(session_records: list) -> dict:
    """Aggregate `judge_made_natural_callback` across all turns scored so far.

    Returns ``{n_turns_judged, n_callbacks, rate}``. Turns where the per-turn
    judge wasn't run (judge_score == 0 and judge_rationale empty) are
    excluded so the rate doesn't get diluted by unjudged turns.
    """
    n_judged = 0
    n_callbacks = 0
    for s in session_records:
        # SessionRecord (dataclass) or dict — handle both
        turns = getattr(s, "turns", None) or (s.get("turns") if isinstance(s, dict) else [])
        for t in turns:
            if hasattr(t, "judge_score"):
                judge_score = t.judge_score
                rationale = t.judge_rationale
                callback = t.judge_made_natural_callback
            elif isinstance(t, dict):
                judge_score = t.get("judge_score", 0)
                rationale = t.get("judge_rationale", "")
                callback = t.get("judge_made_natural_callback", False)
            else:
                continue
            if judge_score > 0 or rationale:
                n_judged += 1
                if callback:
                    n_callbacks += 1
    rate = (n_callbacks / n_judged) if n_judged else 0.0
    return {
        "n_turns_judged": n_judged,
        "n_callbacks": n_callbacks,
        "rate": rate,
    }


def _parse_snapshots(spec: str, n_sessions: int) -> list[int]:
    """Parse '10,20,30' into [10, 20, 30], dropping out-of-range values."""
    if not spec.strip():
        return []
    out: list[int] = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            i = int(piece)
        except ValueError:
            continue
        if 1 <= i <= n_sessions:
            out.append(i)
    return sorted(set(out))


async def _personality_snapshot(
    client, *, agent_id: str, history_limit: int = 200
) -> dict:
    """Capture personality + recent-shifts for the trajectory record.

    Returns a slim dict the viewer + downstream tools can read:
      {profile, evolution[..N], recent_shifts}
    """
    from .backends._compat import (
        async_personality,
        get_personality_recent_shifts_async,
    )
    out: dict = {}
    try:
        resp = await async_personality(client).get_personality(
            agent_id=agent_id, history_limit=history_limit
        )
        if hasattr(resp, "model_dump"):
            payload = resp.model_dump()
        else:
            payload = dict(resp) if isinstance(resp, dict) else {}
        out["profile"] = payload.get("profile") or {}
        out["evolution"] = payload.get("evolution") or []
    except Exception as e:
        logger.debug("personality snapshot failed: %s", e)
    try:
        shifts = await get_personality_recent_shifts_async(
            client, agent_id=agent_id
        )
        out["recent_shifts"] = (
            shifts.get("shifts") if isinstance(shifts, dict) and "shifts" in shifts
            else shifts
        )
    except Exception as e:
        logger.debug("recent_shifts snapshot failed: %s", e)
    return out


async def _run_sonzai(args: argparse.Namespace) -> dict:
    from sonzai import AsyncSonzai

    from .backends import sonzai as sonzai_backend
    from .backends._compat import clear_agent_memory_async
    from .simulator import make_judge_client, simulate_all_sessions

    personas = load_personas()
    inventory_seed = load_inventory_seed()
    sessions = load_sessions()
    qa_all = load_qa()
    qa = _filter_qa(qa_all, args.categories, args.limit)

    # Maps for the simulator's prompts.
    user_display_names = {m.user_id: m.display_name for m in personas.members}
    user_ages = {m.user_id: m.age for m in personas.members}
    # Users for advance_time sweep — every household member, including the
    # household-anon synthetic user (so its memory partition still ages
    # alongside the family's, even though it never speaks).
    advance_users = [m.user_id for m in personas.members] if args.advance_time else None

    snapshots = _parse_snapshots(args.snapshot_at, len(sessions))

    client = AsyncSonzai(timeout=600.0)
    t0 = time.time()
    try:
        state = await sonzai_backend.init_backend(
            client=client,
            personas=personas,
            inventory_seed=inventory_seed,
            seed=not args.skip_seed_inventory,
        )

        # Reset memory partitions BEFORE any ingest, for every household user.
        # Without this, re-running pollutes memory with duplicate facts from
        # earlier ingests.
        if args.reset_memory and not args.skip_simulation:
            logger.info("resetting memory for %d users", len(personas.members))
            await asyncio.gather(*(
                clear_agent_memory_async(
                    client, agent_id=state.agent_id, user_id=m.user_id
                )
                for m in personas.members
            ))

        # PHASE 1 — live conversation simulation, optionally with snapshots
        session_records: list = []
        snapshot_records: list[dict] = []

        async def _qa_pass(label: str) -> tuple[list[QAResult], list[dict]]:
            """Run the full QA set against the current agent state."""
            sem = asyncio.Semaphore(max(1, args.qa_concurrency))

            async def _one(q: QA) -> QAResult:
                async with sem:
                    answer = await sonzai_backend.ask(
                        client=client, state=state, qa=q
                    )
                    return await _grade_one(
                        qa=q, agent_answer=answer, judge_model=args.judge_model
                    )

            results = list(await asyncio.gather(*(_one(q) for q in qa)))
            rows = [{**_qa_to_meta(q), **asdict(r)} for q, r in zip(qa, results)]
            logger.info("snapshot %s: %d/%d correct", label, sum(r.correct for r in results), len(results))
            return results, rows

        if not args.skip_simulation:
            judge_client = make_judge_client() if not args.skip_turn_judge else None

            if snapshots:
                # Trajectory mode: cut the session list at each snapshot index,
                # ingest that slice, then run QA, then continue.
                last_idx = 0
                for snap_idx in snapshots:
                    slice_sessions = sessions[last_idx:snap_idx]
                    if slice_sessions:
                        # Pass `prev_ts` semantics: simulate_all_sessions starts
                        # advance_time AFTER its first session. To preserve the
                        # gap between the LAST session of the prior slice and
                        # the FIRST session of this slice, we feed the prior
                        # tail in as a sentinel via advance_time_min_hours
                        # OR re-use the same call. Simplest: just call again.
                        slice_records = await simulate_all_sessions(
                            sonzai_client=client,
                            agent_id=state.agent_id,
                            sessions=slice_sessions,
                            user_display_names=user_display_names,
                            user_ages=user_ages,
                            judge_client=judge_client,
                            judge_model=args.judge_model,
                            judge_each_turn=not args.skip_turn_judge,
                            advance_time_users=advance_users,
                            advance_time_min_hours=args.advance_time_min_hours,
                        )
                        session_records.extend(slice_records)
                    last_idx = snap_idx
                    qa_results_at, qa_rows_at = await _qa_pass(f"@s{snap_idx}")
                    personality_at = await _personality_snapshot(
                        client, agent_id=state.agent_id
                    )
                    callback_rate_at = _callback_rate(session_records)
                    snapshot_records.append({
                        "at_session": snap_idx,
                        "qa_aggregate": aggregate(qa_results_at),
                        "qa": qa_rows_at,
                        "personality": personality_at,
                        "callback_rate": callback_rate_at,
                    })
                # Tail beyond last snapshot
                if last_idx < len(sessions):
                    tail = sessions[last_idx:]
                    tail_records = await simulate_all_sessions(
                        sonzai_client=client,
                        agent_id=state.agent_id,
                        sessions=tail,
                        user_display_names=user_display_names,
                        user_ages=user_ages,
                        judge_client=judge_client,
                        judge_model=args.judge_model,
                        judge_each_turn=not args.skip_turn_judge,
                        advance_time_users=advance_users,
                        advance_time_min_hours=args.advance_time_min_hours,
                    )
                    session_records.extend(tail_records)
            else:
                # Single-pass mode (no trajectory).
                session_records = await simulate_all_sessions(
                    sonzai_client=client,
                    agent_id=state.agent_id,
                    sessions=sessions,
                    user_display_names=user_display_names,
                    user_ages=user_ages,
                    judge_client=judge_client,
                    judge_model=args.judge_model,
                    judge_each_turn=not args.skip_turn_judge,
                    advance_time_users=advance_users,
                    advance_time_min_hours=args.advance_time_min_hours,
                )

        # PHASE 2 — final QA (always runs unless --ingest-only)
        qa_results: list[QAResult] = []
        if not args.ingest_only:
            qa_results, _ = await _qa_pass("final")

        # Final personality + callback-rate snapshot (always — even on
        # single-pass mode, gives the run dict the same companion-grade
        # measurements that trajectory snapshots carry).
        final_personality = await _personality_snapshot(
            client, agent_id=state.agent_id
        )
        final_callback_rate = _callback_rate(session_records)

    finally:
        await client.close()

    elapsed = time.time() - t0
    qa_agg = aggregate(qa_results)
    sessions_dump = [asdict(s) for s in session_records]
    pillars = compute_pillars(
        qa_aggregate=qa_agg,
        callback_rate_final=final_callback_rate,
        personality_final=final_personality,
        session_records=sessions_dump,
        baseline_qa_aggregate=_load_baseline_aggregate(args.compare_with),
    )
    return {
        "backend": "sonzai",
        "elapsed_seconds": elapsed,
        "agent_id": state.agent_id,
        "personas_summary": {
            m.user_id: {
                "display_name": m.display_name,
                "age": m.age,
                "role": m.role,
                "background": m.background,
            }
            for m in personas.members
        },
        "household_relationships": [
            {"a": r.a, "b": r.b, "type": r.type, "since": r.since}
            for r in personas.household_relationships
        ],
        "sessions": sessions_dump,
        "snapshots": snapshot_records,
        "qa": [
            {**_qa_to_meta(q), **asdict(r)}
            for q, r in zip(qa, qa_results)
        ],
        "qa_aggregate": qa_agg,
        "personality_final": final_personality,
        "callback_rate_final": final_callback_rate,
        "pillars": [p.to_dict() for p in pillars],
    }


def _load_baseline_aggregate(path: Path | None) -> dict | None:
    """Read a prior run JSON and pull out its qa_aggregate for P2.

    Returns None if no path given or read fails — pillar P2 will then
    surface as 'requires baseline comparison' rather than silently passing.
    """
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("qa_aggregate")
    except Exception as e:
        logger.warning("could not load --compare-with %s: %s", path, e)
        return None


# ---------------------------------------------------------------------------
# Baseline backend: full-history-in-prompt, QA only (no live convo)
# ---------------------------------------------------------------------------


async def _run_mempalace(args: argparse.Namespace) -> dict:
    from .backends import mempalace as mp_backend

    personas = load_personas()
    inventory_seed = load_inventory_seed()
    sessions = load_sessions()
    qa = _filter_qa(load_qa(), args.categories, args.limit)

    user_display_names = {m.user_id: m.display_name for m in personas.members}
    user_ages = {m.user_id: m.age for m in personas.members}

    t0 = time.time()
    state = mp_backend.init_backend(
        personas=personas,
        inventory_seed=inventory_seed,
        gen_model=args.baseline_model,
    )

    session_records: list = []
    if not args.skip_simulation:
        session_records = await mp_backend.simulate_all_sessions(
            state=state,
            sessions=sessions,
            user_display_names=user_display_names,
            user_ages=user_ages,
        )

    qa_results: list[QAResult] = []
    if not args.ingest_only:
        sem = asyncio.Semaphore(max(1, args.qa_concurrency))

        async def _one(q: QA) -> QAResult:
            async with sem:
                answer = await mp_backend.ask(state=state, qa=q)
                return await _grade_one(
                    qa=q, agent_answer=answer, judge_model=args.judge_model
                )

        qa_results = list(await asyncio.gather(*(_one(q) for q in qa)))

    elapsed = time.time() - t0
    qa_agg = aggregate(qa_results)
    pillars = compute_pillars(
        qa_aggregate=qa_agg,
        callback_rate_final={},  # MemPalace has no callback measurement
        personality_final={},    # MemPalace has no personality model
        session_records=session_records,
        baseline_qa_aggregate=_load_baseline_aggregate(args.compare_with),
    )
    return {
        "backend": "mempalace",
        "elapsed_seconds": elapsed,
        "personas_summary": {
            m.user_id: {
                "display_name": m.display_name,
                "age": m.age,
                "role": m.role,
                "background": m.background,
            }
            for m in personas.members
        },
        "household_relationships": [
            {"a": r.a, "b": r.b, "type": r.type, "since": r.since}
            for r in personas.household_relationships
        ],
        "sessions": session_records,
        "qa": [
            {**_qa_to_meta(q), **asdict(r)}
            for q, r in zip(qa, qa_results)
        ],
        "qa_aggregate": qa_agg,
        "pillars": [p.to_dict() for p in pillars],
    }


async def _run_baseline(args: argparse.Namespace) -> dict:
    from .backends import baseline as baseline_backend

    personas = load_personas()
    inventory_seed = load_inventory_seed()
    sessions = load_sessions()
    qa = _filter_qa(load_qa(), args.categories, args.limit)

    t0 = time.time()
    state = baseline_backend.ingest_sessions(
        personas=personas,
        inventory_seed=inventory_seed,
        sessions=sessions,
        model=args.baseline_model,
    )
    qa_results: list[QAResult] = []
    if not args.ingest_only:
        sem = asyncio.Semaphore(max(1, args.qa_concurrency))

        async def _one(q: QA) -> QAResult:
            async with sem:
                answer = await baseline_backend.ask(state=state, qa=q)
                return await _grade_one(
                    qa=q, agent_answer=answer, judge_model=args.judge_model
                )

        qa_results = list(await asyncio.gather(*(_one(q) for q in qa)))
    elapsed = time.time() - t0
    qa_agg = aggregate(qa_results)
    pillars = compute_pillars(
        qa_aggregate=qa_agg,
        callback_rate_final={},
        personality_final={},
        session_records=[],
        baseline_qa_aggregate=_load_baseline_aggregate(args.compare_with),
    )
    return {
        "backend": "baseline",
        "elapsed_seconds": elapsed,
        "personas_summary": {
            m.user_id: {
                "display_name": m.display_name,
                "age": m.age,
                "role": m.role,
                "background": m.background,
            }
            for m in personas.members
        },
        "sessions": [],  # baseline doesn't run a live convo
        "qa": [
            {**_qa_to_meta(q), **asdict(r)}
            for q, r in zip(qa, qa_results)
        ],
        "qa_aggregate": qa_agg,
        "pillars": [p.to_dict() for p in pillars],
    }


async def _run_stateless_rag(args: argparse.Namespace) -> dict:
    from .backends import stateless_rag as rag_backend

    personas = load_personas()
    inventory_seed = load_inventory_seed()
    sessions = load_sessions()
    qa = _filter_qa(load_qa(), args.categories, args.limit)

    t0 = time.time()
    state = rag_backend.ingest_sessions(
        personas=personas,
        inventory_seed=inventory_seed,
        sessions=sessions,
        model=args.baseline_model,
        top_k=args.rag_top_k,
    )
    qa_results: list[QAResult] = []
    if not args.ingest_only:
        sem = asyncio.Semaphore(max(1, args.qa_concurrency))

        async def _one(q: QA) -> QAResult:
            async with sem:
                answer = await rag_backend.ask(state=state, qa=q)
                return await _grade_one(
                    qa=q, agent_answer=answer, judge_model=args.judge_model
                )

        qa_results = list(await asyncio.gather(*(_one(q) for q in qa)))
    elapsed = time.time() - t0
    qa_agg = aggregate(qa_results)
    pillars = compute_pillars(
        qa_aggregate=qa_agg,
        callback_rate_final={},
        personality_final={},
        session_records=[],
        baseline_qa_aggregate=_load_baseline_aggregate(args.compare_with),
    )
    return {
        "backend": "stateless-rag",
        "elapsed_seconds": elapsed,
        "rag_top_k": args.rag_top_k,
        "personas_summary": {
            m.user_id: {
                "display_name": m.display_name,
                "age": m.age,
                "role": m.role,
                "background": m.background,
            }
            for m in personas.members
        },
        "sessions": [],  # stateless-rag doesn't run a live convo
        "qa": [
            {**_qa_to_meta(q), **asdict(r)}
            for q, r in zip(qa, qa_results)
        ],
        "qa_aggregate": qa_agg,
        "pillars": [p.to_dict() for p in pillars],
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_run_json(path: Path, result: dict) -> None:
    """Single JSON document so the viewer can fetch it as one resource."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m sonzai_razer_bench",
        description="Razer customer-companion benchmark for the Sonzai Mind Layer.",
    )
    p.add_argument(
        "--backend",
        choices=("sonzai", "baseline", "mempalace", "stateless-rag"),
        default="sonzai",
        help="Which memory backend to evaluate. `baseline` = stateless full-"
        "history-in-prompt (upper bound of context-stuffing). `stateless-rag` "
        "= top-K embedded retrieval over sessions (cheapest off-the-shelf "
        "RAG). `mempalace` = verbatim drawer retrieval. `sonzai` = the Mind "
        "Layer.",
    )
    p.add_argument(
        "--categories",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated subset of QA categories to run "
        "(e.g. inventory,kb-relationship). Default: all.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of QAs (after category filter).",
    )
    p.add_argument(
        "--ingest-only", action="store_true",
        help="Run conversation simulation only, skip QA.",
    )
    p.add_argument(
        "--skip-simulation", action="store_true",
        help="(sonzai backend) Skip the live conversation phase. Useful "
        "when re-running QA against an agent whose memory is already populated.",
    )
    p.add_argument(
        "--skip-seed-inventory", action="store_true",
        help="(sonzai backend) Skip seeding inventory items. Useful on re-runs "
        "where the agent already has the seed data.",
    )
    p.add_argument(
        "--reset-memory", dest="reset_memory", action="store_true", default=True,
        help="(sonzai backend) Call memory.reset for every household user_id "
        "before any ingest. Default ON for clean comparison runs.",
    )
    p.add_argument(
        "--no-reset-memory", dest="reset_memory", action="store_false",
        help="Skip the per-user memory.reset before ingest (use for resuming).",
    )
    p.add_argument(
        "--advance-time", dest="advance_time", action="store_true", default=True,
        help="(sonzai backend) Advance simulated time between sessions, computed "
        "from real timestamps in sessions.json. Critical for CE consolidation. "
        "Default ON.",
    )
    p.add_argument(
        "--no-advance-time", dest="advance_time", action="store_false",
        help="Skip advance_time between sessions (debug / fast-iteration only).",
    )
    p.add_argument(
        "--advance-time-min-hours", type=float, default=0.0,
        help="Floor on the per-gap advance_time call (hours). Use 25.0 to mirror "
        "the existing sotopia bench's flat default. Default 0 = use real deltas.",
    )
    p.add_argument(
        "--snapshot-at",
        default="",
        help="Comma-separated 1-based session indices to run a QA pass at. "
        "Example: --snapshot-at 10,20,30 ingests sessions 1..10 → asks all QAs, "
        "then ingests 11..20 → asks all QAs, then 21..30 → asks all QAs. "
        "Output JSON gets a `snapshots` array with one entry per cut-point. "
        "Empty (default) means no trajectory; QA runs once at the end.",
    )
    p.add_argument(
        "--skip-turn-judge", action="store_true",
        help="(sonzai backend) Don't run the per-turn LLM judge. "
        "Faster + cheaper; loses the per-turn score data the viewer renders.",
    )
    p.add_argument(
        "--qa-concurrency", type=int, default=4,
        help="Max concurrent QAs in flight (default 4).",
    )
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument(
        "--baseline-model", default="gemini-3.1-flash-lite-preview",
        help="Model for the baseline / stateless-rag / mempalace backends.",
    )
    p.add_argument(
        "--rag-top-k", type=int, default=5,
        help="(stateless-rag backend) Top-K sessions retrieved per question. "
        "Lower = harder for the baseline, higher = easier. Default 5.",
    )
    p.add_argument(
        "--compare-with", type=Path, default=None,
        help="Path to a prior result JSON (typically a stateless baseline) "
        "to use as the comparison input for AVA-readiness pillar P2 "
        "(differentiation from a stateless LLM). If omitted, P2 is reported "
        "as requiring a comparison input rather than silently passing.",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.backend == "sonzai" and not os.environ.get("SONZAI_API_KEY"):
        print("error: SONZAI_API_KEY must be set for --backend sonzai", file=sys.stderr)
        return 2
    if not os.environ.get("GEMINI_API_KEY"):
        print("error: GEMINI_API_KEY must be set", file=sys.stderr)
        return 2

    print(f"Running Razer benchmark [{args.backend}]...", file=sys.stderr)

    if args.backend == "sonzai":
        result = await _run_sonzai(args)
    elif args.backend == "mempalace":
        result = await _run_mempalace(args)
    elif args.backend == "stateless-rag":
        result = await _run_stateless_rag(args)
    else:
        result = await _run_baseline(args)

    ts = time.strftime("%Y%m%d-%H%M%S")
    out = args.output or RESULTS_DIR / f"{args.backend}_{ts}.json"
    _write_run_json(out, result)

    if not args.ingest_only:
        _print_table(result.get("qa_aggregate", {}))
        pillar_dicts = result.get("pillars") or []
        if pillar_dicts:
            pillars = [
                PillarGrade(
                    name=p["name"], title=p["title"], passed=p["passed"],
                    subscores=p.get("subscores", {}), notes=p.get("notes", ""),
                )
                for p in pillar_dicts
            ]
            _print_pillars(pillars)

    print(f"\nElapsed : {result['elapsed_seconds']:.1f}s")
    print(f"Output  : {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return asyncio.run(_amain(args))
