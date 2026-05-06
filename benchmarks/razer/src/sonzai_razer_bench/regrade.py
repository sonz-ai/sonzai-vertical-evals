"""Re-grade an existing run JSON against the current scoring rules.

Useful when the guard semantics change (e.g. must_mention switched from
hard to soft) and you don't want to re-run the agent — the agent answers
are already on disk; just rewrite the verdicts.

Usage::

    python -m sonzai_razer_bench.regrade results/sonzai_*.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .dataset import QA
from .scoring import (
    DEFAULT_JUDGE_MODEL,
    aggregate,
    check_guards,
    llm_judge_async,
    QAResult,
)


def _qa_from_row(row: dict) -> QA:
    return QA(
        qa_id=row["qa_id"],
        category=row["category"],
        ask_as_user=row["ask_as_user"],
        ask_on_device=row["ask_on_device"],
        question=row["question"],
        gold_answer=row["gold_answer"],
        evidence_sessions=tuple(row.get("evidence_sessions", [])),
        # The original QA file had must_mention/must_not_mention; the run
        # JSON doesn't capture those. We can't re-grade with full guard
        # context unless we cross-reference data/qa.json. Loaded below.
    )


def _load_qa_guards() -> dict[str, QA]:
    """Cross-reference the run row → guard fields stored in data/qa.json."""
    from .dataset import load_qa
    return {q.qa_id: q for q in load_qa()}


async def _grade_one(
    *, qa: QA, agent_answer: str, judge_model: str
) -> QAResult:
    guards = check_guards(qa, agent_answer)
    judge_correct = False
    rationale = ""
    # Run the LLM judge regardless of soft-guard misses; only hard guard
    # failures (must_not_mention) short-circuit.
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
        rationale = "; ".join(guards.hard_failures)
    correct = guards.passed and judge_correct
    # Annotate rationale with soft-guard hints when present
    if guards.soft_failures:
        rationale = (rationale + " | " if rationale else "") + (
            "; ".join(guards.soft_failures)
        )
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


async def regrade(path: Path, *, judge_model: str = DEFAULT_JUDGE_MODEL) -> None:
    with open(path) as f:
        doc = json.load(f)

    qa_specs = _load_qa_guards()
    new_rows: list = []
    new_results: list[QAResult] = []
    for row in doc.get("qa", []):
        qa = qa_specs.get(row["qa_id"])
        if qa is None:
            print(f"  ! skipping unknown qa_id: {row['qa_id']}")
            continue
        result = await _grade_one(
            qa=qa,
            agent_answer=row.get("agent_answer", ""),
            judge_model=judge_model,
        )
        new_results.append(result)
        # Rebuild the row preserving the run-time meta but with refreshed verdict.
        row["correct"] = result.correct
        row["guards_passed"] = result.guards_passed
        row["guard_failures"] = result.guard_failures
        row["judge_correct"] = result.judge_correct
        row["judge_rationale"] = result.judge_rationale
        new_rows.append(row)

    doc["qa"] = new_rows
    doc["qa_aggregate"] = aggregate(new_results)

    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=str)
    total = doc["qa_aggregate"].get("TOTAL", {})
    print(
        f"  regraded {path.name}: "
        f"{total.get('correct', 0)}/{total.get('n', 0)} = "
        f"{(total.get('accuracy', 0) * 100):.0f}%"
    )


async def main_async(paths: list[Path], judge_model: str) -> int:
    for p in paths:
        await regrade(p, judge_model=judge_model)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sonzai_razer_bench.regrade")
    p.add_argument("inputs", type=Path, nargs="+", help="Run JSON paths to re-grade.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    args = p.parse_args(argv)
    return asyncio.run(main_async(args.inputs, args.judge_model))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
