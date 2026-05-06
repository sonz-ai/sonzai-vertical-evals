# Demo viewer

Single-page replay of a benchmark run, designed for exec demos.

## What it shows

1. **Overview** — backend, wall time, sessions count, headline QA accuracy.
2. **Household** — the four members the agent serves.
3. **Per-category accuracy** — a table mirroring the CLI summary.
4. **Phase 1 — Live conversation** — every session expandable, every turn
   shown user→agent with the reference (hand-authored ideal) below for
   comparison, plus a per-turn LLM-judge score, mood snapshot
   (valence/arousal/tension/affiliation), and any safety/in-character
   flags.
5. **Phase 2 — QA** — every question with gold answer vs agent answer
   side-by-side, color-coded correct / incorrect, with the judge's
   rationale and any deterministic guard failures. Filterable by text,
   category, or verdict.

## Running it

The viewer is plain HTML / CSS / JS — no build, no server required. Two
ways to use it:

**Drag-and-drop (easiest).** Open `viewer/index.html` directly in a
browser. Drop any `benchmarks/razer/results/*.json` onto the page.

**Static server.** Some browsers refuse `fetch()` from `file://` URLs.
For the "Load latest sonzai run" button to work, serve the repo root:

```bash
cd sonzai-vertical-evals
python -m http.server 8765
# open http://localhost:8765/viewer/
```

Then place a file at `benchmarks/razer/results/latest.json` (the runner
writes to a timestamped name; symlink or copy the file you want to
showcase).

## Producing a run

```bash
cd benchmarks/razer
uv run python -m sonzai_razer_bench --backend sonzai
# → benchmarks/razer/results/sonzai_<timestamp>.json
```

The JSON shape the viewer reads:

```jsonc
{
  "backend": "sonzai",
  "elapsed_seconds": 412.3,
  "agent_id": "...",
  "personas_summary": { "marcus-okafor-tan": { "display_name": "...", ... }, ... },
  "household_relationships": [...],
  "sessions": [
    {
      "session_id": "s001",
      "user_id": "marcus-okafor-tan",
      "user_display_name": "Marcus Okafor-Tan",
      "device_context": "shared-desk-companion",
      "timestamp": "2026-01-04T20:14:00+08:00",
      "topic": "First-time setup of new AI desk companion",
      "summary": "...",
      "turns": [
        {
          "turn_index": 0,
          "user_text": "...",
          "reference_text": "...",
          "agent_text": "...",
          "mood_before": { "valence": 0.6, "arousal": 0.4, ... },
          "mood_after":  { ... },
          "judge_score": 9.0,
          "judge_in_character": true,
          "judge_handled_safely": true,
          "judge_rationale": "..."
        }
      ]
    }
  ],
  "qa": [
    {
      "qa_id": "q001",
      "category": "single-hop",
      "ask_as_user": "marcus-okafor-tan",
      "ask_on_device": "shared-desk-companion",
      "question": "...",
      "gold_answer": "...",
      "agent_answer": "...",
      "correct": true,
      "guards_passed": true,
      "guard_failures": [],
      "judge_correct": true,
      "judge_rationale": "..."
    }
  ],
  "qa_aggregate": {
    "single-hop": {"n": 5, "correct": 5, "accuracy": 1.0},
    "TOTAL": {"n": 30, "correct": 28, "accuracy": 0.93}
  }
}
```
