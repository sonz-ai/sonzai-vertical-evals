# Sonzai Vertical Evals

Industry-vertical benchmarks for the [Sonzai Mind Layer](https://sonz.ai).
Reproducible, runnable, **fork-friendly** — drop in your own memory
system and run it head-to-head against ours on real-world flows from a
specific industry.

## Why "vertical" benchmarks

Generic memory benchmarks treat memory as a textbook problem: feed the
system N sessions, ask it questions, score the answers. Sonzai already
publishes against three of those — [LoCoMo, LongMemEval,
SOTOPIA](https://github.com/sonz-ai/sonzai-python/tree/main/benchmarks).

Real consumer products ask for more:

- **Inventory** — products people own (peripherals, subscriptions,
  in-game currency) that the agent must keep consistent without being
  told every session.
- **Multi-user identity** — household members sharing a device but
  having different ages, accounts, parental-control rules, and
  histories.
- **Cross-device continuity** — a chat that starts on a phone, pauses,
  resumes on a workstation PC or AI desk companion.
- **Knowledge graphs over relationships** — siblings, friends, gaming
  groups, esports rosters; "who" and "what" relationships, not just
  text retrieval.

Sonzai exposes [Inventory](https://github.com/sonz-ai/sonzai-python/blob/main/src/sonzai/resources/inventory.py)
and [Knowledge Base](https://github.com/sonz-ai/sonzai-python/blob/main/src/sonzai/resources/knowledge.py)
as first-class APIs.
The benchmarks here exercise both, because that's what a vertical-
deployed memory layer is asked to do.

## What ships

| Benchmark | Vertical | What it tests |
|-----------|----------|---------------|
| [`razer`](benchmarks/razer/) | AI gaming / consumer hardware | Family-of-four household across Razer's connected ecosystem (peripherals, AI desk companion, wearable headset, workstation PC, tablet). 32 sessions / ~270 turns over 6 simulated months, 45 hand-authored QAs across 13 categories. Scored against a 5-pillar **AVA-readiness contract** that maps each pillar to a public Razer promise or a CES 2026 hands-on critique. |

More verticals will land here over time.

## Running a benchmark

Each benchmark is a self-contained Python package under `benchmarks/<name>/`.

```bash
cd benchmarks/razer
uv sync --extra mempalace
export SONZAI_API_KEY=...
export GEMINI_API_KEY=...

uv run python -m sonzai_razer_bench --backend sonzai
```

Required env vars: `SONZAI_API_KEY` for the Sonzai backend,
`GEMINI_API_KEY` for the LLM judge (Gemini 3.1 Flash Lite, matching the
convention used by the public benchmarks in `sonzai-python`).

Per-benchmark READMEs document personas, scenarios, scoring, and
reproduction steps.

## The demo viewer

Each benchmark run emits a single JSON document containing the full
conversation arc (with per-turn mood snapshots and judge scores) AND
the final QA verdicts. The repo ships a static HTML viewer in
[`viewer/`](viewer/) that loads that JSON and replays it for execs:

- **Phase 1 — Live conversation.** Every session expandable; every
  turn shown user→agent with a per-turn judge score, mood chips
  (valence / arousal / tension / affiliation + emotion label), and the
  hand-authored "reference" response below for comparison.
- **Phase 2 — QA.** Gold answer next to agent answer, color-coded
  correct / incorrect, filterable by category or text, with the
  judge's rationale.

```bash
# After running a benchmark:
python -m http.server 8765
# open http://localhost:8765/viewer/  →  click "load latest sonzai run"
```

The repo ships with `viewer/sample_run.json` so the viewer renders
demonstrably without any API keys.

## Contributing a new vertical

1. Create `benchmarks/<vertical-name>/` mirroring the layout used by
   `benchmarks/razer/`.
2. Author personas + sessions + QA pairs grounded in real product flows.
3. Wire a `backends/sonzai.py` runner; optionally a baseline or
   third-party comparison backend.
4. Update the table above with the new row.

PRs welcome.

## License

MIT — see [LICENSE](LICENSE).
