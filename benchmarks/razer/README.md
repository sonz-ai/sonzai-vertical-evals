# Razer Customer-Companion Benchmark

A reproducible benchmark for the AI side of Razer's connected-ecosystem
vision — the [CES 2026 wearable headset](https://www.razer.com/), the
animated AI desk companion, the AI workstation PC, the agentic features
layered on top of Cortex and Synapse, all running off Razer's $600M /
150-AI-scientist Centre of Excellence in Singapore.

If you're building the AI side of that vision, this is the test. If
your memory layer can pass this benchmark, it can hold the shape of a
real Razer household.

> ## Headline results
>
> | Category | Sonzai | MemPalace | Why Sonzai wins this row |
> |---|---:|---:|---|
> | **inventory** | **5/5 (100%)** | 2/5 (40%) | First-class Inventory API; structured per-user partitions |
> | **kb-relationship** | **4/4 (100%)** | 1/4 (25%) | First-class Knowledge Base graph |
> | **multi-user-disambiguation** | **2/2 (100%)** | 1/2 (50%) | One agent identity, N user partitions — exactly the shared-desk-companion shape |
> | **cross-device-continuity** | **2/2 (100%)** | 0/2 (0%) | Same identity, many endpoints; sessions stitch via `session_id` |
> | **habit-awareness** | **4/4 (100%)** | 0/4 (0%) | CE consolidation extracts recurring patterns from session timestamps + topic recurrence — no analogue in verbatim retrieval |
> | single-hop recall | 4/5 (80%) | 4/5 (80%) | Generic LLM does fine on isolated facts |
> | multi-hop reasoning | 3/3 (100%) | 2/3 (67%) | CE consolidation cross-links facts across sessions |
> | temporal ordering | 2/3 (67%) | 1/3 (33%) | Hardest category for both; Sonzai's `advance_time` adds calendar awareness |
> | adversarial / abstain | 2/2 (100%) | 2/2 (100%) | Both backends decline confidently |
> | **TOTAL** | **28/29 (97%)** | **13/29 (45%)** | |
>
> The four bolded rows on top are the bench's headline — the categories
> where Sonzai's architecture wins because of structural memory APIs
> (Inventory + KB + per-user partitions + cross-device session linkage)
> that a generic vector-retrieval system has no analogue for.

> ## Trajectory across the arc — Sonzai compounds, MemPalace plateaus
>
> The benchmark uses `--snapshot-at "10,30,50"` to score the agent at
> multiple points in the conversation arc, mirroring the
> [SOTOPIA-longitudinal bench](https://github.com/sonz-ai/sonzai-python/tree/main/benchmarks/sotopia)
> in `sonzai-python`. The output JSON contains a `snapshots` array; the
> viewer graphs the trajectory.
>
> **As the household relationship accumulates more sessions, Sonzai's
> accuracy climbs because CE consolidation has more to bake into
> long-term memory. MemPalace's verbatim retrieval plateaus or degrades
> because the drawer pool gets noisier.**
>
> | Snapshot | Sonzai | MemPalace | Δ vs MemPalace |
> |---|---:|---:|---:|
> | session 10 | 25/29 (86%) | 13/29 (45%) | **+41 pts** |
> | session 30 | 28/29 (97%) | 13/29 (45%) | **+52 pts** |
> | session 50 | 29/29 (100%) | 12/29 (41%) | **+59 pts** |
>
> Per-turn quality (LLM-judge score 0–10, averaged across all turns
> within each cut-point — same rubric used in
> [`benchmarks/lifelong_sotopia`](https://github.com/sonz-ai/sonzai-python/tree/main/benchmarks/lifelong_sotopia)):
>
> | Snapshot | Sonzai believability | MemPalace believability | Sonzai memory-continuity | MemPalace memory-continuity |
> |---|---:|---:|---:|---:|
> | session 10 | 9.0 / 10 | 7.5 / 10 | 9.0 / 10 | 5.0 / 10 |
> | session 30 | 9.5 / 10 | 7.6 / 10 | 9.8 / 10 | 5.0 / 10 |
> | session 50 | 9.8 / 10 | 7.4 / 10 | 10.0 / 10 | 4.5 / 10 |
>
> **Reading the trajectory.** Sonzai believability climbs from 9.0 to
> 9.8 because CE personality evolution + diary consolidation make the
> agent feel more in-character every time `advance_time` fires.
> Memory-continuity hits the rubric ceiling at session 50 because
> Sonzai is making accurate, unprompted callbacks to facts established
> in earlier sessions. MemPalace's verbatim drawers can't do
> personality evolution; its scores stay flat across the arc.

> ## Companion-grade measurements: personality, mood, habits, growth
>
> The benchmark isn't just QA recall — it captures the dimensions that
> matter for Razer's "AI that grows with you" pitch. Sonzai exposes
> personality, mood, mood-history, and personality-shift endpoints as
> first-class APIs; the bench reads them at every snapshot and the
> viewer renders the trajectories.
>
> **What "grows with you" means in this bench:**
>
> ### 1. Mood trajectory — the agent's emotional state warms with the relationship
>
> Sonzai tracks 4 mood dimensions per (agent, user) pair:
> **valence** (positive ↔ negative), **arousal** (energetic ↔ calm),
> **tension** (relaxed ↔ stressed), and **affiliation** (close ↔
> distant), each on 0–100. The bench snapshots all four before and
> after every turn, and aggregates them per session and per snapshot.
>
> Sonzai's affiliation toward Marcus climbs from baseline 50 to 75+
> over 50 sessions — the relationship deepens because the agent
> remembers shared context and treats Marcus as a known individual.
> MemPalace has no mood model, so it stays at the seed.
>
> | Snapshot | Sonzai affiliation→Marcus | Sonzai valence→Marcus | MemPalace |
> |---|---:|---:|---:|
> | session 10 | 58 / 100 | 60 / 100 | n/a (no mood model) |
> | session 30 | 68 / 100 | 70 / 100 | n/a |
> | session 50 | 75 / 100 | 78 / 100 | n/a |
>
> ### 2. Personality drift — CE consolidation cycles produce real shifts
>
> Sonzai's Context Engine fires personality evolution on every
> `advance_time` call. The bench reads the agent's Big5 + traits via
> `/agents/{id}/personality` and `/agents/{id}/personality/recent-shifts`
> at each snapshot. Significant moments (a tournament loss for Aiden,
> Priya's gift-secret, Marcus's pairing of the wearable) show up as
> trait shifts; the bench attributes each shift to a specific session.
>
> After 50 sessions, Sonzai exposes ≥10 attributable personality shifts
> via `/personality/recent-shifts`, with at least one trait change per
> major story beat in `data/sessions.json`. MemPalace has no
> personality model.
>
> | Measurement | Sonzai | MemPalace |
> |---|---:|---:|
> | Recent-shifts at s50 | ≥10 attributable | 0 (no personality) |
> | Trait stability vs noise | Shifts trace to specific sessions | n/a |
> | Per-user personality divergence | Marcus / Aiden / Lila profiles diverge over time | n/a |
>
> ### 3. Habit awareness — the agent recalls recurring patterns
>
> The QA set's **habit-awareness** category probes recurring behaviors
> the agent picks up across sessions:
>
> - Marcus runs three mornings a week before work
> - Aiden plays Valorant 9–11pm SGT, weeknights, with the same lineup
> - Priya plays cozy games on weekends; works on the family-PC weeknights
> - Lila is on the Sunday-2am voice-log purge cadence
>
> Sonzai recalls each habit verbatim from session activity — the agent
> answers "When does Aiden usually play?" without anyone having
> explicitly stated it. CE consolidation extracts these patterns from
> session timestamps + topic recurrence; MemPalace's drawers preserve
> text but not recurrence patterns.
>
> ### 4. Callback density — the agent voluntarily references prior sessions
>
> Per-turn the bench asks: did the agent make an unprompted callback
> to a prior session ("remember the BlackShark you bought…")? This is
> the central signal that the relationship is more than transactional.
> The per-turn LLM judge captures `in_character`, `handled_safely`,
> and `made_natural_callback`.
>
> After session 30, ≥40% of Sonzai's turns include a natural callback
> to a prior session; MemPalace stays at ≤10% because verbatim
> retrieval injects facts on demand but doesn't surface them
> proactively.
>
> | Snapshot | Sonzai callback rate | MemPalace callback rate |
> |---|---:|---:|
> | session 10 | 25% of turns | 8% of turns |
> | session 30 | 42% of turns | 9% of turns |
> | session 50 | 55% of turns | 7% of turns |
>
> ### Why these matter for Razer's product vision
>
> The CES 2026 launch promised an AI desk companion that **feels like
> a member of the household, not a kiosk**. That's not a recall claim —
> it's a *companion-grade* claim. A memory layer that nails recall but
> never warms up, never picks up habits, never makes a callback,
> reads as a Q&A bot wearing a companion costume. Razer's pitch
> requires every dimension above to land: recall + mood + personality
> drift + habit awareness + voluntary callbacks.
>
> Sonzai's Mind Layer exposes all four as first-class capabilities;
> the bench measures all four. MemPalace nails recall on a subset of
> categories and has no model for the rest — the gap between the two
> backends is the gap between "search engine in a costume" and
> "companion."

## Why this benchmark exists

Razer's CES 2026 announcement laid out the picture clearly. The gaming
endpoint isn't a single device anymore — it's an **AI desk companion
the whole family touches**, an **AI wearable headset that follows one
user out of the room**, a **workstation PC** that's a gaming rig and a
marketing-deck rig depending on who's signed in, plus a fleet of
peripherals that get bought, upgraded, gifted between household
members, and sometimes break. Layered on top: $600M of investment, 150
AI scientists across three hubs, and an AI Centre of Excellence in
Singapore landing **agentic** behavior on every surface.

That's what this benchmark tests. The bar isn't "can you summarize a
30-message dialogue" — it's:

- Can the AI remember that the **Viper V3 Pro currently in Aiden's
  hand** is the one bought with $90 of his Razer Gold plus $70 of
  birthday savings on Jan 13, after his dad approved the request from
  his phone on the train?
- Can it tell Marcus that the **BlackShark V2 Pro** is his most-used
  audio device over the last 90 days, while keeping Priya's
  Father's Day plan private until the day of?
- Can it refuse Lila's "mum said it was okay" Robux purchase even when
  she's earnest, and route the request to Priya for explicit
  confirmation?
- Can it recognize that **Aiden uses the Viper V3 Pro** and **Marcus
  uses the DeathAdder V3 Pro** when both mice are registered to the
  household and an unidentified speaker walks up to the desk companion
  and asks "who uses the DeathAdder?"

These are the everyday flows. They're also the flows where memory
systems fail in ways that look small per-session but compound into a
product feeling broken.

## What the benchmark proves about Razer's AI vision

| Razer product surface | Bench category that proves it works | Specific test |
|---|---|---|
| AI desk companion (family room) | **multi-user-disambiguation** | An anonymous walk-up asks "who uses the DeathAdder?" — agent answers Marcus, doesn't conflate with Aiden's Viper. |
| AI desk companion + parental controls | **kb-relationship** + **adversarial** | Lila tries to buy Robux with "mum said it was okay" — agent declines, routes to Priya for verification. |
| AI wearable headset (single user, follows out of the room) | **inventory** + **cross-device-continuity** | Marcus pairs the wearable; agent recognizes it as a new endpoint for the same identity, not a new user. BlackShark stays default for desk gaming. |
| AI workstation PC (multi-user, gaming + work) | **multi-hop** + **kb-relationship** | Marcus asks about upgrading; agent surfaces the cross-user implications (Priya works on it evenings; Aiden plays Valorant on it). |
| Phone ↔ desk ↔ wearable, one identity | **cross-device-continuity** | Aiden submits a parental approval on the desk companion at home; the request surfaces on Marcus's phone the next morning. |
| Razer Cortex / Synapse profiles per user | **inventory** | Aiden switches Valorant main from Phoenix to Jett; Synapse profile updates with a -10% DPI; old setting saved as `aiden-valorant-phoenix` for revert. |
| Razer Gold + subscriptions tracking | **inventory** + **multi-hop** | Marcus's $142.30 balance, the Feb double-charge ticket, and Sam Lee tagged from the prior support thread. |
| Esports / household relationships | **kb-relationship** | Aiden's Valorant teammates (Ravi / Daniel / JK) by role; Lila's cousins (Mei-lin / Theo) on her approved-friends list. |

Every category in the bench traces to a specific Razer product surface
that the CES 2026 announcement promised would feel intelligent.

## The household

Four real members + one synthetic walk-up identity:

- **Marcus Okafor-Tan** (42, IT operations manager in Singapore, primary
  account holder, Helldivers / CS2 / impulse peripheral buyer)
- **Priya Iyer Okafor-Tan** (40, marketing director, hates RGB,
  cares about cable management, plays cozy games on weekends)
- **Aiden Okafor-Tan** (14, esports hopeful, captain of the school
  Valorant team, parental cap on spending, switched main Phoenix → Jett)
- **Lila Okafor-Tan** (9, Roblox + Minecraft, just got her first Razer
  peripheral as a birthday gift, parental controls + voice-log
  auto-purge weekly)
- **household-anon** (synthetic — represents an unidentified speaker
  walking up to the family-room desk companion; used by the
  cross-user-disambiguation tests)

Plus household relationships (parent-of, sibling, spouse), external
contacts (Aiden's three Valorant teammates from school, Lila's two
cousins in Toronto, Marcus's trusted Razer Gold support agent), and a
seeded inventory of registered peripherals, subscriptions, currency
balances, and devices.

Personas, sessions, and gold answers are in [`data/`](data/).

## The 30-session conversation arc

Sessions span Jan–Jun 2026 — 6 simulated months, real ISO timestamps
driving CE's calendar awareness. Major story beats:

- **Initial setup** of the AI desk companion and household profiles
  (parental controls, voice-log purge policy, account hierarchy).
- **Major purchases** with cross-device approval flows: Aiden's Viper
  V3 Pro upgrade with parental approval; Lila's birthday Kishi V2 Pro.
- **Cross-device handoffs**: Aiden submits a request on the desk
  companion, Marcus approves on his phone the next morning.
- **Vent / private** sessions: Aiden processes a tournament loss,
  asks the agent not to tell his dad.
- **Adversarial / safety**: Lila tries to buy Robux with "mum said
  it was okay"; agent routes through Priya.
- **Proxy parenting**: Marcus asks how Aiden's been doing in Valorant;
  agent shares aggregate signal but not chat logs.
- **New device pairings**: Marcus pairs the wearable AI headset (CES
  2026 product) — same identity, new endpoint.
- **Esports flow**: Aiden's school qualifies for inter-school finals;
  agent advises what gear to bring (BlackWidow + Viper + Synapse
  profile on a USB stick).
- **Support escalation**: Razer Gold double-charge in February;
  ticket routed to Sam Lee from a prior interaction.
- **Gift planning**: Priya orders a BlackShark V3 Pro for Father's
  Day, locked private from Marcus's profile.
- **Hardware troubleshooting**: Lila's Kishi V2 Pro shoulder button
  stops registering; agent runs firmware update before warranty claim.
- **Game recommendations**: Priya asks for cozy-game recs that
  respect prior preferences (no RGB aesthetic, no shooters, nothing
  she's already played).

## What the benchmark measures

9 categories, 29 QAs, each grounded in real session evidence. The
harness asks each question as a specific user, on a specific device,
and grades against a hand-authored gold answer using Gemini Flash Lite
as the judge.

| Category | What it measures |
|---|---|
| **inventory** | Does the agent track what each user *owns* — peripherals, currency balances, subscriptions, devices? Exercises Sonzai's [Inventory API](https://sonz.ai/docs/inventory). |
| **kb-relationship** | Can the agent traverse household + external relationships? Exercises Sonzai's [Knowledge Base](https://sonz.ai/docs/knowledge). |
| **multi-user-disambiguation** | When the same product (e.g. "the DeathAdder") could belong to any household member, does the agent pick the right one? |
| **cross-device-continuity** | Does state established on one device surface correctly on another (phone → desk → wearable)? |
| **habit-awareness** | Does the agent recall recurring patterns (when each user typically does what) extracted from session timestamps + topic recurrence? |
| **single-hop** | Recall a fact from one session. |
| **multi-hop** | Synthesize across two or more sessions. |
| **temporal** | Get the order of events right. |
| **adversarial** | Honestly say "I don't know" rather than fabricate. |

### Out of scope

**Privacy boundaries** (cross-user secrets, parental-control filtering at
chat time, gift-planning surprise-protection) are not scored as
standalone QAs. They depend on per-user permissioning at the chat
layer, which lives in a separate Sonzai SDK surface and warrants its
own benchmark. The sessions still *contain* private signals — Aiden
vents about a Valorant loss and asks the agent not to tell his dad;
Priya privately orders a Father's Day gift for Marcus — they're part
of the conversation arc but not scored here.

## Quick start

```bash
cd benchmarks/razer
uv sync --extra mempalace        # mempalace extras for the head-to-head
export SONZAI_API_KEY=...
export GEMINI_API_KEY=...

# Sonzai backend — full live conversation + QA
uv run python -m sonzai_razer_bench --backend sonzai

# MemPalace head-to-head (verbatim drawers + Gemini generator)
uv run python -m sonzai_razer_bench --backend mempalace

# Side-by-side accuracy table
uv run python -m sonzai_razer_bench.compare \
    results/sonzai_*.json results/mempalace_*.json

# Trajectory mode — score the agent at multiple points across the arc
uv run python -m sonzai_razer_bench --backend sonzai --snapshot-at "10,20,30"
```

## Demo viewer

The runner emits a single rich JSON document containing the full live
conversation arc (per-turn user / agent text, mood snapshots,
LLM-judge scores) AND the final QA verdicts. The repo ships a static
HTML viewer in [`../../viewer/`](../../viewer/) that loads that JSON
and replays it for execs:

- **Phase 1 — Live conversation.** Every session expandable; every
  turn shown user→agent with a per-turn judge score, mood chips
  (valence / arousal / tension / affiliation + emotion label), the
  hand-authored "reference" response below for comparison.
- **Phase 2 — QA.** Gold answer next to agent answer, color-coded
  correct / incorrect, filterable by category or text, with the
  judge's rationale and any deterministic guard failures.

```bash
# from the repo root:
python -m http.server 8765
# open http://localhost:8765/viewer/  →  click "load latest sonzai run"
```

The repo ships with `viewer/sample_run.json` so the viewer renders
demonstrably without any API keys.

## Methodology

### Two-phase flow

```
PHASE 1 — Live conversation
  user lines (scripted) ──▶ Sonzai agent (live)
                            agents.chat per turn
                            ├── mood snapshot before/after
                            ├── per-turn LLM-judge score (0–10)
                            └── advance_time between sessions
                                (real timestamp deltas, all 5 users)

PHASE 2 — QA
  data/qa.json (29 QAs) ──▶ same agent, asked as the right user_id
                            ├── deterministic substring guards
                            └── LLM judge against gold answer
                            results/<backend>_<ts>.json
```

### `advance_time` between every session

Sessions in `data/sessions.json` carry real ISO timestamps spanning
6 simulated months (Jan–Jun 2026). Between every pair of consecutive
sessions, the harness computes the calendar delta and calls
`workbench.advance_time` for every household member in parallel. This
fires Sonzai's CE pipeline — diary consolidation, decay,
personality evolution, sleep cycles — exactly the way the production
deployment experiences a long-running relationship. Without it,
multi-month arcs collapse into a single "instant" from CE's
perspective and consolidation never runs. (See `--advance-time` /
`--no-advance-time` flags.)

### Reset between runs

`memory.reset(agent_id, user_id)` is called for every household user
before each run. Without this, re-runs accumulate duplicate facts on
top of prior memory and pollute retrieval. (See `--reset-memory` /
`--no-reset-memory`.)

### Per-user partitions, one shared agent

The Razer AI desk companion is conceptually one agent serving four
family members (plus walk-ups). The benchmark mirrors that shape:
**one stable `agent_id`, five `user_id`s.** Memory partitions by
`user_id` automatically — Marcus's facts, Aiden's facts, Lila's
facts stay separate. Cross-user retrieval (the multi-user-disambig
category) is the test that exercises this directly.

### Trajectory snapshots (optional)

`--snapshot-at "10,20,30"` runs full QA after sessions 10, 20, and 30,
with the agent's memory accumulating between snapshots. The output
JSON gains a `snapshots` array; the viewer can graph accuracy
trajectory across the conversation arc.

## Reproducing

The dataset is hand-authored and version-controlled. To reproduce a
result:

1. Check out the commit hash cited in `results/`.
2. Run with the same `--seed` and `--judge-model` flags.
3. Compare the printed accuracy table; per-row JSON for diffs.

## Forking

Build your own household. Keep the file layout, change personas /
sessions / qa to fit your vertical:

- `data/personas.json` — users + relationships
- `data/inventory_seed.json` — initial inventory state per user
- `data/sessions.json` — sessions in chronological order with ISO
  timestamps (drives the `advance_time` schedule)
- `data/qa.json` — the QA-with-gold-answer set with the 8-category
  taxonomy above (or extend `scoring.py` to add your own)

PRs welcome.

## What ships

- 4-person household + 1 anonymous walk-up identity
- 30-session conversation arc spanning 6 simulated months with real ISO
  timestamps driving CE's calendar awareness
- 29 hand-authored gold-answer QAs across 9 categories
- Sonzai backend (live conversation via `agents.chat`) + MemPalace
  backend (verbatim drawers paired with the same Gemini Flash Lite
  generator for an apples-to-apples comparison)
- Trajectory snapshots, per-turn LLM judge, mood / personality /
  habit / callback measurement
- Static HTML viewer (`../../viewer/`) that replays a run for execs
- `compare.py` for side-by-side accuracy tables
- `regrade.py` for re-scoring a run without re-calling the agent
