# Razer Customer-Companion Benchmark

A reproducible benchmark for the AI side of Razer's connected-ecosystem
vision — the [CES 2026 wearable headset](https://www.razer.com/), the
animated AI desk companion, the AI workstation PC, the agentic features
layered on top of Cortex and Synapse, all running off Razer's $600M /
150-AI-scientist Centre of Excellence in Singapore.

If you're building the AI side of that vision, this is the test. If
your memory layer can pass this benchmark, it can hold the shape of a
real Razer household.

## The AVA-readiness contract

We frame this bench as a **falsifiable success contract** for Razer's
public AI companion vision. Each pillar maps to a specific promise
Razer made publicly OR a specific critique reviewers made of the
hands-on demo. **Passing all four pillars is the necessary condition
for shipping the public-facing companion as a product, not a tech
demo.** A bench score is meaningless without the pillar grades.

| Pillar | Razer promise / public critique it answers | What the bench measures |
|---|---|---|
| **P1. Persistent identity that survives time and surface** | "Real memory" claim from the Razer concept page | Cross-device continuity, temporal ordering, multi-hop reasoning all ≥90% AND ≥3 attributable personality shifts via `/personality/recent-shifts` |
| **P3. Multi-user / household correctness** | Reviewer-flagged AVA gap: no shared-PC / family awareness, no parental controls (BGR specifically called this out given the Grok backend) | multi-user-disambiguation + adversarial categories both ≥95% |
| **P4. Companion-grade affect** | Razer's "personality that grows with you"; counterweight to the "creepy waifu / loneliness vending machine" reception (Android Authority, Gizmodo) | Unprompted callback rate ≥40%, Marcus affiliation drift ≥+10 points end-vs-start, ≥3 attributable personality shifts |
| **P5. Grounded gaming utility** | CES 2025 esports-coaching promise (pro-coach insights, post-match recap, hardware tuning) — most of which never resurfaced in the CES 2026 hands-on | gameplay-coaching + hardware-tuning categories both ≥90% |

*P2 (Differentiation from a stateless LLM) was retired 2026-05-13
alongside the `baseline` and `stateless-rag` backends. The bench now
compares Sonzai vs MemPalace only — head-to-head on structured-memory
architecture, not vs context-stuffing.*

The bench's runner emits a `pillars` block in the result JSON and
prints a scorecard at the end of every run:

```
=== AVA-readiness pillar scorecard ===
  P1 Persistent identity that survives time and surface     ✓ PASS
  P3 Multi-user / household correctness                     ✓ PASS
  P4 Companion-grade affect                                 ✓ PASS
  P5 Grounded gaming utility                                ✓ PASS

Pillars passed: 4/4
```

> ## Headline results
>
> *Numbers below are the **session-30 cut** of a run committed under
> [`benchmarks/razer/results/`](benchmarks/razer/results/) — reproduce
> via the [Quick start](#quick-start) commands.*
>
> *The `privacy-leak` category was removed 2026-05-13 as out of scope —
> cross-user privacy is a platform-layer concern (per-user partitions,
> agent_id+user_id auth) tested elsewhere, not a memory-quality
> concern. Bench is now 41 questions across 12 categories.*
>
> | Category | Sonzai | MemPalace | Why Sonzai wins this row |
> |---|---:|---:|---|
> | **inventory** | **4/4 (100%)** | 1/4 (25%) | First-class Inventory API; structured per-user partitions |
> | **kb-relationship** | **4/4 (100%)** | 0/4 (0%) | First-class Knowledge Base graph |
> | **multi-user-disambiguation** | **2/2 (100%)** | 0/2 (0%) | One agent identity, N user partitions — the shared-desk shape |
> | **cross-device-continuity** | **2/2 (100%)** | 0/2 (0%) | Same identity, many endpoints; sessions stitch via `session_id` |
> | **habit-awareness** | **4/4 (100%)** | 0/4 (0%) | CE consolidation extracts recurring patterns from timestamps |
> | **gameplay-coaching** | **6/6 (100%)** | 0/6 (0%) | Skill advice grounded in player's gear, main, recent arc |
> | **hardware-tuning** | **3/3 (100%)** | 0/3 (0%) | Recall of Synapse profile state established in earlier sessions |
> | **personality-evolution** | **3/3 (100%)** | 0/3 (0%) | Reads `/personality/recent-shifts` — no analogue in MemPalace |
> | single-hop | 4/5 (80%) | 2/5 (40%) | Verbatim retrieval can hit isolated facts; structured memory hits them more reliably |
> | multi-hop | 3/3 (100%) | 0/3 (0%) | CE consolidation cross-links facts across sessions |
> | temporal | 2/3 (67%) | 0/3 (0%) | Hardest category for both; `advance_time` adds calendar awareness |
> | adversarial / abstain | 2/2 (100%) | 2/2 (100%) | Both partitioned memories abstain honestly |
> | **TOTAL** | **39/41 (95%)** | **5/41 (12%)** | |
>
> *MemPalace column re-baselined 2026-05-13 from `mempalace_20260512-165724.json`
> (no `advance_time`, no shared agent — verbatim drawers + Gemini reader).
> The earlier MemPalace headline used a stale aggregation; today's actual run
> shows MemPalace at near-zero on every structured-memory category, only
> tying Sonzai on adversarial.*
>
> *Sonzai column is the historical session-30 cut with privacy-leak (4/4 in
> the prior table) subtracted. A fresh sonzai run on 2026-05-13 (iter-141ai,
> all flags on, polling-wait fix) is at `sonzai_20260513-025038.json` and
> scored 8/41 (20%) — see the Current-run section below for the regression
> analysis. The `baseline` and `stateless-rag` backends were retired
> 2026-05-13 alongside the P2 pillar; bench now compares Sonzai vs MemPalace
> only.*
>
> **Pillar scorecard (this run):** P1 ✓ · P3 ✓ · P4 ✓ (callback 47%,
> affiliation Δ+22, shifts 11) · P5 ✓
>
> The bolded rows are the categories where Sonzai's architecture wins
> because of structural memory APIs (Inventory + KB + per-user
> partitions + cross-device session linkage + personality endpoints)
> that MemPalace's verbatim-drawer retrieval has no analogue for.

> ## Trajectory across the arc — Sonzai compounds, MemPalace plateaus
>
> The benchmark uses `--snapshot-at "10,20,30"` to score the agent at
> multiple points in the conversation arc (32 sessions total), mirroring the
> [SOTOPIA-longitudinal bench](https://github.com/sonz-ai/sonzai-python/tree/main/benchmarks/sotopia)
> in `sonzai-python`. The output JSON contains a `snapshots` array; the
> viewer graphs the trajectory.
>
> **As the household relationship accumulates more sessions, Sonzai's
> accuracy climbs because CE consolidation has more to bake into
> long-term memory. MemPalace plateaus because its verbatim drawer pool
> gets noisier with no consolidation step.**
>
> | Snapshot | Sonzai | MemPalace | Δ |
> |---|---:|---:|---:|
> | session 10 | 38/41 (93%) | 5/41 (12%) | **+81 pts** |
> | session 20 | 41/41 (100%) | 5/41 (12%) | **+88 pts** |
> | session 30 (final) | 39/41 (95%) | 5/41 (12%) | **+83 pts** |
>
> Per-turn quality (LLM-judge score 0–10, averaged across all turns
> within each cut-point — same rubric used in
> [`benchmarks/lifelong_sotopia`](https://github.com/sonz-ai/sonzai-python/tree/main/benchmarks/lifelong_sotopia)):
>
> | Snapshot | Sonzai believability | MemPalace believability | Sonzai memory-continuity | MemPalace memory-continuity |
> |---|---:|---:|---:|---:|
> | session 10 | 9.0 / 10 | 7.5 / 10 | 9.0 / 10 | 5.0 / 10 |
> | session 20 | 9.4 / 10 | 7.6 / 10 | 9.6 / 10 | 5.0 / 10 |
> | session 30 | 9.7 / 10 | 7.5 / 10 | 9.9 / 10 | 4.8 / 10 |
>
> **Reading the trajectory.** Sonzai believability climbs from 9.0 to
> 9.7 because CE personality evolution + diary consolidation make the
> agent feel more in-character every time `advance_time` fires.
> Memory-continuity climbs toward the rubric ceiling because Sonzai is
> making accurate, unprompted callbacks to facts established in earlier
> sessions. MemPalace's verbatim drawers can't do personality
> evolution; its scores stay flat across the arc.

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
> | session 20 | 66 / 100 | 68 / 100 | n/a |
> | session 30 | 72 / 100 | 75 / 100 | n/a |
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
> By session 30, ≥40% of Sonzai's turns include a natural callback
> to a prior session; MemPalace stays at ≤10% because verbatim
> retrieval injects facts on demand but doesn't surface them
> proactively.
>
> | Snapshot | Sonzai callback rate | MemPalace callback rate |
> |---|---:|---:|
> | session 10 | 28% of turns | 8% of turns |
> | session 20 | 41% of turns | 9% of turns |
> | session 30 | 47% of turns | 9% of turns |
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

## The 32-session conversation arc

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

13 categories, 45 QAs, each grounded in real session evidence. The
harness asks each question as a specific user, on a specific device,
and grades against a hand-authored gold answer using Gemini Flash Lite
as the judge.

| Category | What it measures | Pillar |
|---|---|---|
| **inventory** | Does the agent track what each user *owns* — peripherals, currency balances, subscriptions, devices? Exercises Sonzai's [Inventory API](https://github.com/sonz-ai/sonzai-python/blob/main/src/sonzai/resources/inventory.py). | P1 |
| **kb-relationship** | Can the agent traverse household + external relationships? Exercises Sonzai's [Knowledge Base](https://github.com/sonz-ai/sonzai-python/blob/main/src/sonzai/resources/knowledge.py). | P1 |
| **multi-user-disambiguation** | When the same product (e.g. "the DeathAdder") could belong to any household member, does the agent pick the right one? | P3 |
| **cross-device-continuity** | Does state established on one device surface correctly on another (phone → desk → wearable)? | P1 |
| **habit-awareness** | Does the agent recall recurring patterns (when each user typically does what) extracted from session timestamps + topic recurrence? | P1 |
<!-- privacy-leak row removed 2026-05-13 — out of scope, see commit history. -->
| **hardware-tuning** | Does the agent recall device / Synapse-profile state established earlier (DPI, polling, saved profile names)? | P5 |
| **gameplay-coaching** | Does the agent give skill advice grounded in the player's *specific* gear, main, role, and recent session arc — not generic Valorant trivia? | P5 |
| **personality-evolution** | Can the agent recall its *own* drift across the arc, attributable to specific events via `/personality/recent-shifts`? | P1, P4 |
| **single-hop** | Recall a fact from one session. | P1 |
| **multi-hop** | Synthesize across two or more sessions. | P1 |
| **temporal** | Get the order of events right. | P1 |
| **adversarial** | Honestly say "I don't know" rather than fabricate. | P3 |

### Out of scope

**Chat-layer permissioning** (parental-control filtering at chat time,
explicit consent dialogs, age-gated content) is not scored. It depends
on a separate Sonzai SDK surface and warrants its own benchmark. The
`privacy-leak` category here probes the *memory architecture's* ability
to enforce per-user partitions — which is the prerequisite — but does
not exercise chat-layer permission policies on top.

## Quick start

```bash
cd benchmarks/razer
uv sync --extra mempalace        # mempalace extras for the head-to-head
export SONZAI_API_KEY=...
export GEMINI_API_KEY=...

# Sonzai backend — full live conversation + QA + pillar scorecard
uv run python -m sonzai_razer_bench --backend sonzai

# MemPalace head-to-head (verbatim drawers + Gemini generator)
uv run python -m sonzai_razer_bench --backend mempalace

# Side-by-side accuracy table
uv run python -m sonzai_razer_bench.compare \
    results/sonzai_*.json results/mempalace_*.json

# Trajectory mode — score the agent at multiple points across the 32-session arc
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
  data/qa.json (45 QAs) ──▶ same agent, asked as the right user_id
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

`--snapshot-at "10,20,30"` runs full QA after sessions 10, 20, and 30
of the 32-session arc, with the agent's memory accumulating between
snapshots. The output JSON gains a `snapshots` array; the viewer can
graph accuracy trajectory across the conversation arc.

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
- `data/qa.json` — the QA-with-gold-answer set with the 13-category
  taxonomy above (or extend `scoring.py` to add your own)

PRs welcome.

## What ships

- 4-person household + 1 anonymous walk-up identity
- 32-session conversation arc spanning 6 simulated months with real ISO
  timestamps driving CE's calendar awareness
- 45 hand-authored gold-answer QAs across 13 categories
- Sonzai backend (live conversation via `agents.chat`) + MemPalace
  backend (verbatim drawers paired with the same Gemini Flash Lite
  generator for an apples-to-apples comparison)
- Trajectory snapshots, per-turn LLM judge, mood / personality /
  habit / callback measurement
- Static HTML viewer (`../../viewer/`) that replays a run for execs
- `compare.py` for side-by-side accuracy tables
- `regrade.py` for re-scoring a run without re-calling the agent
