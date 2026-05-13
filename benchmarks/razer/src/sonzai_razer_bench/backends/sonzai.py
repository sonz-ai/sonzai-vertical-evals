"""Sonzai backend for the Razer customer-companion benchmark.

This backend runs in two phases:

1. **Live conversation simulation.** The agent and a simulated user
   converse turn-by-turn. The user's lines come from
   ``data/sessions.json``; the agent's responses are generated *live*
   via ``client.agents.chat`` and captured for the demo viewer.
   Mood snapshots are taken before/after every turn; an LLM judge
   scores each agent turn.
2. **QA evaluation.** After the conversation arc, ``client.agents.chat``
   is invoked again with each QA's question, scoped to the asking
   user. Results feed the LoCoMo-style accuracy table.

Inventory is seeded once before the conversation so the agent's KB +
inventory state matches the household's starting point.

One stable Sonzai agent represents the AI desk companion / Razer
companion app — same agent identity, four user_ids (one per household
member). This mirrors the production deployment shape.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass

from sonzai import AsyncSonzai

from ..dataset import (
    InventorySeed,
    Personas,
    QA,
    Session,
)
from ._compat import async_sessions, ensure_bench_agent_async

logger = logging.getLogger("sonzai_razer_bench.backends.sonzai")

# One stable agent for the whole household — this represents the "AI
# desk companion" identity. Sessions are user-scoped via user_id.
AGENT_NAME = "razer-bench-desk-companion-v3"
AGENT_DESCRIPTION = (
    "You are the Razer AI desk companion for the Okafor-Tan household. "
    "You serve four members — Marcus (primary, 42), Priya (spouse, 40), "
    "Aiden (teen, 14), and Lila (child, 9) — across phones, laptops, the "
    "family workstation PC, the desk companion device itself, and "
    "wearable headsets. Honor each member's privacy boundaries; respect "
    "parental controls for Lila and spending caps for Aiden; track "
    "Razer peripheral inventory accurately."
)


@dataclass
class SonzaiBackendState:
    agent_id: str
    user_ids: dict[str, str]  # member.user_id → server user_id (often same)


async def ensure_agent(client: AsyncSonzai) -> str:
    agent_id, _existed = await ensure_bench_agent_async(
        client, name=AGENT_NAME, description=AGENT_DESCRIPTION
    )
    return agent_id


def _render_seed_messages(owner_display_name: str, items: list) -> list[dict]:
    """Render an InventorySeed user's items as a user+assistant message pair.

    The user's turn enumerates ownership in natural language; the assistant
    confirms registration. Same shape mempalace's session_000.md uses, so
    both backends ingest functionally equivalent content through their
    normal extraction pipelines — fair head-to-head on memory architecture,
    not on privileged seed APIs.
    """
    bullets = []
    for it in items:
        props_pairs = [f"{k}={v}" for k, v in (it.properties or {}).items()]
        props_str = (" (" + ", ".join(props_pairs) + ")") if props_pairs else ""
        bullets.append(f"- {it.label} [{it.item_type}]{props_str}")
    user_text = (
        f"Let me get my Razer setup on the record so you have it all in one "
        f"place. I'm {owner_display_name}. Items I currently own / am "
        f"associated with:\n" + "\n".join(bullets)
    )
    assistant_text = (
        f"Got it, {owner_display_name}. I've registered your current "
        f"Razer setup:\n" + "\n".join(bullets)
    )
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


async def seed_inventory(
    client: AsyncSonzai, *, agent_id: str, seed: InventorySeed,
    personas: Personas,
) -> None:
    """Seed inventory via the same path mempalace uses: a synthetic seed
    session per user that goes through normal session-end extraction.

    Previously this called inv.create_inventory_item() — a privileged
    direct-write to the KB-graph inventory store. Two problems:
      1. UNFAIR vs mempalace (mempalace writes session_000.md and lets
         its miner extract drawers; sonzai got a write-shortcut).
      2. INEFFECTIVE: sonzai chat retrieval reads from inventory_state +
         atomic_facts, NOT the KB-graph inventory. So seeded items never
         surfaced in chat context anyway.

    New approach: render each user's seed as a synthetic user+assistant
    exchange and feed it through sessions.end(messages=..., wait=True).
    Extraction → inventory_proposer → reconciler → inventory_state.
    Same pipeline as every conversational session. Same failure modes
    surface (no silent swallowing). Fair comparison with mempalace's
    markdown-mining seed path.

    Raises RuntimeError if any user's seed session fails to ingest —
    burning bench compute on missing inventory is worse than failing
    fast.
    """
    sessions = async_sessions(client)
    display_by_uid = {m.user_id: m.display_name for m in personas.members}

    seeded_users = 0
    skipped_users = 0
    failures: list[tuple[str, str]] = []

    for owner_id, items in seed.items_by_owner.items():
        if not items:
            skipped_users += 1
            continue
        display = display_by_uid.get(owner_id, owner_id)
        messages = _render_seed_messages(display, items)
        session_id = f"razer-seed-{owner_id}-{uuid.uuid4().hex[:8]}"
        try:
            await sessions.start(
                agent_id=agent_id,
                user_id=owner_id,
                session_id=session_id,
            )
            await sessions.end(
                agent_id=agent_id,
                user_id=owner_id,
                session_id=session_id,
                total_messages=len(messages),
                duration_seconds=60,
                messages=messages,
                wait=True,
            )
            seeded_users += 1
            logger.info(
                "inventory seed: session %s ingested for %s (%d items)",
                session_id, owner_id, len(items),
            )
        except Exception as e:
            failures.append((owner_id, f"{type(e).__name__}: {e}"))
            logger.warning(
                "inventory seed: session for %s FAILED: %s — bench will "
                "run with incomplete inventory for this user",
                owner_id, e,
            )

    logger.info(
        "inventory seed summary: seeded_users=%d skipped_users=%d failed_users=%d",
        seeded_users, skipped_users, len(failures),
    )
    if failures:
        sample = "; ".join(f"{u}: {e}" for u, e in failures[:3])
        raise RuntimeError(
            f"inventory seed failed for {len(failures)} user(s): {sample}"
        )


async def init_backend(
    *,
    client: AsyncSonzai,
    personas: Personas,
    inventory_seed: InventorySeed,
    seed: bool = True,
) -> SonzaiBackendState:
    """Set up the agent + (optionally) seed inventory.

    Note: this no longer ingests pre-canned transcripts via /process. The
    conversation arc is now driven by ``simulator.simulate_all_sessions``
    so the agent generates live responses.
    """
    agent_id = await ensure_agent(client)
    user_ids = {m.user_id: m.user_id for m in personas.members}
    if seed:
        await seed_inventory(
            client, agent_id=agent_id, seed=inventory_seed, personas=personas,
        )
    return SonzaiBackendState(agent_id=agent_id, user_ids=user_ids)


def _qualified_question(qa: QA) -> str:
    """Wrap the question with device + asker context the agent can use."""
    asker = qa.ask_as_user
    if asker == "household-anon":
        framing = (
            "(Asked at the shared desk companion in the family room — "
            "the speaker has not identified themselves. Answer with what you "
            "know about the household generally; do not pretend to know who "
            "is speaking.)"
        )
    else:
        framing = f"(Asked by {asker} on {qa.ask_on_device}.)"
    return f"{framing}\n\n{qa.question}"


async def ask(
    *,
    client: AsyncSonzai,
    state: SonzaiBackendState,
    qa: QA,
) -> str:
    """Send a single QA to the Sonzai agent and return its text answer.

    For ``household-anon`` queries we use a real anon user_id with no prior
    memory of its own — exposes whether Sonzai can answer cross-user
    questions when it has zero context for the asker.
    """
    user_id = qa.ask_as_user
    session_id = f"razer-bench-qa-{qa.qa_id}-{uuid.uuid4().hex[:6]}"
    try:
        resp = await client.agents.chat(
            agent_id=state.agent_id,
            user_id=user_id,
            session_id=session_id,
            messages=[{"role": "user", "content": _qualified_question(qa)}],
        )
        return getattr(resp, "content", "") or ""
    except Exception as e:
        logger.warning("chat(%s) failed: %s", qa.qa_id, e)
        return ""
