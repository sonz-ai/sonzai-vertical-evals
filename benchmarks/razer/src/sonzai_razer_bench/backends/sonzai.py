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
from ._compat import async_inventory, ensure_bench_agent_async

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


async def seed_inventory(
    client: AsyncSonzai, *, agent_id: str, seed: InventorySeed
) -> None:
    """Seed Sonzai inventory state for each user before conversation."""
    inv = async_inventory(client)
    for owner_id, items in seed.items_by_owner.items():
        for item in items:
            try:
                await inv.create_inventory_item(
                    agent_id=agent_id,
                    user_id=owner_id,
                    item_type=item.item_type,
                    label=item.label,
                    properties=item.properties,
                )
            except Exception as e:
                # Best-effort: many seed items will already exist on a re-run.
                logger.debug(
                    "inventory seed item %s skipped: %s", item.item_id, e
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
        await seed_inventory(client, agent_id=agent_id, seed=inventory_seed)
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
