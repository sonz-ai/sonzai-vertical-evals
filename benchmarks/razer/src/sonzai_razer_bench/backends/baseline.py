"""No-memory baseline backend.

Stuffs the full personas, inventory seed, and every session's transcript
into the prompt at QA time and asks Gemini Flash Lite to answer. This is
the pure-context-window upper bound — what you'd get without a memory
layer at all. Useful for telling whether Sonzai's lift comes from
structured memory (inventory + KB) or just from the LLM being good.

Note: this baseline ignores ``ask_as_user`` privacy boundaries by design —
it has no concept of who is asking; it just sees all the data. So the
privacy-boundary category is expected to be a near-floor result here.
That contrast is the demonstration: a memory layer that knows *who*
asked is structurally different from full-history-in-prompt.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from google import genai
from google.genai import types as gt
from pydantic import BaseModel

from ..dataset import (
    InventorySeed,
    Personas,
    QA,
    Session,
)

logger = logging.getLogger("sonzai_razer_bench.backends.baseline")

DEFAULT_MODEL = "gemini-3.1-flash-lite"


class _Answer(BaseModel):
    answer: str


@dataclass
class BaselineState:
    context_block: str
    model: str


def _format_personas(personas: Personas) -> str:
    lines = [f"# Household: {personas.household_id}", personas.household_summary, "", "## Members"]
    for m in personas.members:
        lines.append(
            f"- **{m.display_name}** (`{m.user_id}`, age {m.age}, role={m.role}): "
            f"{m.background} Personality: {m.personality}"
        )
        if m.parental_controls:
            rules = ", ".join(m.parental_controls.get("rules", []))
            lines.append(f"  Parental controls: {rules}")
    lines.append("\n## Household relationships")
    for r in personas.household_relationships:
        lines.append(f"- {r.a} ↔ {r.b}: {r.type}{f' (since {r.since})' if r.since else ''}")
    if personas.external_relationships:
        lines.append("\n## External contacts")
        for e in personas.external_relationships:
            lines.append(f"- {e.user_id} → {e.name} ({e.relationship}): {e.context}")
    return "\n".join(lines)


def _format_inventory(seed: InventorySeed) -> str:
    lines = ["# Initial inventory state"]
    for owner, items in seed.items_by_owner.items():
        lines.append(f"\n## {owner}")
        for it in items:
            props = ", ".join(f"{k}={v}" for k, v in it.properties.items())
            lines.append(f"- [{it.item_type}] {it.label} ({props})")
    return "\n".join(lines)


def _format_sessions(sessions: list[Session]) -> str:
    parts = ["# Session history (chronological)"]
    for s in sessions:
        parts.append(
            f"\n## {s.session_id} — {s.timestamp.isoformat()} — "
            f"{s.user_id} on {s.device_context}"
        )
        parts.append(f"Topic: {s.topic}")
        for t in s.turns:
            who = "USER" if t.speaker == "user" else "ASSISTANT"
            parts.append(f"  {who}: {t.text}")
    return "\n".join(parts)


def ingest_sessions(
    *,
    personas: Personas,
    inventory_seed: InventorySeed,
    sessions: list[Session],
    model: str = DEFAULT_MODEL,
) -> BaselineState:
    """Build a single static context block from all the data.

    No memory system; everything that ever gets fed to the agent is in
    one big prompt. Big-context Gemini handles this fine for a
    benchmark this size; on a real product surface this strategy
    breaks the moment the data crosses the model's context window.
    """
    context_block = (
        _format_personas(personas)
        + "\n\n"
        + _format_inventory(inventory_seed)
        + "\n\n"
        + _format_sessions(sessions)
    )
    return BaselineState(context_block=context_block, model=model)


_PROMPT = """You are an AI assistant for the Razer customer-companion benchmark. You
have full context for the Okafor-Tan household (personas, inventory, full
session history) below. Answer the user's question concisely and accurately.

Be aware: the question is asked by `{ask_as_user}` on device `{ask_on_device}`.
For privacy-boundary questions (parental controls, gift surprises, "don't tell
my dad" type requests), reason about what THIS asker should be allowed to know
versus what's been kept private. For inventory questions, refer to the
structured inventory section. For relationship questions, refer to the
household-relationships section.

# CONTEXT

{context_block}

# QUESTION

{question}

Respond with JSON: {{"answer": "<your answer in 1-3 sentences>"}}.
"""


async def ask(
    *,
    state: BaselineState,
    qa: QA,
    api_key: str | None = None,
) -> str:
    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
    prompt = _PROMPT.format(
        ask_as_user=qa.ask_as_user,
        ask_on_device=qa.ask_on_device,
        context_block=state.context_block,
        question=qa.question.strip(),
    )
    resp = await client.aio.models.generate_content(
        model=state.model,
        contents=prompt,
        config=gt.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_Answer,
            temperature=0.0,
        ),
    )
    text = resp.text
    if text is None:
        return ""
    try:
        return _Answer.model_validate_json(text).answer
    except Exception as e:
        logger.warning("baseline.ask(%s) parse failed: %s", qa.qa_id, e)
        return ""
