"""Stateless RAG baseline.

Companion to :mod:`backends.baseline` (stateless-stuff). Same model, same
prompt scaffolding, but instead of dumping every session into the context
window we embed each session once at ingest time, embed the question at
ask time, and feed the model only the top-K most similar sessions plus
the personas + inventory seed.

Why both baselines: ``baseline`` is the *upper bound* of what context-
stuffing can achieve (no retrieval errors possible — everything is in
the prompt). ``stateless-rag`` is what you get with the cheapest off-
the-shelf RAG. Sonzai's lift over ``baseline`` proves the structured
APIs matter; Sonzai's lift over ``stateless-rag`` proves consolidation
matters.

No new dependencies — uses the same ``google-genai`` SDK already used
by every other backend, and a tiny dot-product implementation in pure
Python.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types as gt
from pydantic import BaseModel

from ..dataset import (
    InventorySeed,
    Personas,
    QA,
    Session,
)
from .baseline import _format_inventory, _format_personas

logger = logging.getLogger("sonzai_razer_bench.backends.stateless_rag")

DEFAULT_MODEL = "gemini-3.1-flash-lite"
DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_TOP_K = 5


class _Answer(BaseModel):
    answer: str


@dataclass
class _Doc:
    """One retrievable unit — session_id, header, body, embedding."""
    doc_id: str
    header: str
    body: str
    embedding: list[float] = field(default_factory=list)


@dataclass
class StatelessRagState:
    personas_block: str
    inventory_block: str
    docs: list[_Doc]
    model: str
    embed_model: str
    top_k: int


def _format_session_doc(s: Session) -> tuple[str, str]:
    header = (
        f"{s.session_id} — {s.timestamp.isoformat()} — "
        f"{s.user_id} on {s.device_context} — {s.topic}"
    )
    lines = [f"Topic: {s.topic}", f"Summary: {s.summary}"]
    for t in s.turns:
        who = "USER" if t.speaker == "user" else "ASSISTANT"
        lines.append(f"  {who}: {t.text}")
    return header, "\n".join(lines)


def _embed_one(client: genai.Client, model: str, text: str) -> list[float]:
    resp = client.models.embed_content(model=model, contents=text)
    embeddings = getattr(resp, "embeddings", None) or []
    if not embeddings:
        return []
    first = embeddings[0]
    values = getattr(first, "values", None)
    if values is None and isinstance(first, dict):
        values = first.get("values")
    return list(values or [])


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def ingest_sessions(
    *,
    personas: Personas,
    inventory_seed: InventorySeed,
    sessions: list[Session],
    model: str = DEFAULT_MODEL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    top_k: int = DEFAULT_TOP_K,
    api_key: str | None = None,
) -> StatelessRagState:
    """Embed every session once. Personas + inventory are always-prepended,
    not retrieved — they're small and questions almost always need them.
    """
    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    docs: list[_Doc] = []
    for s in sessions:
        header, body = _format_session_doc(s)
        emb = _embed_one(client, embed_model, f"{header}\n{body}")
        docs.append(_Doc(doc_id=s.session_id, header=header, body=body, embedding=emb))

    return StatelessRagState(
        personas_block=_format_personas(personas),
        inventory_block=_format_inventory(inventory_seed),
        docs=docs,
        model=model,
        embed_model=embed_model,
        top_k=top_k,
    )


def _retrieve(state: StatelessRagState, query_emb: list[float]) -> list[_Doc]:
    scored = [(_cosine(query_emb, d.embedding), d) for d in state.docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[: state.top_k]]


_PROMPT = """You are an AI assistant for the Razer customer-companion benchmark. You
have the household personas, an initial inventory snapshot, and a *retrieved
subset* of session transcripts (selected by similarity to the question).
Answer concisely and accurately using only the provided context.

Be aware: the question is asked by `{ask_as_user}` on device `{ask_on_device}`.
For privacy-sensitive questions, reason about what THIS asker should be
allowed to know based on what's in the retrieved context.

# PERSONAS

{personas_block}

# INVENTORY (initial state — sessions may have changed it)

{inventory_block}

# RETRIEVED SESSIONS (top-{k} by similarity to the question)

{retrieved_block}

# QUESTION

{question}

Respond with JSON: {{"answer": "<your answer in 1-3 sentences>"}}.
"""


async def ask(
    *,
    state: StatelessRagState,
    qa: QA,
    api_key: str | None = None,
) -> str:
    client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    # Embedding call is sync in google-genai; offload to a thread so we
    # don't block the async QA loop.
    query = (
        f"Asked by {qa.ask_as_user} on {qa.ask_on_device}: {qa.question.strip()}"
    )
    query_emb = await asyncio.to_thread(
        _embed_one, client, state.embed_model, query
    )
    if not query_emb:
        logger.warning("stateless-rag: empty query embedding for %s", qa.qa_id)

    hits = _retrieve(state, query_emb)
    retrieved_block = "\n\n".join(
        f"## {d.header}\n{d.body}" for d in hits
    ) or "(no sessions retrieved)"

    prompt = _PROMPT.format(
        ask_as_user=qa.ask_as_user,
        ask_on_device=qa.ask_on_device,
        personas_block=state.personas_block,
        inventory_block=state.inventory_block,
        k=state.top_k,
        retrieved_block=retrieved_block,
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
        logger.warning("stateless_rag.ask(%s) parse failed: %s", qa.qa_id, e)
        return ""
