"""MemPalace backend for the Razer customer-companion benchmark.

Pairs MemPalace's verbatim retrieval with Gemini Flash Lite for agent-turn
generation — same Gemini model the Sonzai backend's chat handler uses
internally — so the head-to-head comparison isolates the *memory layer*,
not the LLM.

Per-user palace + convos:

- ``<bench_root>/palaces/<household>/<user>/`` — ChromaDB drawers
- ``<bench_root>/convos/<household>/<user>/session_NNN.md`` — MD transcripts
  in MemPalace's ``> user``/reply convention so ``chunk_exchanges`` pairs
  them into exchange-level drawers.

Phase 1 (live conversation):

1. For each scripted user turn, search the *speaking user's* palace with
   the user-line as the query (top-K drawers).
2. Inject the retrieved drawers as context into a Gemini ``agent_turn_async``
   prompt; capture the response.
3. After each session, append turns to ``session_NNN.md`` and call
   ``mine_convos`` on that user's convos dir to refresh their palace.

Phase 2 (QA):

- For each QA, search ``ask_as_user``'s palace with the question, format
  drawers into the prompt, ask Gemini for an answer.

Note: MemPalace has no shared agent identity, so cross-user privacy
(e.g. "don't tell Aiden about Priya's Father's Day plan") is *structurally*
not enforceable in this backend — each user's palace is self-contained.
That's the demo: Sonzai's shared-agent-with-per-user-permissions story has
no analogue in raw retrieval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types as gt
from pydantic import BaseModel, Field

from ..dataset import (
    InventorySeed,
    Personas,
    QA,
    Session,
)

logger = logging.getLogger("sonzai_razer_bench.backends.mempalace")

DEFAULT_SEARCH_K = 5
DEFAULT_MAX_DRAWER_CHARS = 600
DEFAULT_GEN_MODEL = "gemini-3.1-flash-lite"


# ---------------------------------------------------------------------------
# MemPalace import — sibling checkout convention
# ---------------------------------------------------------------------------


def _import_mempalace():
    try:
        from mempalace import convo_miner, searcher  # type: ignore
        return convo_miner, searcher
    except ImportError as e:
        # Walk up to sonzai-sdk/ and look for a sibling 'mempalace' clone.
        # backends/mempalace.py
        #   parents[0] = backends/
        #   parents[1] = sonzai_razer_bench/
        #   parents[2] = src/
        #   parents[3] = razer/
        #   parents[4] = benchmarks/
        #   parents[5] = sonzai-vertical-evals/
        #   parents[6] = sonzai-sdk/   ← sibling 'mempalace' lives here
        sibling = Path(__file__).resolve().parents[6] / "mempalace"
        if sibling.exists():
            sys.path.insert(0, str(sibling))
            from mempalace import convo_miner, searcher  # type: ignore
            return convo_miner, searcher
        raise RuntimeError(
            "mempalace package not importable and no sibling checkout at "
            f"{sibling}. Either `pip install mempalace` or clone the repo "
            "alongside sonzai-vertical-evals."
        ) from e


_convo_miner, _searcher = _import_mempalace()


# ---------------------------------------------------------------------------
# Palace paths
# ---------------------------------------------------------------------------


def _bench_root() -> Path:
    override = os.environ.get("SONZAI_BENCH_CACHE")
    base = Path(override).expanduser() if override else Path.home() / ".cache" / "sonzai-bench"
    return base / "razer-mempalace"


def _user_paths(household_id: str, user_id: str) -> tuple[Path, Path]:
    safe_h = re.sub(r"[^a-zA-Z0-9_-]", "_", household_id)[:64]
    safe_u = re.sub(r"[^a-zA-Z0-9_-]", "_", user_id)[:64]
    root = _bench_root()
    palace = root / "palaces" / safe_h / safe_u
    convos = root / "convos" / safe_h / safe_u
    palace.mkdir(parents=True, exist_ok=True)
    convos.mkdir(parents=True, exist_ok=True)
    # Pre-create the drawers collection so first-session searches don't
    # spam "collection does not exist" warnings.
    try:
        from mempalace.palace import get_collection  # type: ignore
        get_collection(str(palace), create=True)
    except Exception as e:
        logger.debug("pre-create palace failed (%s/%s): %s", safe_h, safe_u, e)
    return palace, convos


# ---------------------------------------------------------------------------
# Memory formatting + retrieval
# ---------------------------------------------------------------------------


def _format_retrieved(hits: list[dict[str, Any]], max_chars: int) -> str:
    if not hits:
        return ""
    lines: list[str] = []
    for i, h in enumerate(hits, 1):
        text = (h.get("text") or "").strip()
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "…"
        sim = h.get("similarity")
        header = f"[{i}]" + (f" (sim={sim})" if isinstance(sim, (int, float)) else "")
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


async def _search_async(query: str, palace_path: Path, n_results: int):
    def _run():
        try:
            return _searcher.search_memories(
                query=query, palace_path=str(palace_path), n_results=n_results
            )
        except Exception as e:
            logger.debug("mempalace search failed: %s", e)
            return {"results": []}

    result = await asyncio.to_thread(_run)
    if isinstance(result, dict) and result.get("results"):
        return list(result["results"])
    return []


async def _mine_async(convos_dir: Path, palace_path: Path) -> None:
    def _run():
        try:
            _convo_miner.mine_convos(
                str(convos_dir),
                str(palace_path),
                wing="razer",
                agent="mempalace",
                extract_mode="exchange",
            )
        except Exception as e:
            logger.warning("mempalace mine_convos failed: %s", e)

    await asyncio.to_thread(_run)


def _write_session_md(
    convos_dir: Path, session_index: int, transcript: list[dict[str, str]]
) -> Path:
    """Write a transcript in MemPalace's ``> user``/reply convention."""
    path = convos_dir / f"session_{session_index:03d}.md"
    lines: list[str] = [f"# Session {session_index}", ""]
    for turn in transcript:
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role = turn.get("role", "user")
        if role == "user":
            lines.append(f"> {content}")
        else:
            lines.append(content)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Gemini generation — agent role + answer role
# ---------------------------------------------------------------------------


_AGENT_TURN_PROMPT = """You are the Razer AI desk companion talking to {user_name} (user_id={user_id}, age {age}).
Stay in character; this is a real consumer-product conversation. Acknowledge
parental controls and privacy boundaries when relevant.

# Conversation context (this session)
{transcript}

# What you remember about {user_name} (retrieved from your memory)
{retrieved}

# {user_name}'s latest message
{user_text}

Reply naturally, 1–3 sentences. If retrieved memory contradicts what was just
said, prefer the most recent / specific signal. If the message asks about
something you don't have memory of, say so honestly rather than making it up.

Respond with JSON: {{"reply": "<your reply>"}}.
"""


_QA_ANSWER_PROMPT = """You are the Razer AI desk companion answering a question for {user_name}
(user_id={user_id}, age {age}, asking on {device_context}).

# Retrieved memories (top-{k} drawers from this user's palace)
{retrieved}

# Question
{question}

Answer concisely (1–3 sentences). Use only the retrieved memories — do not
fabricate. If the answer requires information you don't have, say so.

For privacy-sensitive questions (parental controls, gift surprises, "don't
tell my dad" style), reason about what the asking user should be allowed to
know based on what's in your retrieved memory.

Respond with JSON: {{"answer": "<your answer>"}}.
"""


class _Reply(BaseModel):
    reply: str


class _Answer(BaseModel):
    answer: str = Field(default="")


async def _gemini_agent_turn(
    *,
    client: genai.Client,
    model: str,
    user_name: str,
    user_id: str,
    age: int,
    transcript_text: str,
    retrieved: str,
    user_text: str,
) -> str:
    prompt = _AGENT_TURN_PROMPT.format(
        user_name=user_name,
        user_id=user_id,
        age=age,
        transcript=transcript_text or "(start of session)",
        retrieved=retrieved or "(no relevant prior memory)",
        user_text=user_text,
    )
    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gt.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_Reply,
            temperature=0.4,
        ),
    )
    if not resp.text:
        return ""
    try:
        return _Reply.model_validate_json(resp.text).reply
    except Exception as e:
        logger.warning("agent_turn parse failed: %s", e)
        return ""


async def _gemini_qa_answer(
    *,
    client: genai.Client,
    model: str,
    user_name: str,
    user_id: str,
    age: int,
    device_context: str,
    retrieved: str,
    k: int,
    question: str,
) -> str:
    prompt = _QA_ANSWER_PROMPT.format(
        user_name=user_name,
        user_id=user_id,
        age=age,
        device_context=device_context or "(unspecified)",
        retrieved=retrieved or "(no relevant prior memory)",
        k=k,
        question=question,
    )
    resp = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=gt.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_Answer,
            temperature=0.0,
        ),
    )
    if not resp.text:
        return ""
    try:
        return _Answer.model_validate_json(resp.text).answer
    except Exception as e:
        logger.warning("qa_answer parse failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# State + entry points (mirrors the sonzai backend's surface)
# ---------------------------------------------------------------------------


@dataclass
class MempalaceBackendState:
    household_id: str
    user_palace: dict[str, Path] = field(default_factory=dict)
    user_convos: dict[str, Path] = field(default_factory=dict)
    user_meta: dict[str, dict] = field(default_factory=dict)
    gen_model: str = DEFAULT_GEN_MODEL
    search_k: int = DEFAULT_SEARCH_K
    max_drawer_chars: int = DEFAULT_MAX_DRAWER_CHARS


def init_backend(
    *,
    personas: Personas,
    inventory_seed: InventorySeed,
    gen_model: str = DEFAULT_GEN_MODEL,
    search_k: int = DEFAULT_SEARCH_K,
    max_drawer_chars: int = DEFAULT_MAX_DRAWER_CHARS,
) -> MempalaceBackendState:
    """Set up a per-user palace + convos dir; seed inventory as MD facts.

    Inventory seed becomes a synthetic ``session_000.md`` file per user — one
    line per item — so MemPalace's miner picks up "Marcus owns the BlackShark"
    as a drawer alongside the conversational drawers.
    """
    household_id = "the-okafor-tan-household"
    state = MempalaceBackendState(
        household_id=household_id,
        gen_model=gen_model,
        search_k=search_k,
        max_drawer_chars=max_drawer_chars,
    )
    for m in personas.members:
        palace, convos = _user_paths(household_id, m.user_id)
        state.user_palace[m.user_id] = palace
        state.user_convos[m.user_id] = convos
        state.user_meta[m.user_id] = {
            "display_name": m.display_name,
            "age": m.age,
        }

        # Seed inventory as session_000.md
        items = inventory_seed.items_by_owner.get(m.user_id, [])
        if items:
            seed_md = convos / "session_000.md"
            lines = [f"# Inventory seed for {m.display_name}", ""]
            for it in items:
                props = ", ".join(f"{k}={v}" for k, v in it.properties.items())
                lines.append(f"> {m.display_name} owns: {it.label} ({it.item_type}; {props})")
                lines.append(
                    f"Confirmed item registered to {m.display_name}: "
                    f"{it.label} — type {it.item_type}, properties {props}."
                )
                lines.append("")
            seed_md.write_text("\n".join(lines), encoding="utf-8")

    return state


async def simulate_session(
    *,
    state: MempalaceBackendState,
    session: Session,
    judge_client: genai.Client | None = None,  # unused here; kept for API parity
    judge_model: str = DEFAULT_GEN_MODEL,
    judge_each_turn: bool = False,
    server_session_id: str | None = None,
    user_display_name: str | None = None,
    user_age: int | None = None,
) -> dict:
    """Run one session as a live conversation against MemPalace + Gemini.

    Returns a session-record dict matching the schema the run.py orchestrator
    serializes (matches simulator.SessionRecord but without mood — MemPalace
    has no mood model).
    """
    palace = state.user_palace[session.user_id]
    convos = state.user_convos[session.user_id]
    meta = state.user_meta.get(session.user_id, {})
    display_name = user_display_name or meta.get("display_name", session.user_id)
    age = user_age if user_age is not None else int(meta.get("age", 0))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    # session index — count existing session_NNN.md files
    sidx = sum(1 for _ in convos.glob("session_*.md"))

    # Walk turns: for each user turn, retrieve from palace + generate reply.
    transcript: list[dict[str, str]] = []
    record_turns: list[dict[str, Any]] = []
    pending_user: str | None = None
    pending_ref: str = ""
    turn_index = 0

    async def _flush(user_text: str, reference_text: str) -> None:
        nonlocal turn_index
        # Render this-session transcript so far
        ts_text = "\n".join(
            f"{'AGENT' if t['role'] == 'assistant' else 'USER'}: {t['content']}"
            for t in transcript
        )
        # Search + generate
        hits = await _search_async(user_text, palace, state.search_k)
        retrieved = _format_retrieved(hits, state.max_drawer_chars)
        agent_text = await _gemini_agent_turn(
            client=client,
            model=state.gen_model,
            user_name=display_name,
            user_id=session.user_id,
            age=age,
            transcript_text=ts_text,
            retrieved=retrieved,
            user_text=user_text,
        )
        transcript.append({"role": "user", "content": user_text})
        transcript.append({"role": "assistant", "content": agent_text})
        record_turns.append(
            {
                "turn_index": turn_index,
                "user_text": user_text,
                "reference_text": reference_text,
                "agent_text": agent_text,
                "mood_before": {},
                "mood_after": {},
                "judge_score": 0.0,
                "judge_in_character": False,
                "judge_handled_safely": True,
                "judge_made_natural_callback": False,
                "judge_rationale": "",
            }
        )
        turn_index += 1

    for t in session.turns:
        if t.speaker == "user":
            if pending_user is not None:
                await _flush(pending_user, "")
            pending_user = t.text
            pending_ref = ""
        else:
            if pending_user is not None:
                await _flush(pending_user, t.text)
                pending_user = None
                pending_ref = ""
    if pending_user is not None:
        await _flush(pending_user, pending_ref)

    # Persist transcript + mine into palace.
    sidx += 1
    _write_session_md(convos, sidx, transcript)
    await _mine_async(convos, palace)

    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "user_display_name": display_name,
        "device_context": session.device_context,
        "timestamp": session.timestamp.isoformat() if hasattr(session.timestamp, "isoformat") else str(session.timestamp),
        "topic": session.topic,
        "summary": session.summary,
        "turns": record_turns,
    }


async def simulate_all_sessions(
    *,
    state: MempalaceBackendState,
    sessions: list[Session],
    user_display_names: dict[str, str],
    user_ages: dict[str, int],
    **kwargs: Any,
) -> list[dict]:
    """Drive every session sequentially (palace updates can't safely interleave)."""
    out: list[dict] = []
    for s in sessions:
        out.append(
            await simulate_session(
                state=state,
                session=s,
                user_display_name=user_display_names.get(s.user_id, s.user_id),
                user_age=user_ages.get(s.user_id, 0),
            )
        )
    return out


def _qualified_question(qa: QA) -> str:
    asker = qa.ask_as_user
    if asker == "household-anon":
        return (
            "(Asked anonymously at the shared desk companion in the family room — "
            "the speaker has not been identified. Answer with what you know about "
            "the household; do not pretend to know who's speaking.)\n\n"
            f"{qa.question}"
        )
    return f"(Asked by {asker} on {qa.ask_on_device}.)\n\n{qa.question}"


async def ask(
    *,
    state: MempalaceBackendState,
    qa: QA,
) -> str:
    """Answer one QA via mempalace search + Gemini answer."""
    user_id = qa.ask_as_user
    if user_id not in state.user_palace:
        logger.warning("ask: unknown user %r — returning empty", user_id)
        return ""
    meta = state.user_meta.get(user_id, {})
    palace = state.user_palace[user_id]

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    hits = await _search_async(qa.question, palace, state.search_k)
    retrieved = _format_retrieved(hits, state.max_drawer_chars)
    return await _gemini_qa_answer(
        client=client,
        model=state.gen_model,
        user_name=meta.get("display_name", user_id),
        user_id=user_id,
        age=int(meta.get("age", 0)),
        device_context=qa.ask_on_device,
        retrieved=retrieved,
        k=state.search_k,
        question=_qualified_question(qa),
    )
