"""Forward-compatible adapters for SDK calls not yet on the typed surface.

Mirrors the same pattern used in ``sonzai-python/benchmarks/common/sdk_extras.py``:
prefer the native SDK binding when available, fall back to the raw transport
when not. Once the SDK regenerates with these as first-class methods, the
fallbacks become dead code we delete.
"""

from __future__ import annotations

from typing import Any

import asyncio
import logging

from sonzai import AsyncSonzai

logger = logging.getLogger("sonzai_razer_bench.backends._compat")

# Sessions + Inventory live on the customization layer; Mood is auto-generated.
# Try the customization path first (so customizations win), fall back to the
# raw _generated module for resources without a custom layer.
try:
    from sonzai.resources.inventory import AsyncInventory
except ImportError:  # pragma: no cover
    from sonzai._generated.resources.inventory import AsyncInventory
try:
    from sonzai.resources.sessions import AsyncSessions
except ImportError:  # pragma: no cover
    from sonzai._generated.resources.sessions import AsyncSessions
try:
    from sonzai.resources.mood import AsyncMood
except ImportError:
    from sonzai._generated.resources.mood import AsyncMood
try:
    from sonzai.resources.memory import AsyncMemory
except ImportError:  # pragma: no cover
    from sonzai._generated.resources.memory import AsyncMemory
try:
    from sonzai.resources.workbench import AsyncWorkbench
except ImportError:  # pragma: no cover
    from sonzai._generated.resources.workbench import AsyncWorkbench
try:
    from sonzai.resources.personality import AsyncPersonality
except ImportError:  # pragma: no cover
    from sonzai._generated.resources.personality import AsyncPersonality


def async_sessions(client: AsyncSonzai) -> AsyncSessions:
    existing = getattr(client, "sessions", None)
    if existing is not None:
        return existing
    return AsyncSessions(client._http)  # type: ignore[attr-defined]


def async_mood(client: AsyncSonzai) -> AsyncMood:
    existing = getattr(client, "mood", None)
    if existing is not None:
        return existing
    return AsyncMood(client._http)  # type: ignore[attr-defined]


def async_inventory(client: AsyncSonzai) -> AsyncInventory:
    existing = getattr(client, "inventory", None)
    if existing is not None:
        return existing
    return AsyncInventory(client._http)  # type: ignore[attr-defined]


def async_memory(client: AsyncSonzai) -> AsyncMemory:
    existing = getattr(client, "memory", None)
    if existing is not None:
        return existing
    return AsyncMemory(client._http)  # type: ignore[attr-defined]


def async_personality(client: AsyncSonzai) -> AsyncPersonality:
    existing = getattr(client, "personality", None)
    if existing is not None:
        return existing
    return AsyncPersonality(client._http)  # type: ignore[attr-defined]


# /agents/{id}/personality/recent-shifts has no codegen'd binding (the
# generator skipped this endpoint). Call via the raw transport so we
# capture trait shifts the bench needs for the personality-drift
# trajectory.
async def get_personality_recent_shifts_async(
    client: AsyncSonzai, *, agent_id: str
) -> dict:
    try:
        resp = await client._http.get(  # type: ignore[attr-defined]
            f"/api/v1/agents/{agent_id}/personality/recent-shifts"
        )
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            return resp.model_dump()  # type: ignore[no-any-return]
        return {}
    except Exception as e:
        logger.debug("recent-shifts failed: %s", e)
        return {}


def async_workbench(client: AsyncSonzai):
    """Return whatever workbench accessor the SDK has (custom or generated).

    The hand-written customization layer's ``Workbench`` exposes the
    documented ``advance_time(...)``; the auto-generated one only exposes
    ``workbench_advance_time()`` with no body. We prefer the hand-written
    when available.
    """
    existing = getattr(client, "workbench", None)
    if existing is not None:
        return existing
    return AsyncWorkbench(client._http)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# memory.reset — wipe per-(agent, user) memory partitions before a fresh run.
# Without this, re-running the bench accumulates fact duplicates from prior
# ingests; chat retrieval starts pulling from the polluted set.
# ---------------------------------------------------------------------------


_RESET_WALL_CLOCK_TIMEOUT_S = 120.0


async def clear_agent_memory_async(
    client: AsyncSonzai,
    *,
    agent_id: str,
    user_id: str,
    instance_id: str | None = None,
) -> None:
    """Best-effort ``memory.reset(agent_id, user_id=...)``. Swallows known
    transport errors since the post-condition — "this user's partition is
    empty" — is satisfied either way.
    """
    mem = async_memory(client)
    try:
        await asyncio.wait_for(
            mem.reset(agent_id=agent_id, user_id=user_id, instance_id=instance_id),
            timeout=_RESET_WALL_CLOCK_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "memory.reset for agent=%s user=%s timed out after %ss — continuing",
            agent_id, user_id, _RESET_WALL_CLOCK_TIMEOUT_S,
        )
    except Exception as e:
        logger.debug("memory.reset (%s/%s) raised %s — continuing", agent_id, user_id, e)


# ---------------------------------------------------------------------------
# advance_time — fires CE consolidation / decay / diary / personality between
# sessions. Without this, every session ingest in a multi-session arc happens
# at the same wall-clock instant from CE's perspective, and consolidation
# never runs. Critical for benchmarks that simulate calendar time.
# ---------------------------------------------------------------------------


_ADVANCE_TIME_CHUNK_HOURS = 48.0  # bigger chunks for the Razer arc; gaps are days/weeks
_ADVANCE_TIME_MAX_ATTEMPTS = 3
_ADVANCE_TIME_BACKOFF_S = 4.0


async def advance_time_async(
    client: AsyncSonzai,
    *,
    agent_id: str,
    user_id: str,
    simulated_hours: float,
    simulated_base_offset_hours: float = 0.0,
    instance_id: str | None = None,
) -> dict:
    """Single-call advance-time. Posts to ``/workbench/advance-time`` with
    ``async=true`` and polls the job to bypass Cloudflare's 100s origin-read
    timeout — same pattern the existing sonzai-python bench uses.
    """
    body = {
        "agent_id": agent_id,
        "user_id": user_id,
        "simulated_hours": simulated_hours,
        "simulated_base_offset_hours": simulated_base_offset_hours,
        "instance_id": instance_id or "",
        "character_config": {},
        "async": True,
    }
    submit = await client._http.post(  # type: ignore[attr-defined]
        "/api/v1/workbench/advance-time", json_data=body
    )
    if isinstance(submit, dict):
        job_id = str(submit.get("job_id") or "")
    else:
        job_id = str(getattr(submit, "job_id", "") or "")
    if not job_id:
        # Server didn't accept async mode — fall back to direct sync (no poll).
        body.pop("async", None)
        sync_resp = await client._http.post(  # type: ignore[attr-defined]
            "/api/v1/workbench/advance-time", json_data=body
        )
        return sync_resp if isinstance(sync_resp, dict) else {}

    # Poll the job
    deadline_s = 30 * 60
    started = asyncio.get_event_loop().time()
    interval = 2.0
    while True:
        if asyncio.get_event_loop().time() - started > deadline_s:
            raise TimeoutError(f"advance-time job {job_id} did not finish")
        try:
            state = await client._http.get(  # type: ignore[attr-defined]
                f"/api/v1/workbench/advance-time/jobs/{job_id}",
            )
        except Exception:
            await asyncio.sleep(interval)
            interval = min(interval * 1.5, 15.0)
            continue
        status = (
            (state.get("status") if isinstance(state, dict) else getattr(state, "status", ""))
            or ""
        )
        if status == "succeeded":
            result = (
                state.get("result") if isinstance(state, dict) else getattr(state, "result", None)
            )
            return result if isinstance(result, dict) else {}
        if status == "failed":
            err = (
                state.get("error") if isinstance(state, dict) else getattr(state, "error", "")
            ) or "unknown"
            raise RuntimeError(f"advance-time job {job_id} failed: {err}")
        await asyncio.sleep(interval)
        interval = min(interval * 1.5, 15.0)


async def advance_time_chunked_async(
    client: AsyncSonzai,
    *,
    agent_id: str,
    user_id: str,
    total_hours: float,
    chunk_hours: float = _ADVANCE_TIME_CHUNK_HOURS,
    instance_id: str | None = None,
) -> list[dict]:
    """Break a long gap into ``chunk_hours`` pieces, run sequentially with
    per-chunk retry. Returns the list of successful chunk responses.

    Mirrors ``benchmarks/common/workbench_compat.advance_time_chunked_async``
    in sonzai-python — same chunking and retry behavior. We default the chunk
    size to 48h (vs 24h there) because Razer simulates real days/weeks
    between sessions and we'd rather make fewer calls.
    """
    if total_hours <= 0:
        return []
    responses: list[dict] = []
    elapsed = 0.0
    remaining = total_hours
    while remaining > 0.0:
        size = min(chunk_hours, remaining)
        got = False
        for attempt in range(1, _ADVANCE_TIME_MAX_ATTEMPTS + 1):
            try:
                r = await advance_time_async(
                    client,
                    agent_id=agent_id,
                    user_id=user_id,
                    simulated_hours=size,
                    simulated_base_offset_hours=elapsed,
                    instance_id=instance_id,
                )
                responses.append(r)
                got = True
                break
            except Exception as e:
                if attempt >= _ADVANCE_TIME_MAX_ATTEMPTS:
                    logger.warning(
                        "advance_time chunk failed (gave up after %d): hours=%.1f err=%s",
                        attempt, size, e,
                    )
                    break
                backoff = _ADVANCE_TIME_BACKOFF_S * (2 ** (attempt - 1))
                logger.debug(
                    "advance_time chunk attempt %d failed; retry in %.1fs: %s",
                    attempt, backoff, e,
                )
                await asyncio.sleep(backoff)
        elapsed += size
        remaining -= size
        if not got:
            # Drop the chunk; outer loop continues so we don't lose all progress.
            continue
    return responses


async def ensure_bench_agent_async(
    client: AsyncSonzai,
    *,
    name: str,
    description: str,
    gender: str = "",
    model: str | None = None,
) -> tuple[str, bool]:
    """Idempotent ``generate-and-create`` for a stable benchmark agent.

    Returns ``(agent_id, existed_before)``. First call with a given ``name``
    spends one LLM call to expand the description into a full profile;
    subsequent calls return the same agent for free.
    """
    body: dict[str, Any] = {"name": name, "description": description}
    if gender:
        body["gender"] = gender
    if model:
        body["model"] = model
    resp = await client._http.post(  # type: ignore[attr-defined]
        "/api/v1/agents/generate-and-create", json_data=body
    )
    if not isinstance(resp, dict):
        try:
            resp = resp.model_dump()  # type: ignore[attr-defined]
        except Exception as e:
            raise RuntimeError(
                f"generate-and-create returned unexpected type {type(resp)}"
            ) from e
    agent_id = str(resp.get("agent_id") or "")
    if not agent_id:
        raise RuntimeError(
            f"generate-and-create response missing agent_id: keys={list(resp.keys())}"
        )
    return agent_id, bool(resp.get("existing"))
