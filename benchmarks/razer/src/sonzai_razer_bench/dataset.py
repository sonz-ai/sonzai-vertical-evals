"""Dataset loaders for the Razer customer-companion benchmark.

Loads the four data files (personas, inventory_seed, sessions, qa) into
typed dataclasses. The harness uses these for both ingest (sessions →
backend memory) and evaluation (qa → ask-and-judge).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Member:
    user_id: str
    display_name: str
    age: int
    role: str
    occupation: str
    background: str
    personality: str
    primary_devices: tuple[str, ...]
    spending_authority: str
    monthly_razer_gold_budget: float
    parental_controls: dict | None = None


@dataclass(frozen=True)
class HouseholdRel:
    a: str
    b: str
    type: str
    since: str = ""


@dataclass(frozen=True)
class ExternalRel:
    user_id: str
    name: str
    relationship: str
    context: str


@dataclass
class Personas:
    household_id: str
    household_summary: str
    members: list[Member]
    household_relationships: list[HouseholdRel]
    external_relationships: list[ExternalRel]

    def member(self, user_id: str) -> Member:
        for m in self.members:
            if m.user_id == user_id:
                return m
        raise KeyError(f"unknown user_id: {user_id!r}")


@dataclass(frozen=True)
class InventoryItem:
    item_id: str
    item_type: str
    label: str
    properties: dict


@dataclass
class InventorySeed:
    items_by_owner: dict[str, list[InventoryItem]]


@dataclass(frozen=True)
class Turn:
    speaker: str  # "user" or "assistant"
    text: str


@dataclass(frozen=True)
class Session:
    session_id: str
    user_id: str
    device_context: str
    timestamp: datetime
    topic: str
    summary: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class QA:
    qa_id: str
    category: str
    ask_as_user: str
    ask_on_device: str
    question: str
    gold_answer: str
    evidence_sessions: tuple[str, ...]
    must_mention: tuple[str, ...] = ()
    must_mention_one_of: tuple[str, ...] = ()
    must_not_mention: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_personas(*, path: Path | None = None) -> Personas:
    p = path or (DATA_DIR / "personas.json")
    with open(p) as f:
        data = json.load(f)
    members = [
        Member(
            user_id=m["user_id"],
            display_name=m["display_name"],
            age=int(m["age"]),
            role=m["role"],
            occupation=m["occupation"],
            background=m["background"],
            personality=m["personality"],
            primary_devices=tuple(m.get("primary_devices", [])),
            spending_authority=m.get("spending_authority", "unrestricted"),
            monthly_razer_gold_budget=float(m.get("monthly_razer_gold_budget", 0.0)),
            parental_controls=m.get("parental_controls"),
        )
        for m in data["members"]
    ]
    rels = [HouseholdRel(**r) for r in data.get("household_relationships", [])]
    ext = [ExternalRel(**r) for r in data.get("external_relationships", [])]
    return Personas(
        household_id=data["household_id"],
        household_summary=data["household_summary"],
        members=members,
        household_relationships=rels,
        external_relationships=ext,
    )


def load_inventory_seed(*, path: Path | None = None) -> InventorySeed:
    p = path or (DATA_DIR / "inventory_seed.json")
    with open(p) as f:
        data = json.load(f)
    out: dict[str, list[InventoryItem]] = {}
    for owner, items in data.get("items_by_owner", {}).items():
        out[owner] = [
            InventoryItem(
                item_id=i["item_id"],
                item_type=i["item_type"],
                label=i["label"],
                properties=dict(i.get("properties", {})),
            )
            for i in items
        ]
    return InventorySeed(items_by_owner=out)


def load_sessions(*, path: Path | None = None) -> list[Session]:
    p = path or (DATA_DIR / "sessions.json")
    with open(p) as f:
        data = json.load(f)
    out: list[Session] = []
    for s in data["sessions"]:
        ts = s["timestamp"]
        try:
            parsed = datetime.fromisoformat(ts)
        except ValueError:
            parsed = datetime.min
        turns = tuple(
            Turn(speaker=t["speaker"], text=t["text"]) for t in s.get("turns", [])
        )
        out.append(
            Session(
                session_id=s["session_id"],
                user_id=s["user_id"],
                device_context=s.get("device_context", ""),
                timestamp=parsed,
                topic=s.get("topic", ""),
                summary=s.get("summary", ""),
                turns=turns,
            )
        )
    out.sort(key=lambda x: x.timestamp)
    return out


def load_qa(*, path: Path | None = None) -> list[QA]:
    p = path or (DATA_DIR / "qa.json")
    with open(p) as f:
        data = json.load(f)
    return [
        QA(
            qa_id=q["qa_id"],
            category=q["category"],
            ask_as_user=q["ask_as_user"],
            ask_on_device=q["ask_on_device"],
            question=q["question"],
            gold_answer=q["gold_answer"],
            evidence_sessions=tuple(q.get("evidence_sessions", [])),
            must_mention=tuple(q.get("must_mention", [])),
            must_mention_one_of=tuple(q.get("must_mention_one_of", [])),
            must_not_mention=tuple(q.get("must_not_mention", [])),
        )
        for q in data["qa"]
    ]
