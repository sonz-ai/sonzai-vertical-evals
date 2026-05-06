"""Razer customer-companion benchmark for the Sonzai Mind Layer.

Models a 4-person Razer household (Marcus, Priya, Aiden, Lila) sharing an
AI desk companion, peripherals, a workstation PC, and personal devices
across 18 sessions and 6 simulated months. Probes 8 capability categories
(single-hop, multi-hop, temporal, inventory, kb-relationship,
multi-user-disambiguation, privacy-boundary, cross-device-continuity,
adversarial) via 30 hand-authored QA pairs with gold answers and an LLM
judge.

Invoke via::

    python -m sonzai_razer_bench --backend sonzai
"""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path


def _load_dotenv_if_present() -> None:
    """Walk up looking for a .env, populate os.environ. Shell wins over file.

    Mirrors the convention used by sonzai-python/benchmarks so credentials
    set once in the sibling SDK checkout work here too.
    """
    here = _Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        dotenv = candidate / ".env"
        if dotenv.is_file():
            for raw in dotenv.read_text().splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in _os.environ:
                    _os.environ[key] = value
            return


_load_dotenv_if_present()
