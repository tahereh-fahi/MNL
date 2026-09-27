"""Load and validate the event catalog that defines framework scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TEMPORAL_TYPES = {"point", "interval", "sampled_state", "interval_censored", "derived"}
MATURITY_STATES = {"existing", "partial", "planned", "derived"}


def load_event_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Event catalog must contain a non-empty events list")

    codes: set[str] = set()
    for event in events:
        code = event.get("code")
        if not isinstance(code, str) or not code:
            raise ValueError("Every event definition needs a code")
        if code in codes:
            raise ValueError(f"Duplicate event code: {code}")
        codes.add(code)
        if event.get("temporal_type") not in TEMPORAL_TYPES:
            raise ValueError(f"Invalid temporal_type for {code}")
        if event.get("automation_maturity") not in MATURITY_STATES:
            raise ValueError(f"Invalid automation_maturity for {code}")
    return payload

